from __future__ import annotations

import argparse
import ast
import concurrent.futures
import json
import multiprocessing as mp
import os
import random
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch


def normalize_op(raw: str) -> str | None:
    op = raw.strip().strip("`'\" ")
    if not op:
        return None
    if op.startswith("torch.nn.modules."):
        leaf = op.rsplit(".", maxsplit=1)[-1]
        if hasattr(torch.nn, leaf):
            return f"torch.nn.{leaf}"
    if op.startswith("nn.modules."):
        leaf = op.rsplit(".", maxsplit=1)[-1]
        if hasattr(torch.nn, leaf):
            return f"torch.nn.{leaf}"
    if op.startswith("torch."):
        return op
    if op.startswith("nn."):
        return "torch.nn." + op.removeprefix("nn.")
    if op.startswith("F."):
        return "torch.nn.functional." + op.removeprefix("F.")
    if op.startswith("functional."):
        return "torch.nn.functional." + op.removeprefix("functional.")
    if op.startswith("transformers."):
        return op
    return op


CODE_BLOCK_RE = re.compile(r"```(?:python|py)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


@dataclass
class LLMConfig:
    api_url: str
    api_key: str
    model: str
    temperature: float
    max_tokens: int
    timeout: int


@dataclass
class FilterConfig:
    device: str
    min_eager_ms: float
    max_eager_ms: float
    warmup: int
    repeat: int
    atol: float
    rtol: float
    require_compile: bool


def original_op_name(op: str) -> str:
    if op.startswith("torch.nn.functional."):
        return "F." + op.removeprefix("torch.nn.functional.")
    if op.startswith("torch.nn."):
        return "nn." + op.removeprefix("torch.nn.")
    return op


def canonical_op_name(op: str) -> str:
    normalized = normalize_op(op)
    return normalized or op


def load_ops(path: Path) -> list[str]:
    df = pd.read_csv(path)
    ops = sorted({str(op) for op in df["op"]})
    blocked_prefixes = ("torch._", "torch.Generator")
    return [op for op in ops if not op.startswith(blocked_prefixes)]


def weighted_sample_ops(rng: random.Random, ops: list[str], min_ops: int, max_ops: int) -> list[str]:
    nn_ops = [op for op in ops if op.startswith("torch.nn.") and not op.startswith("torch.nn.functional.")]
    fn_ops = [op for op in ops if op.startswith("torch.nn.functional.")]
    torch_ops = [op for op in ops if op.startswith("torch.") and not op.startswith("torch.nn.")]
    count = rng.randint(min_ops, max_ops)
    selected: list[str] = []
    pools = [nn_ops, fn_ops, torch_ops]
    weights = [0.45, 0.25, 0.30]
    for _ in range(count):
        pool = rng.choices(pools, weights=weights, k=1)[0]
        if pool:
            selected.append(rng.choice(pool))
    return sorted(set(selected), key=selected.index)


def build_prompt(ops: list[str], sample_id: int) -> str:
    ops_for_dataset = [original_op_name(op) for op in ops]
    return f"""生成一个 CUDA-Agent 格式 PyTorch 模型样本。最终代码用 ```python 代码块输出。

必须尽量使用这些 ops：
{json.dumps(ops_for_dataset, ensure_ascii=False)}

硬性要求：
- 不要输出 JSON。
- 最终代码用```python 代码块输出。
- 代码必须包含 import torch、class Model、get_inputs()、get_init_inputs()。
- 必须能直接运行：Model(*get_init_inputs())(*get_inputs())。
- 输出必须依赖输入，不能全零/常量，不能 NaN/Inf。
- 可以使用复杂模块，但 shape 必须自洽；复杂 op 难以组合时，至少使用给定 ops 中的大部分。
- 不写自定义 CUDA/Triton，只写 PyTorch。
- get_inputs() 可以用 torch.randn/torch.randint 构造输入，但 forward 的目标 ops 不能是随机/配置/init/容器 API。
- 输入规模参考 CUDA Agent 数据，不要写 toy shape：
  * 2D CNN: batch_size 16-128, channels 16-128, height/width 128-1024。
  * 1D/sequence: batch_size 32-512, length 1024-16384, channels 16-256。
  * Transformer/attention: batch_size 8-64, seq_len 256-2048, embed_dim 128-1024, num_heads 4/8/16。
  * Linear/MLP: batch_size 128-2048, in_features/hidden/out_features 256-4096。
  * Embedding: batch_size 64-512, seq_len 128-2048, embedding_dim 128-1024。
  * 3D ops: batch_size 4-32, channels 8-64, depth/height/width 16-128。
  选择能在 CUDA eager 下接近 0.1ms-100ms 的规模，避免 batch_size=1、seq_len=16、height=32 这类过小输入。
- 如果使用 bool 比较，不要直接返回 bool 或全 0 结果；用 torch.where 或 float 运算转成非零、依赖输入的输出。
- 如果使用 EmbeddingBag，1D indices 才传 offsets；2D indices 不要传 offsets。

样本 id: {sample_id}
"""


def build_repair_prompt(ops: list[str], code_or_response: str, error: dict[str, Any], sample_id: int) -> str:
    ops_for_dataset = [original_op_name(op) for op in ops]
    clipped = code_or_response[-6000:]
    return f"""下面这个 PyTorch 样本没有通过验证。请修复它。

目标 ops:
{json.dumps(ops_for_dataset, ensure_ascii=False)}

失败原因:
{json.dumps(error, ensure_ascii=False)}

待修复代码或模型回复片段:
{clipped}

要求：
- 不要输出 JSON。
- 最终代码用 ```python 代码块输出。
- 保留 Model、get_inputs、get_init_inputs。
- 修复 shape/API/输出问题，确保 Model(*get_init_inputs())(*get_inputs()) 可运行。
- 如果输入 shape 太小，请放大到 CUDA Agent 风格规模：batch 32-512，H/W 128-1024，seq_len 256-2048 或 length 1024-16384，feature/embed 256-4096。
- 输出依赖输入，不能全零/常量，不能 NaN/Inf。
- 如果 bool 比较导致全零，改用 torch.where 或连续值运算。
- 如果 EmbeddingBag 报 offsets 错误，使用 1D indices + offsets 或改成合法调用。

样本 id: {sample_id}
"""


def call_chat_completion(cfg: LLMConfig, prompt: str) -> str:
    payload = {
        "model": cfg.model,
        "messages": [
            {"role": "user", "content": prompt},
        ],
        "temperature": cfg.temperature,
        "max_tokens": cfg.max_tokens,
    }
    body = json.dumps(payload).encode()
    request = urllib.request.Request(
        cfg.api_url,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {cfg.api_key}",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=cfg.timeout) as response:
        result = json.loads(response.read().decode())
    return result["choices"][0]["message"]["content"]


def extract_code_response(text: str) -> str:
    text = text.strip()
    text = THINK_RE.sub("", text).strip()
    blocks = [match.group(1).strip() for match in CODE_BLOCK_RE.finditer(text)]
    python_blocks = [block for block in blocks if "class Model" in block and "get_inputs" in block]
    if python_blocks:
        return trim_to_python_module(python_blocks[-1])
    start_candidates = [idx for idx in [text.rfind("import torch"), text.rfind("from torch")] if idx >= 0]
    if start_candidates:
        text = text[max(start_candidates) :]
    end_markers = ["\n\nExplanation:", "\n\n说明：", "\n\nThis code", "\n\n这个代码"]
    for marker in end_markers:
        marker_idx = text.find(marker)
        if marker_idx > 0:
            text = text[:marker_idx]
    if "class Model" not in text or "get_inputs" not in text:
        raise ValueError("could not extract runnable Python code from response")
    return trim_to_python_module(text.strip())


def trim_to_python_module(text: str) -> str:
    lines = text.splitlines()
    best: str | None = None
    for end in range(len(lines), 0, -1):
        candidate = "\n".join(lines[:end]).strip()
        try:
            tree = ast.parse(candidate)
        except SyntaxError:
            continue
        has_model = any(isinstance(node, ast.ClassDef) and node.name == "Model" for node in tree.body)
        has_get_inputs = any(isinstance(node, ast.FunctionDef) and node.name == "get_inputs" for node in tree.body)
        has_get_init_inputs = any(isinstance(node, ast.FunctionDef) and node.name == "get_init_inputs" for node in tree.body)
        if has_model and has_get_inputs and has_get_init_inputs:
            best = candidate
            break
    if best is None:
        raise ValueError("extracted code is not a complete Python module")
    return strip_top_level_test_code(best)


def strip_top_level_test_code(code: str) -> str:
    """Remove demo/test execution from the code saved into the parquet field."""

    tree = ast.parse(code)
    kept: list[ast.stmt] = []
    suspicious_targets = {
        "model",
        "inputs",
        "init_inputs",
        "out",
        "output",
        "outputs",
        "result",
        "compiled",
    }
    for node in tree.body:
        if isinstance(node, ast.If) and is_main_guard(node.test):
            continue
        if isinstance(node, ast.Expr):
            continue
        if isinstance(node, (ast.For, ast.While, ast.With, ast.Try)):
            continue
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            names = assignment_target_names(node)
            if names & suspicious_targets:
                continue
            value = node.value if isinstance(node, (ast.Assign, ast.AnnAssign)) else None
            if isinstance(value, ast.Call) and call_name(value.func) in {"Model", "get_inputs", "get_init_inputs", "print"}:
                continue
        kept.append(node)
    tree.body = kept
    ast.fix_missing_locations(tree)
    cleaned = ast.unparse(tree)
    code_has_required_symbols(cleaned)
    return cleaned + "\n"


def is_main_guard(node: ast.AST) -> bool:
    if not isinstance(node, ast.Compare):
        return False
    left = node.left
    if not (isinstance(left, ast.Name) and left.id == "__name__"):
        return False
    return any(isinstance(comp, ast.Constant) and comp.value == "__main__" for comp in node.comparators)


def assignment_target_names(node: ast.Assign | ast.AnnAssign) -> set[str]:
    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
    names: set[str] = set()
    for target in targets:
        for child in ast.walk(target):
            if isinstance(child, ast.Name):
                names.add(child.id)
    return names


def call_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = call_name(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    return None


def code_has_required_symbols(code: str) -> None:
    tree = ast.parse(code)
    has_model = any(isinstance(node, ast.ClassDef) and node.name == "Model" for node in tree.body)
    has_get_inputs = any(isinstance(node, ast.FunctionDef) and node.name == "get_inputs" for node in tree.body)
    has_get_init_inputs = any(isinstance(node, ast.FunctionDef) and node.name == "get_init_inputs" for node in tree.body)
    if not (has_model and has_get_inputs and has_get_init_inputs):
        raise ValueError("code must define Model, get_inputs, and get_init_inputs")


def cheat_check(code: str) -> dict[str, Any]:
    lowered = code.lower()
    banned_patterns = {
        "custom_cuda_or_triton": ["triton", "cudaextension", "load_inline", "cpp_extension"],
        "network_or_filesystem": ["requests.", "urllib.", "open(", "socket.", "subprocess", "os.system"],
        "hardcoded_constant_return": ["return torch.zeros", "return torch.ones", "return 0", "return None"],
    }
    hits: dict[str, list[str]] = {}
    for category, patterns in banned_patterns.items():
        found = [pattern for pattern in patterns if pattern in lowered]
        if found:
            hits[category] = found
    return {"passed": not hits, "hits": hits}


def shape_scale_check(code: str) -> dict[str, Any]:
    interesting = {
        "batch_size",
        "seq_len",
        "sequence_length",
        "length",
        "height",
        "width",
        "depth",
        "channels",
        "in_channels",
        "out_channels",
        "hidden_size",
        "hidden_dim",
        "embed_dim",
        "in_features",
        "out_features",
        "features",
        "embedding_dim",
    }
    values: dict[str, int] = {}
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return {"passed": False, "values": values, "warnings": ["syntax_error"]}
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        try:
            value = ast.literal_eval(node.value)
        except Exception:
            continue
        targets = node.targets
        if isinstance(value, tuple):
            for target, item in zip(targets[0].elts if targets and isinstance(targets[0], ast.Tuple) else [], value):
                if isinstance(target, ast.Name) and target.id in interesting and isinstance(item, int):
                    values[target.id] = item
        elif isinstance(value, int):
            for target in targets:
                if isinstance(target, ast.Name) and target.id in interesting:
                    values[target.id] = value
    warnings: list[str] = []
    if values.get("batch_size", 32) < 16:
        warnings.append("batch_size_is_small")
    if any(name in values for name in ["height", "width"]) and max(values.get("height", 0), values.get("width", 0)) < 128:
        warnings.append("spatial_size_is_small")
    if any(name in values for name in ["seq_len", "sequence_length"]) and max(
        values.get("seq_len", 0), values.get("sequence_length", 0)
    ) < 128:
        warnings.append("sequence_length_is_small")
    if "length" in values and values["length"] < 1024:
        warnings.append("length_is_small")
    if any(name in values for name in ["in_features", "out_features", "hidden_dim", "embed_dim"]) and max(
        values.get("in_features", 0),
        values.get("out_features", 0),
        values.get("hidden_dim", 0),
        values.get("embed_dim", 0),
    ) < 128:
        warnings.append("feature_dim_is_small")
    return {"passed": not warnings, "values": values, "warnings": warnings}


def move_to_device(value: Any, device: str) -> Any:
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, list):
        return [move_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(move_to_device(item, device) for item in value)
    if isinstance(value, dict):
        return {key: move_to_device(item, device) for key, item in value.items()}
    return value


def first_tensor(value: Any) -> torch.Tensor | None:
    if torch.is_tensor(value):
        return value
    if isinstance(value, (list, tuple)):
        for item in value:
            tensor = first_tensor(item)
            if tensor is not None:
                return tensor
    if isinstance(value, dict):
        for item in value.values():
            tensor = first_tensor(item)
            if tensor is not None:
                return tensor
    return None


def outputs_allclose(a: Any, b: Any, atol: float, rtol: float) -> bool:
    if torch.is_tensor(a) and torch.is_tensor(b):
        return torch.allclose(a, b, atol=atol, rtol=rtol)
    if isinstance(a, tuple) and isinstance(b, tuple) and len(a) == len(b):
        return all(outputs_allclose(x, y, atol, rtol) for x, y in zip(a, b))
    if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        return all(outputs_allclose(x, y, atol, rtol) for x, y in zip(a, b))
    return False


def sync(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def measure_ms(model: torch.nn.Module, inputs: list[Any], cfg: FilterConfig) -> float:
    with torch.no_grad():
        for _ in range(cfg.warmup):
            model(*inputs)
        sync(cfg.device)
        start = time.perf_counter()
        for _ in range(cfg.repeat):
            model(*inputs)
        sync(cfg.device)
    return (time.perf_counter() - start) * 1000.0 / cfg.repeat


def validate_sample(code: str, cfg: FilterConfig) -> tuple[bool, dict[str, Any]]:
    details: dict[str, Any] = {
        "checks": {
            "required_symbols": False,
            "no_cheating": False,
            "setup": False,
            "eager_forward": False,
            "finite_nonempty": False,
            "deterministic_same_input": False,
            "nonzero_output": False,
            "input_dependent": False,
            "eager_runtime": False,
            "torch_compile": False if cfg.require_compile else "skipped",
        }
    }
    try:
        code_has_required_symbols(code)
        details["checks"]["required_symbols"] = True
        cheat = cheat_check(code)
        details["cheat_check"] = cheat
        details["shape_scale_check"] = shape_scale_check(code)
        if not cheat["passed"]:
            details["reason"] = "cheating_or_forbidden_api"
            return False, details
        details["checks"]["no_cheating"] = True
        namespace: dict[str, Any] = {}
        exec(compile(code, "<llm_cuda_agent_sample>", "exec"), namespace)  # noqa: S102
        model = namespace["Model"](*namespace["get_init_inputs"]()).to(cfg.device).eval()
        details["checks"]["setup"] = True
        torch.manual_seed(1234)
        inputs1 = move_to_device(namespace["get_inputs"](), cfg.device)
        torch.manual_seed(5678)
        inputs2 = move_to_device(namespace["get_inputs"](), cfg.device)
        with torch.no_grad():
            out1 = model(*inputs1)
            out1_repeat = model(*inputs1)
            out2 = model(*inputs2)
        details["checks"]["eager_forward"] = True
    except Exception as exc:  # noqa: BLE001
        details.update({"reason": f"execution_failed:{type(exc).__name__}", "error": str(exc)})
        return False, details

    tensor = first_tensor(out1)
    if tensor is None or tensor.numel() == 0:
        details["reason"] = "empty_output"
        return False, details
    details["output_shape"] = list(tensor.shape)
    details["output_dtype"] = str(tensor.dtype)
    if not torch.isfinite(tensor).all().item():
        details["reason"] = "non_finite_output"
        return False, details
    details["checks"]["finite_nonempty"] = True
    if not outputs_allclose(out1, out1_repeat, cfg.atol, cfg.rtol):
        details["reason"] = "stochastic_output"
        return False, details
    details["checks"]["deterministic_same_input"] = True
    if torch.allclose(tensor, torch.zeros_like(tensor), atol=cfg.atol, rtol=cfg.rtol):
        details["reason"] = "zero_output"
        return False, details
    details["checks"]["nonzero_output"] = True
    if outputs_allclose(out1, out2, cfg.atol, cfg.rtol):
        details["reason"] = "constant_or_input_indistinguishable_output"
        return False, details
    details["checks"]["input_dependent"] = True
    try:
        eager_ms = measure_ms(model, inputs1, cfg)
    except Exception as exc:  # noqa: BLE001
        details.update({"reason": f"timing_failed:{type(exc).__name__}", "error": str(exc)})
        return False, details
    details["eager_ms"] = eager_ms
    if eager_ms < cfg.min_eager_ms:
        details["reason"] = "too_easy"
        return False, details
    if eager_ms > cfg.max_eager_ms:
        details["reason"] = "too_heavy"
        return False, details
    details["checks"]["eager_runtime"] = True

    if cfg.require_compile:
        try:
            compiled = torch.compile(model)
            with torch.no_grad():
                compiled_out = compiled(*inputs1)
            compile_matches = outputs_allclose(out1, compiled_out, cfg.atol, cfg.rtol)
            details["compile_matches_eager"] = compile_matches
            if not compile_matches:
                details["reason"] = "compile_output_mismatch"
                return False, details
            details["compile_ms"] = measure_ms(compiled, inputs1, cfg)
            details["checks"]["torch_compile"] = True
        except Exception as exc:  # noqa: BLE001
            details.update({"reason": f"compile_failed:{type(exc).__name__}", "error": str(exc)})
            return False, details

    details["reason"] = "accepted"
    return True, details


def normalize_record(code: str, requested_ops: list[str], available_ops: set[str]) -> dict[str, str]:
    normalized = requested_ops
    filtered = []
    for op in normalized:
        if op in available_ops and op not in filtered:
            filtered.append(op)
    if not filtered:
        raise ValueError("no generated ops are in meaningful_ops")
    return {
        "ops": json.dumps([original_op_name(op) for op in filtered], ensure_ascii=False),
        "data_source": f"llm_cuda_agent#{len(filtered)}",
        "code": code,
    }


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    with path.open("a") as f:
        f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def load_existing(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def render_progress(accepted: int, target: int, attempts: int, rejected: int, started_at: float, last_reason: str) -> None:
    width = 32
    ratio = accepted / target if target else 1.0
    filled = min(width, int(width * ratio))
    bar = "#" * filled + "-" * (width - filled)
    elapsed = max(1e-6, time.time() - started_at)
    rate = accepted / elapsed
    accept_rate = accepted / attempts if attempts else 0.0
    message = (
        f"\r[{bar}] {accepted}/{target} accepted | attempts={attempts} "
        f"rejected={rejected} accept_rate={accept_rate:.1%} speed={rate:.2f}/s last={last_reason[:60]}"
    )
    sys.stderr.write(message)
    sys.stderr.flush()


def run_attempt(payload: dict[str, Any]) -> dict[str, Any]:
    attempt = payload["attempt"]
    requested_ops = payload["requested_ops"]
    llm_cfg = LLMConfig(**payload["llm_cfg"])
    filter_cfg = FilterConfig(**payload["filter_cfg"])
    available_ops = set(payload["available_ops"])
    repair_attempts = payload["repair_attempts"]
    raw_records: list[dict[str, Any]] = []

    prompt = build_prompt(requested_ops, attempt)
    content_for_repair = ""
    last_error: dict[str, Any] = {}
    last_trace: dict[str, Any] | None = None

    for repair_round in range(repair_attempts + 1):
        try:
            if repair_round == 0:
                content = call_chat_completion(llm_cfg, prompt)
            else:
                repair_prompt = build_repair_prompt(requested_ops, content_for_repair, last_error, attempt)
                content = call_chat_completion(llm_cfg, repair_prompt)
            content_for_repair = content
            raw_records.append(
                {
                    "attempt": attempt,
                    "repair_round": repair_round,
                    "requested_ops": requested_ops,
                    "content": content,
                }
            )
            code = extract_code_response(content)
            record = normalize_record(code, requested_ops, available_ops)
            ok, details = validate_sample(record["code"], filter_cfg)
            trace = {
                "attempt": attempt,
                "repair_round": repair_round,
                "requested_ops": requested_ops,
                **record,
                "filter": details,
            }
            if ok:
                return {"accepted": True, "record": record, "trace": trace, "raw_records": raw_records}
            last_error = details
            content_for_repair = record["code"]
            last_trace = trace
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, ValueError) as exc:
            last_error = {"reason": f"generation_failed:{type(exc).__name__}", "error": str(exc)}
            last_trace = {
                "attempt": attempt,
                "repair_round": repair_round,
                "requested_ops": requested_ops,
                "filter": last_error,
            }

    return {
        "accepted": False,
        "trace": last_trace
        or {
            "attempt": attempt,
            "requested_ops": requested_ops,
            "filter": {"reason": "unknown_failure"},
        },
        "raw_records": raw_records,
    }


def next_attempt_payload(
    attempt: int,
    rng: random.Random,
    meaningful_ops: list[str],
    available_ops: set[str],
    llm_cfg: LLMConfig,
    filter_cfg: FilterConfig,
    args: argparse.Namespace,
) -> dict[str, Any]:
    requested_ops = weighted_sample_ops(rng, meaningful_ops, args.min_ops, args.max_ops)
    return {
        "attempt": attempt,
        "requested_ops": requested_ops,
        "available_ops": sorted(available_ops),
        "llm_cfg": llm_cfg.__dict__,
        "filter_cfg": filter_cfg.__dict__,
        "repair_attempts": args.repair_attempts,
    }


def consume_attempt_result(
    result: dict[str, Any],
    accepted: list[dict[str, str]],
    accepted_path: Path,
    rejected_path: Path,
    raw_path: Path,
) -> tuple[bool, str]:
    for raw_record in result.get("raw_records", []):
        append_jsonl(raw_path, raw_record)
    if result.get("accepted"):
        accepted.append(result["record"])
        append_jsonl(accepted_path, result["trace"])
        return True, "accepted"
    append_jsonl(rejected_path, result["trace"])
    reason = result.get("trace", {}).get("filter", {}).get("reason", "rejected")
    return False, str(reason)


def run(args: argparse.Namespace) -> None:
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise SystemExit(f"Please set {args.api_key_env} with your API token.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    accepted_path = output_dir / "accepted.jsonl"
    rejected_path = output_dir / "rejected.jsonl"
    raw_path = output_dir / "raw_responses.jsonl"
    if not args.resume:
        for path in [accepted_path, rejected_path, raw_path, output_dir / "summary.json", output_dir / "data.parquet"]:
            if path.exists():
                path.unlink()
    accepted = load_existing(accepted_path) if args.resume else []

    meaningful_ops = load_ops(Path(args.meaningful_ops))
    available_ops = set(meaningful_ops)
    rng = random.Random(args.seed)
    llm_cfg = LLMConfig(args.api_url, api_key, args.model, args.temperature, args.max_tokens, args.timeout)
    filter_cfg = FilterConfig(
        args.device,
        args.min_eager_ms,
        args.max_eager_ms,
        args.warmup,
        args.repeat,
        args.atol,
        args.rtol,
        args.require_compile,
    )

    attempts = 0
    rejected_count = 0
    last_reason = "start"
    started_at = time.time()
    if args.workers <= 1:
        while len(accepted) < args.count and attempts < args.max_attempts:
            attempts += 1
            payload = next_attempt_payload(attempts, rng, meaningful_ops, available_ops, llm_cfg, filter_cfg, args)
            result = run_attempt(payload)
            ok, last_reason = consume_attempt_result(result, accepted, accepted_path, rejected_path, raw_path)
            if not ok:
                rejected_count += 1
            if args.progress:
                render_progress(len(accepted), args.count, attempts, rejected_count, started_at, last_reason)
            elif attempts % args.progress_every == 0:
                print(f"attempts={attempts} accepted={len(accepted)} rejected={rejected_count} last={last_reason}")
    else:
        ctx = mp.get_context("spawn")
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as executor:
            futures: dict[concurrent.futures.Future[dict[str, Any]], int] = {}
            while len(accepted) < args.count and (attempts < args.max_attempts or futures):
                while len(futures) < args.workers and attempts < args.max_attempts and len(accepted) < args.count:
                    attempts += 1
                    payload = next_attempt_payload(attempts, rng, meaningful_ops, available_ops, llm_cfg, filter_cfg, args)
                    futures[executor.submit(run_attempt, payload)] = attempts
                if not futures:
                    break
                done, _ = concurrent.futures.wait(
                    futures,
                    return_when=concurrent.futures.FIRST_COMPLETED,
                )
                for future in done:
                    futures.pop(future)
                    try:
                        result = future.result()
                    except Exception as exc:  # noqa: BLE001
                        result = {
                            "accepted": False,
                            "trace": {"filter": {"reason": f"worker_failed:{type(exc).__name__}", "error": str(exc)}},
                            "raw_records": [],
                        }
                    ok, last_reason = consume_attempt_result(result, accepted, accepted_path, rejected_path, raw_path)
                    if not ok:
                        rejected_count += 1
                    if args.progress:
                        render_progress(len(accepted), args.count, attempts, rejected_count, started_at, last_reason)
                    elif attempts % args.progress_every == 0:
                        print(
                            f"attempts={attempts} accepted={len(accepted)} "
                            f"rejected={rejected_count} last={last_reason}"
                        )
                    if len(accepted) >= args.count:
                        for pending in futures:
                            pending.cancel()
                        futures.clear()
                        break

    if args.progress:
        sys.stderr.write("\n")

    df = pd.DataFrame(accepted[: args.count], columns=["ops", "data_source", "code"])
    df.to_parquet(output_dir / "data.parquet", index=False)
    summary = {
        "accepted": len(df),
        "attempts": attempts,
        "count_target": args.count,
        "output": str(output_dir / "data.parquet"),
        "model": args.model,
        "api_url": args.api_url,
        "workers": args.workers,
        "filter": filter_cfg.__dict__,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate CUDA-Agent-format data with an OpenAI-compatible LLM API.")
    parser.add_argument("--api-url", default="http://192.168.11.18:30055/v1/chat/completions")
    parser.add_argument("--api-key-env", default="MINIMAX_API_KEY")
    parser.add_argument("--model", default="MiniMax-M2.5")
    parser.add_argument("--meaningful-ops", default="ops_catalog/meaningful_ops.csv")
    parser.add_argument("--output-dir", default="data/llm_cuda_agent_ops")
    parser.add_argument("--count", type=int, default=6000)
    parser.add_argument("--max-attempts", type=int, default=30000)
    parser.add_argument("--min-ops", type=int, default=2)
    parser.add_argument("--max-ops", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260513)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument("--timeout", type=int, default=240)
    parser.add_argument("--repair-attempts", type=int, default=2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--min-eager-ms", type=float, default=0.0)
    parser.add_argument("--max-eager-ms", type=float, default=100.0)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--atol", type=float, default=1e-2)
    parser.add_argument("--rtol", type=float, default=1e-2)
    parser.add_argument("--require-compile", action="store_true")
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes for generation/validation.")
    parser.add_argument("--no-progress", dest="progress", action="store_false")
    parser.set_defaults(progress=True)
    parser.add_argument("--resume", action="store_true")
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
