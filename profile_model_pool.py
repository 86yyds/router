import argparse
import ast
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

from model_pool_runtime import ModelPoolRuntime


DEFAULT_BERTSCORE_MODEL = os.getenv("PROFILE_BERTSCORE_MODEL", "distilbert-base-uncased").strip() or "distilbert-base-uncased"
PROFILE_COMPOSITE_WEIGHTS = {
    "quality": 0.60,
    "success": 0.20,
    "cost": 0.12,
    "latency": 0.08,
}


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _normalize_text(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def _tokenize(value: Any) -> List[str]:
    text = _normalize_text(value)
    return re.findall(r"[a-z0-9_:.+-]+", text)


def _split_camel(text: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", " ", str(text or ""))


def _unwrap_label_candidate(value: Any) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    lines = [ln.strip(" -*`'\"") for ln in raw.splitlines() if ln.strip()]
    if not lines:
        lines = [raw]
    candidates: List[str] = []
    for line in lines[:3] + lines[-3:]:
        clean = re.sub(
            r"(?i)^(task|answer|final answer|final selection|prediction|predicted label|label|intent|class|category|output)\s*[:：=\-]\s*",
            "",
            line,
        ).strip("`'\" ")
        if clean:
            candidates.append(clean)
    if not candidates:
        return raw
    return sorted(candidates, key=lambda s: (len(re.findall(r"[A-Za-z0-9_:\-]+", s)), len(s)))[0]


def _normalize_label_text(value: Any) -> str:
    text = _split_camel(_unwrap_label_candidate(value))
    text = text.strip().lower()
    text = text.replace("::", ":")
    text = re.sub(r"[^a-z0-9:]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _label_tail(value: Any) -> str:
    text = _normalize_label_text(value)
    if ":" in text:
        text = text.split(":")[-1].strip()
    return text


def _label_compact(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", _label_tail(value))


def _safe_div(num: float, den: float) -> float:
    return 0.0 if den <= 0 else num / den


def _token_f1(pred: Any, ref: Any) -> float:
    pred_tokens = _tokenize(pred)
    ref_tokens = _tokenize(ref)
    if not pred_tokens or not ref_tokens:
        return 0.0
    pred_set = set(pred_tokens)
    ref_set = set(ref_tokens)
    inter = len(pred_set & ref_set)
    precision = _safe_div(inter, len(pred_set))
    recall = _safe_div(inter, len(ref_set))
    if precision + recall == 0:
        return 0.0
    return round(2 * precision * recall / (precision + recall), 4)


def _char_ngram_recall(pred: Any, ref: Any, n: int = 3) -> float:
    pred_s = _normalize_text(pred)
    ref_s = _normalize_text(ref)
    if len(ref_s) < n or len(pred_s) < n:
        return 1.0 if ref_s and ref_s in pred_s else 0.0
    pred_ngrams = {pred_s[i : i + n] for i in range(len(pred_s) - n + 1)}
    ref_ngrams = {ref_s[i : i + n] for i in range(len(ref_s) - n + 1)}
    if not ref_ngrams:
        return 0.0
    return round(len(pred_ngrams & ref_ngrams) / len(ref_ngrams), 4)


def _lcs_len(a: List[str], b: List[str]) -> int:
    if not a or not b:
        return 0
    dp = [0] * (len(b) + 1)
    for i in range(1, len(a) + 1):
        prev = 0
        for j in range(1, len(b) + 1):
            cur = dp[j]
            if a[i - 1] == b[j - 1]:
                dp[j] = prev + 1
            else:
                dp[j] = max(dp[j], dp[j - 1])
            prev = cur
    return dp[-1]


def _rouge_l_f1(pred: Any, ref: Any) -> float:
    pred_tokens = _tokenize(pred)
    ref_tokens = _tokenize(ref)
    if not pred_tokens or not ref_tokens:
        return 0.0
    lcs = _lcs_len(pred_tokens, ref_tokens)
    precision = _safe_div(lcs, len(pred_tokens))
    recall = _safe_div(lcs, len(ref_tokens))
    if precision + recall == 0:
        return 0.0
    return round(2 * precision * recall / (precision + recall), 4)


def _extract_numbers(text: Any) -> List[str]:
    return re.findall(r"-?\d+(?:\.\d+)?", _normalize_text(text))


def _extract_final_answer(text: Any) -> str:
    raw = str(text or "").strip()
    if not raw:
        return ""
    patterns = [
        r"####\s*(-?\d+(?:\.\d+)?)",
        r"answer\s*(?:is|=|:)\s*(-?\d+(?:\.\d+)?)",
        r"final\s*answer\s*(?:is|=|:)\s*(-?\d+(?:\.\d+)?)",
        r"因此(?:答案|结果)(?:是|为)?\s*(-?\d+(?:\.\d+)?)",
        r"答案(?:是|为)?\s*(-?\d+(?:\.\d+)?)",
    ]
    lowered = _normalize_text(raw)
    for pat in patterns:
        m = re.search(pat, lowered)
        if m:
            return m.group(1)
    nums = _extract_numbers(lowered)
    return nums[-1] if nums else ""


def _extract_python_symbols(text: Any) -> Dict[str, List[str]]:
    raw = str(text or "")
    symbols = {
        "defs": re.findall(r"def\s+([a-zA-Z_][a-zA-Z0-9_]*)", raw),
        "classes": re.findall(r"class\s+([a-zA-Z_][a-zA-Z0-9_]*)", raw),
        "imports": re.findall(r"(?:from|import)\s+([a-zA-Z0-9_\.]+)", raw),
    }
    return symbols


def _extract_visible_answer_from_reasoning(task_type: str, reasoning_text: Any) -> str:
    text = str(reasoning_text or "").strip()
    if not text:
        return ""

    if task_type == "classification":
        patterns = [
            r"final selection:\s*[`'\"]?([a-z0-9_:\-]+)[`'\"]?",
            r"output:\s*[`'\"]?([a-z0-9_:\-]+)[`'\"]?",
            r"therefore.*?[`'\"]([a-z0-9_:\-]+)[`'\"]",
        ]
        lowered = _normalize_text(text)
        for pat in patterns:
            m = re.search(pat, lowered, flags=re.I)
            if m:
                return _unwrap_label_candidate(m.group(1).strip())
        last_token = re.findall(r"[a-z0-9_:\-]+", lowered)
        return _unwrap_label_candidate(last_token[-1] if last_token else "")

    if task_type == "qa":
        lowered = _normalize_text(text)
        m = re.search(r"\b(yes|no)\b(?=[^a-zA-Z]*$)", lowered)
        if m:
            return m.group(1)
        m = re.search(r"answer should be\s+([^\n\.]+)", lowered)
        if m:
            return m.group(1).strip(" \"'`")
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        return lines[-1] if lines else text

    if task_type == "summary":
        patterns = [
            r"final summary:\s*(.+)",
            r"construct final output:\s*(.+)",
            r"summary:\s*(.+)",
        ]
        for pat in patterns:
            m = re.search(pat, text, flags=re.I | re.S)
            if m:
                return m.group(1).strip().splitlines()[0].strip(" \"'`")
        lines = [ln.strip(" -*`\"'") for ln in text.splitlines() if ln.strip()]
        for line in reversed(lines):
            if len(line.split()) >= 6:
                return line
        return lines[-1] if lines else text

    if task_type == "generation":
        patterns = [
            r"final story:\s*(.+)",
            r"final passage:\s*(.+)",
            r"final output:\s*(.+)",
        ]
        for pat in patterns:
            m = re.search(pat, text, flags=re.I | re.S)
            if m:
                return m.group(1).strip().splitlines()[0].strip(" \"'`")
        lines = [ln.rstrip() for ln in text.splitlines() if ln.strip()]
        long_lines = [ln.strip(" -*`\"'") for ln in lines if len(ln.split()) >= 8]
        if long_lines:
            return "\n".join(long_lines[-6:]).strip()
        return text

    if task_type == "extraction":
        block_patterns = [
            r"proposed extraction:\s*(\[[\s\S]+?\])",
            r"final output:\s*(\[[\s\S]+?\])",
            r"(\[[\s\S]*\])",
        ]
        for pat in block_patterns:
            m = re.search(pat, text, flags=re.I)
            if m:
                return m.group(1).strip()
        json_block = re.search(r"```(?:json)?\s*([\s\S]+?)```", text, flags=re.I)
        if json_block:
            return json_block.group(1).strip()
        lines = [ln.strip(" -*`\"'") for ln in text.splitlines() if ln.strip()]
        return "\n".join(lines[-8:]).strip() if lines else text

    if task_type == "reasoning":
        final = _extract_final_answer(text)
        if final:
            return final
        m = re.search(r"final answer:\s*([^\n]+)", text, flags=re.I)
        if m:
            return m.group(1).strip()
        return text

    return ""


def _python_syntax_ok(text: Any) -> float:
    raw = str(text or "").strip()
    if not raw:
        return 0.0
    try:
        ast.parse(raw)
        return 1.0
    except SyntaxError:
        return 0.0
    except Exception:
        return 0.0


def _score_code(pred: Any, ref: Any) -> float:
    pred_s = str(pred or "").strip()
    ref_s = str(ref or "").strip()
    if not pred_s:
        return 0.0
    pred_symbols = _extract_python_symbols(pred_s)
    ref_symbols = _extract_python_symbols(ref_s)
    signature_hits = 0
    signature_total = 0
    for key in ("defs", "classes", "imports"):
        ref_values = set(ref_symbols[key])
        if ref_values:
            signature_total += len(ref_values)
            signature_hits += len(ref_values & set(pred_symbols[key]))
    signature_score = _safe_div(signature_hits, signature_total) if signature_total else 0.0
    token_score = _token_f1(pred_s, ref_s)
    syntax_score = _python_syntax_ok(pred_s)
    def_bonus = 1.0 if "def " in pred_s else 0.0
    score = 0.4 * signature_score + 0.3 * token_score + 0.2 * syntax_score + 0.1 * def_bonus
    return round(min(1.0, score), 4)


def _score_overlap(task_type: str, pred: str, ref: Any) -> float:
    if ref is None:
        return 0.0
    pred_s = _normalize_text(pred)
    ref_s = _normalize_text(json.dumps(ref, ensure_ascii=False) if isinstance(ref, list) else ref)
    if not pred_s:
        return 0.0

    if task_type == "classification":
        ref_norm = _normalize_label_text(ref)
        pred_norm = _normalize_label_text(pred)
        ref_tail = _label_tail(ref)
        pred_tail = _label_tail(pred)
        if pred_norm == ref_norm:
            return 1.0
        if pred_tail and ref_tail and pred_tail == ref_tail:
            return 1.0
        if _label_compact(pred) and _label_compact(pred) == _label_compact(ref):
            return 1.0
        if ref_norm and ref_norm in pred_norm:
            return 0.9
        if ref_tail and ref_tail in pred_tail:
            return 0.9
        return max(_token_f1(pred_norm, ref_norm), _token_f1(pred_tail, ref_tail))

    if task_type == "qa":
        if pred_s == ref_s:
            return 1.0
        if ref_s and ref_s in pred_s:
            return 0.9
        return round(max(_token_f1(pred_s, ref_s), _char_ngram_recall(pred_s, ref_s)), 4)

    if task_type == "reasoning":
        pred_final = _extract_final_answer(pred_s)
        ref_final = _extract_final_answer(ref_s)
        if pred_final and ref_final and pred_final == ref_final:
            return 1.0
        pred_nums = _extract_numbers(pred_s)
        ref_nums = _extract_numbers(ref_s)
        if ref_nums and pred_nums == ref_nums:
            return 1.0
        if ref_nums and pred_nums and pred_nums[-1] == ref_nums[-1]:
            return 1.0
        if ref_nums:
            return 0.0
        return round(max(_token_f1(pred_s, ref_s), _char_ngram_recall(pred_s, ref_s)), 4)

    if task_type == "extraction":
        token_score = _token_f1(pred_s, ref_s)
        char_score = _char_ngram_recall(pred_s, ref_s)
        rouge_score = _rouge_l_f1(pred_s, ref_s)
        return round(0.45 * token_score + 0.2 * char_score + 0.35 * rouge_score, 4)

    if task_type == "summary":
        rouge_score = _rouge_l_f1(pred_s, ref_s)
        token_score = _token_f1(pred_s, ref_s)
        char_score = _char_ngram_recall(pred_s, ref_s)
        return round(0.65 * rouge_score + 0.2 * token_score + 0.15 * char_score, 4)

    if task_type == "generation":
        rouge_score = _rouge_l_f1(pred_s, ref_s)
        token_score = _token_f1(pred_s, ref_s)
        char_score = _char_ngram_recall(pred_s, ref_s)
        return round(0.45 * rouge_score + 0.35 * token_score + 0.2 * char_score, 4)

    if task_type == "codegen":
        return _score_code(pred_s, ref_s)

    token_score = _token_f1(pred_s, ref_s)
    char_score = _char_ngram_recall(pred_s, ref_s)
    return round(0.65 * token_score + 0.35 * char_score, 4)


def _estimate_profile_budget(task_type: str, row: Dict[str, Any], runtime: ModelPoolRuntime) -> Tuple[int, int]:
    expected = str(row.get("expected_output_length", "") or "").strip().lower()
    spec_max = max(spec.default_max_tokens for spec in runtime.list_enabled_models(include_codegen=True))
    caps = {
        "classification": 64,
        "extraction": 96,
        "qa": 160,
        "reasoning": 192,
        "summary": 256,
        "generation": 288,
        "codegen": 320,
    }
    base = caps.get(task_type, 160)
    if expected == "short":
        base = min(base, 96)
    elif expected == "medium":
        base = max(base, 160)
    elif expected == "long":
        base = max(base, 256)
    timeout = 35 if task_type in {"classification", "extraction"} else 45
    timeout = min(timeout, 60)
    return min(base, spec_max), timeout


def _build_profile_prompt(runtime: ModelPoolRuntime, row: Dict[str, Any]) -> str:
    task_type = str(row.get("task_type", "") or "")
    base_prompt = runtime.build_prompt(task_type, row.get("instruction", ""), row.get("input", ""), row.get("metadata"))
    task_guidance = {
        "classification": "Return one non-empty label only.",
        "extraction": "Return a non-empty extraction result only.",
        "qa": "Return a non-empty direct answer.",
        "reasoning": "Return a non-empty answer and include the final answer explicitly.",
        "summary": "Return a non-empty summary.",
        "generation": "Return a non-empty complete passage.",
        "codegen": "Return non-empty runnable code.",
    }.get(task_type, "Return a non-empty answer.")
    return (
        f"{base_prompt}\n\n"
        f"Profile run requirement: you must return a non-empty answer. "
        f"Do not return an empty message.\n{task_guidance}"
    )


def _build_empty_retry_prompt(prompt: str, task_type: str) -> str:
    retry_hint = {
        "classification": "Return just the label text.",
        "extraction": "Return the extracted entities or fields directly.",
        "qa": "Return the answer in one short sentence.",
        "reasoning": "Return the reasoning briefly and end with a clear final answer.",
        "summary": "Return one concise summary paragraph.",
        "generation": "Return one complete coherent passage.",
        "codegen": "Return code only.",
    }.get(task_type, "Return the answer directly.")
    return (
        f"{prompt}\n\n"
        "Your previous attempt was empty. This time you must output visible text. "
        f"{retry_hint}"
    )


def _invoke_profile_with_nonempty_retry(
    runtime: ModelPoolRuntime,
    spec_model_id: str,
    task_type: str,
    prompt: str,
    metadata: Any,
    max_tokens_override: int,
    timeout_override: int,
    empty_retries: int,
) -> Dict[str, Any]:
    total_attempts = 0
    last_result: Dict[str, Any] | None = None
    current_prompt = prompt
    current_max_tokens = max(1, int(max_tokens_override))
    max_model_tokens = max(current_max_tokens, int(runtime.get_spec(spec_model_id).default_max_tokens or current_max_tokens))
    for _ in range(max(0, int(empty_retries)) + 1):
        result = runtime.invoke(
            spec_model_id,
            task_type,
            current_prompt,
            metadata,
            max_tokens_override=current_max_tokens,
            timeout_override=timeout_override,
        )
        total_attempts += int(result.get("attempts") or 1)
        last_result = result
        response_text = result.get("response_text", "")
        raw = result.get("raw")
        if (pd.isna(response_text) or not str(response_text or "").strip()) and raw is not None:
            try:
                reasoning_text = getattr(raw.choices[0].message, "reasoning_content", "") if getattr(raw, "choices", None) else ""
            except Exception:
                reasoning_text = ""
            recovered = _extract_visible_answer_from_reasoning(task_type, reasoning_text)
            if recovered:
                result = dict(result)
                result["response_text"] = recovered
                result["profile_recovered_from_reasoning"] = True
                last_result = result
        response_text = result.get("response_text", "")
        has_text = False if pd.isna(response_text) else bool(str(response_text or "").strip())
        if result.get("success") and has_text:
            original_attempts = int(result.get("attempts") or 1)
            result["attempts"] = total_attempts
            result["profile_empty_retry_count"] = max(0, total_attempts - original_attempts)
            result["profile_retry_max_tokens"] = current_max_tokens
            return result
        current_prompt = _build_empty_retry_prompt(prompt, task_type)
        grown_tokens = min(max_model_tokens, max(current_max_tokens + 64, int(current_max_tokens * 1.5)))
        if spec_model_id == "minimax_m25":
            grown_tokens = min(max_model_tokens, max(256, grown_tokens))
        current_max_tokens = max(current_max_tokens, grown_tokens)
    failed = dict(last_result or {})
    failed["success"] = False
    failed["error_type"] = str(failed.get("error_type", "") or "EmptyResponse")
    failed["error_message"] = str(failed.get("error_message", "") or "Model returned an empty response after profiling retries")
    failed["attempts"] = total_attempts
    failed["response_text"] = "" if pd.isna(failed.get("response_text")) else str(failed.get("response_text", "") or "")
    failed["profile_empty_retry_count"] = max(0, int(empty_retries))
    failed["profile_retry_max_tokens"] = current_max_tokens
    return failed


def _load_existing_trace(trace_path: Path) -> Tuple[List[Dict[str, Any]], Dict[Tuple[str, str], Dict[str, Any]]]:
    rows: List[Dict[str, Any]] = []
    index: Dict[Tuple[str, str], Dict[str, Any]] = {}
    if not trace_path.exists():
        return rows, index
    with trace_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            key = (str(item.get("sample_id", "")), str(item.get("model_id", "")))
            rows.append(item)
            if all(key):
                index[key] = item
    return rows, index


def _is_effective_profile_success(item: Dict[str, Any]) -> bool:
    if not bool(item.get("success")):
        return False
    response = item.get("response_text", "")
    if pd.isna(response):
        return False
    return bool(str(response or "").strip())


def _normalize_profile_result(task_type: str, item: Dict[str, Any]) -> Dict[str, Any]:
    normalized = dict(item)
    effective_success = _is_effective_profile_success(normalized)
    normalized["success_effective"] = effective_success
    if bool(normalized.get("success")) and not effective_success:
        normalized["error_type"] = str(normalized.get("error_type", "") or "EmptyResponse")
        normalized["error_message"] = str(normalized.get("error_message", "") or "Model returned an empty response during profiling")
    normalized["score_raw"] = (
        _score_overlap(task_type, normalized.get("response_text", ""), normalized.get("reference"))
        if effective_success
        else 0.0
    )
    return normalized


def _safe_minmax(series: pd.Series) -> pd.Series:
    vals = pd.to_numeric(series, errors="coerce").fillna(0.0)
    if vals.empty:
        return vals
    lo = float(vals.min())
    hi = float(vals.max())
    if hi - lo <= 1e-12:
        return pd.Series([1.0] * len(vals), index=vals.index)
    return (vals - lo) / (hi - lo)


def _compute_bertscore_f1(preds: List[str], refs: List[str]) -> List[float]:
    if not preds:
        return []
    try:
        from bert_score import score as bertscore_score
    except Exception:
        return [0.0] * len(preds)
    try:
        _, _, f1 = bertscore_score(
            preds,
            refs,
            model_type=DEFAULT_BERTSCORE_MODEL,
            batch_size=min(16, max(1, len(preds))),
            verbose=False,
            rescale_with_baseline=False,
        )
        return [round(float(x), 4) for x in f1.tolist()]
    except Exception:
        return [0.0] * len(preds)


def _apply_semantic_quality(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work["quality_token_f1"] = 0.0
    work["quality_char_recall"] = 0.0
    work["quality_rouge_l"] = 0.0
    work["quality_bertscore_f1"] = 0.0
    work["score_raw_lexical"] = pd.to_numeric(work.get("score_raw"), errors="coerce").fillna(0.0)
    response_series = work.get("response_text", pd.Series("", index=work.index)).fillna("").astype(str)
    reference_series = work.get("reference", pd.Series("", index=work.index)).apply(
        lambda x: json.dumps(x, ensure_ascii=False) if isinstance(x, list) else str(x or "")
    )
    success_series = work.get("success_effective", pd.Series(False, index=work.index)).astype(bool)
    for idx in work.index:
        if not bool(success_series.loc[idx]):
            continue
        pred = response_series.loc[idx]
        ref = reference_series.loc[idx]
        work.loc[idx, "quality_token_f1"] = _token_f1(pred, ref)
        work.loc[idx, "quality_char_recall"] = _char_ngram_recall(pred, ref)
        work.loc[idx, "quality_rouge_l"] = _rouge_l_f1(pred, ref)

    semantic_mask = work["task_type"].astype(str).isin({"summary", "generation"}) & success_series
    semantic_idx = work.index[semantic_mask].tolist()
    if semantic_idx:
        bert_scores = _compute_bertscore_f1(
            [response_series.loc[idx] for idx in semantic_idx],
            [reference_series.loc[idx] for idx in semantic_idx],
        )
        for idx, score in zip(semantic_idx, bert_scores):
            work.loc[idx, "quality_bertscore_f1"] = score

    def _final_quality(row: pd.Series) -> float:
        if not bool(row.get("success_effective")):
            return 0.0
        task_type = str(row.get("task_type", "") or "")
        lexical = float(row.get("score_raw_lexical", 0.0) or 0.0)
        rouge = float(row.get("quality_rouge_l", 0.0) or 0.0)
        token = float(row.get("quality_token_f1", 0.0) or 0.0)
        bert = float(row.get("quality_bertscore_f1", 0.0) or 0.0)
        if task_type == "summary":
            return round(0.40 * rouge + 0.20 * token + 0.40 * max(bert, lexical), 4)
        if task_type == "generation":
            return round(0.25 * rouge + 0.15 * token + 0.60 * max(bert, lexical), 4)
        return round(lexical, 4)

    work["score_raw"] = work.apply(_final_quality, axis=1)
    return work


def _append_profile_composite_summary(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return summary.copy()
    work = summary.copy()
    work["avg_quality"] = pd.to_numeric(work.get("avg_quality"), errors="coerce").fillna(0.0)
    work["success_rate"] = pd.to_numeric(work.get("success_rate"), errors="coerce").fillna(0.0)
    work["avg_cost_usd"] = pd.to_numeric(work.get("avg_cost_usd"), errors="coerce").fillna(0.0)
    work["avg_latency_sec"] = pd.to_numeric(work.get("avg_latency_sec"), errors="coerce").fillna(0.0)
    quality_norm = []
    success_norm = []
    cost_eff = []
    latency_eff = []
    for _, sub_idx in work.groupby("task_type").groups.items():
        idx_list = list(sub_idx)
        sub = work.loc[idx_list]
        quality_norm.append(pd.Series(sub["avg_quality"].clip(lower=0.0, upper=1.0), index=sub.index))
        success_norm.append(pd.Series(sub["success_rate"].clip(lower=0.0, upper=1.0), index=sub.index))
        cost_eff.append(1.0 - _safe_minmax(sub["avg_cost_usd"]))
        latency_eff.append(1.0 - _safe_minmax(sub["avg_latency_sec"]))
    work["quality_component"] = pd.concat(quality_norm).sort_index()
    work["success_component"] = pd.concat(success_norm).sort_index()
    work["cost_efficiency_component"] = pd.concat(cost_eff).sort_index()
    work["latency_efficiency_component"] = pd.concat(latency_eff).sort_index()
    work["profile_composite_score"] = (
        PROFILE_COMPOSITE_WEIGHTS["quality"] * work["quality_component"]
        + PROFILE_COMPOSITE_WEIGHTS["success"] * work["success_component"]
        + PROFILE_COMPOSITE_WEIGHTS["cost"] * work["cost_efficiency_component"]
        + PROFILE_COMPOSITE_WEIGHTS["latency"] * work["latency_efficiency_component"]
    ).round(4)
    work["profile_rank"] = (
        work.groupby("task_type")["profile_composite_score"]
        .rank(method="dense", ascending=False)
        .astype(int)
    )
    return work


def _write_artifacts(trace_rows: List[Dict[str, Any]], out_dir: Path) -> None:
    if not trace_rows:
        return
    normalized_rows = [_normalize_profile_result(str(item.get("task_type", "")), item) for item in trace_rows]
    df = _apply_semantic_quality(pd.DataFrame(normalized_rows))
    df.to_csv(out_dir / "profiling_trace.csv", index=False, encoding="utf-8-sig")
    summary = (
        df.groupby(["task_type", "model_id"], dropna=False)
        .agg(
            avg_cost_usd=("cost_usd", "mean"),
            avg_latency_sec=("latency_sec", "mean"),
            avg_quality=("score_raw", "mean"),
            success_rate=("success_effective", "mean"),
            avg_attempts=("attempts", "mean"),
            samples=("sample_id", "count"),
        )
        .reset_index()
    )
    summary = _append_profile_composite_summary(summary)
    summary.to_csv(out_dir / "profile_summary.csv", index=False, encoding="utf-8-sig")

    defaults = []
    for task_type, sub in summary.groupby("task_type"):
        best = sub.sort_values(
            ["profile_composite_score", "avg_quality", "success_rate", "avg_cost_usd", "avg_latency_sec"],
            ascending=[False, False, False, True, True],
        ).iloc[0]
        defaults.append(
            {
                "task_type": task_type,
                "recommended_model": best["model_id"],
                "profile_composite_score": best["profile_composite_score"],
            }
        )
    pd.DataFrame(defaults).to_csv(out_dir / "task_default_models.csv", index=False, encoding="utf-8-sig")


def _progress_line(done: int, total: int, started: float, success_count: int) -> str:
    elapsed = max(0.001, time.perf_counter() - started)
    rate = done / elapsed
    eta = (total - done) / rate if rate > 0 else 0.0
    width = 24
    fill = int(width * done / max(total, 1))
    bar = "#" * fill + "-" * (width - fill)
    success_rate = 100.0 * success_count / max(done, 1)
    return (
        f"[profile] [{bar}] {done}/{total} "
        f"elapsed={elapsed/60:.1f}m eta={eta/60:.1f}m success={success_rate:.1f}%"
    )


def run_profile(
    calibration_path: Path,
    out_dir: Path,
    include_codegen: bool = False,
    profile_max_retries: int = 0,
    profile_empty_retries: int = 1,
    resume: bool = True,
) -> None:
    runtime = ModelPoolRuntime()
    runtime.max_retries = max(0, profile_max_retries)
    rows = _read_jsonl(calibration_path)
    models = runtime.list_enabled_models(include_codegen=include_codegen)
    if not rows:
        raise FileNotFoundError(f"Calibration set not found or empty: {calibration_path}")

    out_dir.mkdir(parents=True, exist_ok=True)
    trace_jsonl = out_dir / "profiling_trace.jsonl"
    trace_rows, existing = _load_existing_trace(trace_jsonl) if resume else ([], {})
    trace_mode = "a" if resume and trace_jsonl.exists() else "w"

    total = len(rows) * len(models)
    completed = len(existing)
    success_count = sum(1 for item in trace_rows if item.get("success"))
    started = time.perf_counter()
    last_draw = 0.0

    with trace_jsonl.open(trace_mode, encoding="utf-8") as sink:
        for row_idx, row in enumerate(rows, start=1):
            task_type = row["task_type"]
            prompt = _build_profile_prompt(runtime, row)
            max_tokens_override, timeout_override = _estimate_profile_budget(task_type, row, runtime)

            for model_idx, spec in enumerate(models, start=1):
                if task_type == "codegen" and spec.model_id == "qwen_coder" and not include_codegen:
                    continue
                key = (row["sample_id"], spec.model_id)
                if key in existing:
                    now = time.perf_counter()
                    if now - last_draw >= 0.2:
                        sys.stdout.write(
                            "\r"
                            + _progress_line(completed, total, started, success_count)
                            + f" current={row_idx}/{len(rows)}:{row['sample_id']} {model_idx}/{len(models)}:{spec.model_id} (resume)"
                        )
                        sys.stdout.flush()
                        last_draw = now
                    continue

                result = _invoke_profile_with_nonempty_retry(
                    runtime,
                    spec.model_id,
                    task_type,
                    prompt,
                    row.get("metadata"),
                    max_tokens_override,
                    timeout_override,
                    profile_empty_retries,
                )
                result["sample_id"] = row["sample_id"]
                result["source_dataset"] = row["source_dataset"]
                result["reference"] = row.get("reference")
                result["security_level"] = row.get("security_level", "public")
                result["budget_level"] = row.get("budget_level", "medium")
                result["profile_max_tokens"] = max_tokens_override
                result["profile_timeout_sec"] = timeout_override
                result = _normalize_profile_result(task_type, result)
                trace_rows.append(result)
                existing[key] = result
                sink.write(json.dumps(result, ensure_ascii=False) + "\n")
                sink.flush()

                completed += 1
                if result["success_effective"]:
                    success_count += 1
                sys.stdout.write(
                    "\r"
                    + _progress_line(completed, total, started, success_count)
                    + f" current={row_idx}/{len(rows)}:{row['sample_id']} {model_idx}/{len(models)}:{spec.model_id}"
                    + " " * 8
                )
                sys.stdout.flush()

            _write_artifacts(trace_rows, out_dir)

    _write_artifacts(trace_rows, out_dir)
    sys.stdout.write("\n[profile] done\n")
    sys.stdout.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration_path", type=str, default="datasets_unified/calibration_v1.jsonl")
    parser.add_argument("--out_dir", type=str, default="artifacts_v2/profiling")
    parser.add_argument("--include_codegen", action="store_true")
    parser.add_argument("--profile_max_retries", type=int, default=0)
    parser.add_argument("--profile_empty_retries", type=int, default=1)
    parser.add_argument("--no_resume", action="store_true")
    args = parser.parse_args()
    run_profile(
        Path(args.calibration_path),
        Path(args.out_dir),
        include_codegen=args.include_codegen,
        profile_max_retries=args.profile_max_retries,
        profile_empty_retries=args.profile_empty_retries,
        resume=not args.no_resume,
    )
