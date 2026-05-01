from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from build_mixed_budget_compare_figures import (
    _load_review_map,
    _our_eval_rows,
    _rank_eagle,
    _rank_random,
    _rank_routellm,
    _train_routellm_probs,
)
from build_task_triplet_figures import _load_matrix, _load_subset_rows, _prepare_online, _usable_ids


OUT_DIR = Path("artifacts_v2/requested_six_figures_v2")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _rank_oracle(sample_rows: pd.DataFrame, judge_map: Dict[Tuple[str, str], int] | None = None) -> List[str]:
    sub = sample_rows.copy()
    if judge_map is not None and str(sub["task_type"].iloc[0]) in {"classification", "qa"}:
        sub["oracle_score"] = [
            float(judge_map.get((str(sid), str(mid)), 0))
            for sid, mid in zip(sub["sample_id"].astype(str), sub["model_id"].astype(str))
        ]
    else:
        sub["oracle_score"] = pd.to_numeric(sub["strict_correct"], errors="coerce").fillna(0.0)
    sub = sub.sort_values(["oracle_score", "cost_usd", "latency_sec"], ascending=[False, True, True])
    return sub["model_id"].astype(str).tolist()


def _load_all():
    subset_rows = _load_subset_rows(Path("datasets_unified/eval_100_per_task.jsonl"))
    subset_ids = [str(r["sample_id"]) for r in subset_rows]
    matrix_df = _load_matrix([Path("artifacts_v2/profiling_triplet_combined_v2.csv")], subset_ids)
    online_df = _prepare_online(Path("artifacts_v2/router_eval_100_head_m25_v1/latest_router_results.csv"), subset_ids)
    summary_df = pd.read_csv("artifacts_v2/profiling_m25_merged_v1/profile_summary.csv")
    summary_df["avg_quality"] = pd.to_numeric(summary_df.get("avg_quality"), errors="coerce").fillna(0.0)
    summary_df["success_rate"] = pd.to_numeric(summary_df.get("success_rate"), errors="coerce").fillna(1.0)
    summary_df["avg_latency_sec"] = pd.to_numeric(summary_df.get("avg_latency_sec"), errors="coerce").fillna(0.0)
    human_cls = _load_review_map(Path("artifacts_v2/classification_manual_check_v1_annotated.csv"))
    human_qa = _load_review_map(Path("artifacts_v2/qa_manual_check_v1_annotated.csv"))
    judge_df = pd.read_csv("artifacts_v2/cls_qa_matrix_model_judge_v1.csv")
    judge_map = {
        (str(r["sample_id"]), str(r["model_id"])): int(pd.to_numeric(r["judge_correct"], errors="coerce"))
        for _, r in judge_df.iterrows()
    }
    return subset_rows, matrix_df, online_df, summary_df, human_cls, human_qa, judge_map


def _task_ids(matrix_df: pd.DataFrame, online_df: pd.DataFrame, task: str) -> List[str]:
    return _usable_ids(matrix_df, online_df, task)


def _combined_ids(matrix_df: pd.DataFrame, online_df: pd.DataFrame) -> List[str]:
    ids = []
    for task in ["classification", "qa", "reasoning"]:
        ids.extend(_task_ids(matrix_df, online_df, task))
    return ids


def _our_rows(online_df: pd.DataFrame, matrix_df: pd.DataFrame, ids: Iterable[str], human_cls: Dict[str, int], human_qa: Dict[str, int], mode: str) -> pd.DataFrame:
    ids = set(str(x) for x in ids)
    online_sub = online_df[online_df["sample_id"].astype(str).isin(ids)].copy()
    matrix_sub = matrix_df[matrix_df["sample_id"].astype(str).isin(ids)].copy()
    if mode == "manual":
        rows = _our_eval_rows(online_sub, matrix_sub, human_cls, human_qa)
        rows["task_type"] = online_sub["task_type"].tolist()
        return rows
    out = []
    for _, row in online_sub.iterrows():
        sid = str(row["sample_id"])
        mid = str(row["selected_model"])
        match = matrix_sub[(matrix_sub["sample_id"].astype(str) == sid) & (matrix_sub["model_id"].astype(str) == mid)]
        strict = int(match.iloc[0]["strict_correct"]) if not match.empty else 0
        out.append(
            {
                "sample_id": sid,
                "task_type": str(row["task_type"]),
                "model_id": mid,
                "cost_usd": float(row["selected_cost_usd"]),
                "correct": float(strict),
            }
        )
    return pd.DataFrame(out)


def _baseline_rows(matrix_df: pd.DataFrame, summary_df: pd.DataFrame, subset_rows: List[Dict], ids: Iterable[str], family: str, mode: str, judge_map: Dict[Tuple[str, str], int]) -> pd.DataFrame:
    ids = set(str(x) for x in ids)
    sub = matrix_df[matrix_df["sample_id"].astype(str).isin(ids)].copy()
    subset_task = [r for r in subset_rows if str(r["sample_id"]) in ids]
    probs = _train_routellm_probs(sub, subset_task, "qwen35_flash", "qwq_plus")
    rows = []
    params = [0.0]
    if family == "eagle_style":
        params = [float(x) for x in np.linspace(0.0, 2.0, 13)]
    elif family == "routellm_style":
        params = [float(x) for x in np.linspace(0.1, 0.9, 9)]
    best_df = None
    best_score = -1.0
    for param in params:
        chosen = []
        for sample_id, sample_rows in sub.groupby("sample_id"):
            sample_id = str(sample_id)
            available = sample_rows["model_id"].astype(str).tolist()
            if family == "eagle_style":
                ranked = _rank_eagle(sample_rows, summary_df, param)
            elif family == "routellm_style":
                ranked = _rank_routellm(sample_id, probs, param, "qwen35_flash", "qwq_plus", available)
            elif family == "oracle":
                ranked = _rank_oracle(sample_rows, judge_map if mode == "manual" else None)
            elif family == "random":
                ranked = _rank_random(available)
            else:
                raise ValueError(family)
            row = sample_rows[sample_rows["model_id"].astype(str) == ranked[0]].iloc[0]
            if mode == "manual" and str(row["task_type"]) in {"classification", "qa"}:
                correct = float(judge_map.get((sample_id, str(row["model_id"])), 0))
            else:
                correct = float(row["strict_correct"])
            if family == "random":
                # fixed seeded pseudo-random model pick by sample id hash
                idx = abs(hash(sample_id)) % len(available)
                chosen_model = sorted(available)[idx]
                row = sample_rows[sample_rows["model_id"].astype(str) == chosen_model].iloc[0]
                if mode == "manual" and str(row["task_type"]) in {"classification", "qa"}:
                    correct = float(judge_map.get((sample_id, str(row["model_id"])), 0))
                else:
                    correct = float(row["strict_correct"])
            chosen.append(
                {
                    "sample_id": sample_id,
                    "task_type": str(row["task_type"]),
                    "model_id": str(row["model_id"]),
                    "cost_usd": float(row["cost_usd"]),
                    "correct": correct,
                }
            )
        chosen_df = pd.DataFrame(chosen)
        score = float(chosen_df["correct"].mean())
        if score > best_score:
            best_score = score
            best_df = chosen_df.copy()
    return best_df


def _curve(df: pd.DataFrame, total_n: int) -> pd.DataFrame:
    work = df.sort_values(["cost_usd", "sample_id"]).copy().reset_index(drop=True)
    work["budget_total_usd"] = pd.to_numeric(work["cost_usd"], errors="coerce").fillna(0.0).cumsum()
    work["answered_samples"] = range(1, len(work) + 1)
    work["answered_accuracy"] = pd.to_numeric(work["correct"], errors="coerce").fillna(0.0).cumsum() / float(total_n)
    work["coverage"] = work["answered_samples"] / float(total_n)
    return work[["budget_total_usd", "answered_accuracy", "coverage", "answered_samples"]]


def _smooth_and_resample(df: pd.DataFrame, n_points: int = 12) -> pd.DataFrame:
    parts = []
    for method, sub in df.groupby("method"):
        sub = sub.sort_values("budget_total_usd").copy()
        iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
        sub["answered_accuracy"] = iso.fit_transform(sub["budget_total_usd"], sub["answered_accuracy"])
        xs = sub["budget_total_usd"].to_numpy(dtype=float)
        ys = sub["answered_accuracy"].to_numpy(dtype=float)
        lo, hi = float(xs.min()), float(xs.max())
        grid = np.linspace(lo, hi, max(8, int(n_points)))
        y_grid = np.interp(grid, xs, ys)
        cover_grid = np.interp(grid, xs, sub["coverage"].to_numpy(dtype=float))
        ans_grid = np.interp(grid, xs, sub["answered_samples"].to_numpy(dtype=float))
        parts.append(pd.DataFrame({
            "budget_total_usd": grid,
            "answered_accuracy": y_grid,
            "coverage": cover_grid,
            "answered_samples": ans_grid,
            "method": method,
        }))
    return pd.concat(parts, ignore_index=True)


def _plot(df: pd.DataFrame, methods: List[str], labels: Dict[str, str], title: str, out_png: Path) -> None:
    styles = {
        "ours": dict(color="#2f855a", marker="o", linestyle="-", linewidth=2.6, zorder=4, markersize=6, markeredgecolor="white", markeredgewidth=0.8, markevery=1),
        "eagle_style": dict(color="#d62728", marker="s", linestyle="-", linewidth=2.2, zorder=6, markersize=6, markeredgecolor="white", markeredgewidth=0.8, markevery=(0, 2)),
        "routellm_style": dict(color="#1f77b4", marker="^", linestyle="--", linewidth=2.2, zorder=5, markersize=6, markeredgecolor="white", markeredgewidth=0.8, markevery=(1, 2)),
        "random": dict(color="#7f7f7f", marker="D", linestyle="-", linewidth=2.0, zorder=2, markersize=5, markeredgecolor="white", markeredgewidth=0.8, markevery=1),
        "oracle": dict(color="#ff7f0e", marker="P", linestyle="-", linewidth=2.3, zorder=3, markersize=6, markeredgecolor="white", markeredgewidth=0.8, markevery=1),
    }
    plt.figure(figsize=(8.8, 5.5))
    for method in methods:
        sub = df[df["method"] == method].copy().sort_values("budget_total_usd")
        kw = styles[method].copy()
        plt.plot(sub["budget_total_usd"], sub["answered_accuracy"], label=labels[method], **kw)
    plt.xlabel("Total Cost (USD)")
    plt.ylabel("Overall Accuracy")
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.28)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()


def _build_plot(name: str, ids: List[str], mode_ours: str, families: List[Tuple[str, str]], title: str, subset_rows, matrix_df, online_df, summary_df, human_cls, human_qa, judge_map):
    total_n = len(ids)
    rows_parts = []
    ours = _our_rows(online_df, matrix_df, ids, human_cls, human_qa, mode_ours)
    c = _curve(ours, total_n)
    c["method"] = "ours"
    rows_parts.append(c)
    for family, mode in families:
        b = _baseline_rows(matrix_df, summary_df, subset_rows, ids, family, mode, judge_map)
        c = _curve(b, total_n)
        c["method"] = family
        rows_parts.append(c)
    df = _smooth_and_resample(pd.concat(rows_parts, ignore_index=True), n_points=12)
    df.to_csv(OUT_DIR / f"{name}.csv", index=False, encoding="utf-8-sig")
    labels = {
        "ours": "Ours",
        "eagle_style": "Eagle",
        "routellm_style": "RouteLLM-style",
        "random": "Random",
        "oracle": "Oracle",
    }
    methods = ["ours"] + [f for f, _ in families]
    _plot(df, methods, labels, title, OUT_DIR / f"{name}.png")


def main() -> None:
    subset_rows, matrix_df, online_df, summary_df, human_cls, human_qa, judge_map = _load_all()
    ids_cls = _task_ids(matrix_df, online_df, "classification")
    ids_qa = _task_ids(matrix_df, online_df, "qa")
    ids_reason = _task_ids(matrix_df, online_df, "reasoning")
    ids_combined = ids_cls + ids_qa + ids_reason

    _build_plot(
        "1_reasoning_ours_eagle_routellm",
        ids_reason,
        "strict",
        [("eagle_style", "strict"), ("routellm_style", "strict")],
        "reasoning: Ours vs Eagle vs RouteLLM-style",
        subset_rows, matrix_df, online_df, summary_df, human_cls, human_qa, judge_map,
    )
    _build_plot(
        "2_reasoning_ours_random_oracle",
        ids_reason,
        "strict",
        [("random", "strict"), ("oracle", "strict")],
        "reasoning: Ours vs Random vs Oracle",
        subset_rows, matrix_df, online_df, summary_df, human_cls, human_qa, judge_map,
    )
    _build_plot(
        "3_qa_manual_ours_eagle_routellm",
        ids_qa,
        "manual",
        [("eagle_style", "manual"), ("routellm_style", "manual")],
        "qa: manual Ours vs manual Eagle vs manual RouteLLM-style",
        subset_rows, matrix_df, online_df, summary_df, human_cls, human_qa, judge_map,
    )
    _build_plot(
        "4_qa_strict_ours_eagle_routellm",
        ids_qa,
        "strict",
        [("eagle_style", "strict"), ("routellm_style", "strict")],
        "qa: strict Ours vs strict Eagle vs strict RouteLLM-style",
        subset_rows, matrix_df, online_df, summary_df, human_cls, human_qa, judge_map,
    )
    _build_plot(
        "5_combined_manual_ours_strict_eagle_routellm",
        ids_combined,
        "manual",
        [("eagle_style", "strict"), ("routellm_style", "strict")],
        "classification + qa + reasoning: manual Ours vs strict Eagle / RouteLLM-style",
        subset_rows, matrix_df, online_df, summary_df, human_cls, human_qa, judge_map,
    )
    _build_plot(
        "6_combined_manual_ours_random_oracle",
        ids_combined,
        "manual",
        [("random", "manual"), ("oracle", "manual")],
        "classification + qa + reasoning: manual Ours vs manual Random / Oracle",
        subset_rows, matrix_df, online_df, summary_df, human_cls, human_qa, judge_map,
    )


if __name__ == "__main__":
    main()
