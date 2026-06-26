from __future__ import annotations

import math
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
M2_DIR = Path(r"E:\ECNU\SpeechLLM\M2")
OUT = ROOT / "outputs" / "m2_analysis"
FIG = OUT / "figures"
TABLE = OUT / "tables"
PYDEPS = ROOT / "analysis_pydeps"

(OUT / "mplconfig").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(OUT / "mplconfig"))
sys.path.insert(0, str(PYDEPS))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


CONSTRUCT = {
    "competence": "Ability",
    "educatedness": "Ability",
    "professionalism": "Ability",
    "friendliness": "Warmth",
    "likability": "Warmth",
    "approachability": "Warmth",
    "trustworthiness": "Trust",
}

MODEL_LABEL = {
    "qwen2.5": "Qwen2.5-Omni-7B",
    "qwen3": "Qwen3-Omni-Flash",
    "qwen3.5": "Qwen3.5-Omni-Plus",
}

DIALECT_LABEL = {
    "BEI": "Beijing",
    "CHD": "Chengdu",
    "JNN": "Jinan",
    "TYN": "Taiyuan",
    "WHN": "Wuhan",
    "XIA": "Xi'an",
}

MODEL_ORDER = ["qwen2.5", "qwen3", "qwen3.5"]
CONSTRUCT_ORDER = ["Ability", "Warmth", "Trust"]
DIALECT_ORDER = ["BEI", "CHD", "JNN", "TYN", "WHN", "XIA"]

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def ci95(x: pd.Series) -> pd.Series:
    vals = x.dropna().to_numpy(dtype=float)
    n = len(vals)
    mean = float(np.mean(vals)) if n else np.nan
    if n < 2:
        return pd.Series({"mean": mean, "ci_low": np.nan, "ci_high": np.nan, "sd": np.nan, "n": n})
    se = float(np.std(vals, ddof=1) / math.sqrt(n))
    tcrit = float(stats.t.ppf(0.975, n - 1))
    return pd.Series({"mean": mean, "ci_low": mean - tcrit * se, "ci_high": mean + tcrit * se, "sd": float(np.std(vals, ddof=1)), "n": n})


def read_m2() -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = []
    for f in sorted(M2_DIR.rglob("m2_*.csv")):
        df = pd.read_csv(f, encoding="utf-8-sig")
        df["model"] = f.parent.name
        df["source_file"] = f.name
        frames.append(df)
    raw = pd.concat(frames, ignore_index=True)
    raw["run_id"] = pd.to_numeric(raw["run_id"], errors="coerce").fillna(raw["run_id"]).astype(str)
    raw["run_id"] = raw["run_id"].str.replace(r"\.0$", "", regex=True)
    raw["ok_bool"] = raw["ok"].astype(str).eq("True")
    raw["construct"] = raw["dimension"].map(CONSTRUCT)
    raw["choice_dia"] = raw["chosen_guise"].eq("DIA").astype(float)
    raw["choice_a"] = raw["choice"].eq("A").astype(float)
    raw["confidence_num"] = pd.to_numeric(raw["confidence"], errors="coerce")
    return raw, raw[raw["ok_bool"]].copy()


def audit(raw: pd.DataFrame, ok: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    by_file = []
    for (model, source), g in raw.groupby(["model", "source_file"], sort=True):
        by_file.append({
            "model": model,
            "source_file": source,
            "rows": len(g),
            "ok_rows": int(g["ok_bool"].sum()),
            "unique_trials": int(g["trial_id"].nunique()),
            "duplicate_rows": int(len(g) - g["trial_id"].nunique()),
            "error_rows": int((~g["ok_bool"]).sum()),
        })
    file_audit = pd.DataFrame(by_file)

    model_audit = []
    for model, g in ok.groupby("model"):
        model_audit.append({
            "model": model,
            "ok_rows": len(g),
            "runs": int(g["run_id"].nunique()),
            "choice_A_rate": float(g["choice_a"].mean()),
            "choice_B_rate": float(1 - g["choice_a"].mean()),
            "dia_choice_rate": float(g["choice_dia"].mean()),
            "mean_confidence": float(g["confidence_num"].mean()),
            "cmnxdia_dia_rate": float(g[g["order_label"] == "CMNxDIA"]["choice_dia"].mean()),
            "diaxcmn_dia_rate": float(g[g["order_label"] == "DIAxCMN"]["choice_dia"].mean()),
        })
    model_audit = pd.DataFrame(model_audit)

    run_match = []
    for model, g in ok.groupby("model"):
        by_run = {
            run: sub.drop_duplicates("trial_id").set_index("trial_id")["chosen_guise"]
            for run, sub in g.groupby("run_id")
        }
        runs = sorted(by_run)
        for a, b in zip(runs[:-1], runs[1:]):
            common = by_run[a].index.intersection(by_run[b].index)
            same = int((by_run[a].loc[common] == by_run[b].loc[common]).sum())
            run_match.append({
                "model": model,
                "run_a": a,
                "run_b": b,
                "common_trials": len(common),
                "same_chosen_guise": same,
                "same_rate": same / len(common) if len(common) else np.nan,
            })
    run_match = pd.DataFrame(run_match)
    file_audit.to_csv(TABLE / "m2_data_audit_by_file.csv", index=False, encoding="utf-8-sig")
    model_audit.to_csv(TABLE / "m2_model_level_position_audit.csv", index=False, encoding="utf-8-sig")
    run_match.to_csv(TABLE / "m2_run_to_run_exact_match.csv", index=False, encoding="utf-8-sig")
    return file_audit, model_audit, run_match


def build_units(ok: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Average across repeated runs for each concrete trial condition.
    cols = ["model", "speaker_id", "dialect", "text", "dimension", "construct", "order_label", "A_guise", "B_guise"]
    unit = (
        ok.groupby(cols, as_index=False)
        .agg(choice_dia=("choice_dia", "mean"), choice_a=("choice_a", "mean"), confidence=("confidence_num", "mean"))
    )
    # Construct-level pair/order units.
    construct_unit = (
        unit.groupby(["model", "speaker_id", "dialect", "text", "construct", "order_label"], as_index=False)
        .agg(choice_dia=("choice_dia", "mean"), choice_a=("choice_a", "mean"), confidence=("confidence", "mean"))
    )
    unit.to_csv(TABLE / "m2_dimension_units_clean.csv", index=False, encoding="utf-8-sig")
    construct_unit.to_csv(TABLE / "m2_construct_units_clean.csv", index=False, encoding="utf-8-sig")
    return unit, construct_unit


def summarise(unit: pd.DataFrame, construct_unit: pd.DataFrame) -> dict[str, pd.DataFrame]:
    construct = (
        construct_unit.groupby(["model", "construct", "dialect"])["choice_dia"]
        .apply(ci95).unstack().reset_index()
    )
    construct["centered_mean"] = construct["mean"] - 0.5
    construct["centered_low"] = construct["ci_low"] - 0.5
    construct["centered_high"] = construct["ci_high"] - 0.5

    overall = construct_unit.groupby(["model", "construct"])["choice_dia"].apply(ci95).unstack().reset_index()
    overall["centered_mean"] = overall["mean"] - 0.5
    overall["centered_low"] = overall["ci_low"] - 0.5
    overall["centered_high"] = overall["ci_high"] - 0.5

    order = (
        construct_unit.groupby(["model", "construct", "order_label"])["choice_dia"]
        .apply(ci95).unstack().reset_index()
    )

    dimension = (
        unit.groupby(["model", "dimension", "dialect"])["choice_dia"]
        .apply(ci95).unstack().reset_index()
    )
    dimension["centered_mean"] = dimension["mean"] - 0.5

    speaker = (
        construct_unit[construct_unit["model"] == "qwen3.5"]
        .groupby(["construct", "dialect", "speaker_id"])["choice_dia"]
        .apply(ci95).unstack().reset_index()
    )
    speaker["centered_mean"] = speaker["mean"] - 0.5
    speaker["centered_low"] = speaker["ci_low"] - 0.5
    speaker["centered_high"] = speaker["ci_high"] - 0.5

    outputs = {
        "m2_choice_by_model_construct_dialect.csv": construct,
        "m2_choice_overall_by_model_construct.csv": overall,
        "m2_choice_by_model_construct_order.csv": order,
        "m2_choice_by_model_dimension_dialect.csv": dimension,
        "m2_qwen35_speaker_choice.csv": speaker,
    }
    for name, df in outputs.items():
        df.to_csv(TABLE / name, index=False, encoding="utf-8-sig")
    return outputs


def plot_position(model_audit: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    sub = model_audit.set_index("model").loc[MODEL_ORDER].reset_index()
    x = np.arange(len(sub))
    ax.bar(x - 0.18, sub["choice_A_rate"], 0.36, label="Choice A", color="#4C78A8")
    ax.bar(x + 0.18, sub["dia_choice_rate"], 0.36, label="DIA chosen", color="#F58518")
    ax.axhline(0.5, color="#555555", linewidth=1, linestyle="--")
    ax.set_xticks(x, [MODEL_LABEL[m] for m in sub["model"]], rotation=12, ha="right")
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Rate")
    ax.set_title("M2 task audit: position choice vs DIA choice", weight="bold")
    ax.legend(frameon=False)
    ax.grid(axis="y", color="#e6e6e6")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "m2_position_audit.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_core(summary: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.8), sharey=True)
    vmin, vmax = -0.5, 0.15
    for ax, model in zip(axes, MODEL_ORDER):
        data = summary[summary["model"] == model].pivot(index="construct", columns="dialect", values="centered_mean")
        data = data.loc[CONSTRUCT_ORDER, DIALECT_ORDER]
        im = ax.imshow(data.to_numpy(), cmap="RdBu_r", vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_title(MODEL_LABEL[model], fontsize=12, weight="bold")
        ax.set_xticks(range(len(DIALECT_ORDER)), DIALECT_ORDER, rotation=45, ha="right")
        ax.set_yticks(range(len(CONSTRUCT_ORDER)), CONSTRUCT_ORDER)
        for i, cons in enumerate(CONSTRUCT_ORDER):
            for j, dia in enumerate(DIALECT_ORDER):
                ax.text(j, i, f"{data.loc[cons, dia]:.2f}", ha="center", va="center", fontsize=9)
    fig.subplots_adjust(right=0.88, wspace=0.16, top=0.82, bottom=0.18)
    cax = fig.add_axes([0.90, 0.23, 0.018, 0.55])
    fig.colorbar(im, cax=cax, label="DIA choice probability - 0.5")
    fig.suptitle("M2 DIA-choice bias by model, construct, and dialect", y=1.02, fontsize=14, weight="bold")
    fig.savefig(FIG / "m2_core_choice_heatmap.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_qwen35_speaker(speaker: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(14, 6.6), sharex=True)
    for ax, cons in zip(axes, CONSTRUCT_ORDER):
        sub = speaker[speaker["construct"] == cons].copy()
        sub["dialect"] = pd.Categorical(sub["dialect"], DIALECT_ORDER, ordered=True)
        sub = sub.sort_values(["dialect", "speaker_id"])
        y = np.arange(len(sub))
        ax.axvline(0, color="#555555", linewidth=1)
        ax.errorbar(
            sub["centered_mean"], y,
            xerr=[sub["centered_mean"] - sub["centered_low"], sub["centered_high"] - sub["centered_mean"]],
            fmt="o", color="#2F5597", ecolor="#888888", capsize=2, markersize=4,
        )
        ax.set_yticks(y, sub["speaker_id"])
        ax.invert_yaxis()
        ax.set_title(cons, fontsize=12, weight="bold")
        ax.set_xlim(-0.55, 0.2)
        ax.grid(axis="x", color="#e6e6e6")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xlabel("DIA choice probability - 0.5")
    fig.suptitle("Qwen3.5 speaker-level M2 DIA-choice bias", y=1.01, fontsize=14, weight="bold")
    fig.tight_layout()
    fig.savefig(FIG / "m2_qwen35_speaker_choice.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def fmt_table(df: pd.DataFrame, columns: list[str]) -> str:
    d = df[columns].copy().fillna("")
    widths = []
    for col in columns:
        vals = [str(col)] + [str(x) for x in d[col].tolist()]
        widths.append(max(len(v) for v in vals))
    header = "| " + " | ".join(str(col).ljust(widths[i]) for i, col in enumerate(columns)) + " |"
    sep = "| " + " | ".join("-" * widths[i] for i in range(len(columns))) + " |"
    rows = []
    for _, row in d.iterrows():
        rows.append("| " + " | ".join(str(row[col]).ljust(widths[i]) for i, col in enumerate(columns)) + " |")
    return "\n".join([header, sep] + rows)


def write_report(model_audit, run_match, summaries):
    overall = summaries["m2_choice_overall_by_model_construct.csv"].copy()
    overall["model_name"] = overall["model"].map(MODEL_LABEL)
    overall["dia_rate_ci"] = overall.apply(lambda r: f"{r['mean']:.3f} [{r['ci_low']:.3f}, {r['ci_high']:.3f}]", axis=1)
    overall["centered"] = overall["centered_mean"].map(lambda x: f"{x:.3f}")

    core = summaries["m2_choice_by_model_construct_dialect.csv"].copy()
    core_wide = core.pivot_table(index=["model", "construct"], columns="dialect", values="centered_mean").reset_index()
    core_wide["model_name"] = core_wide["model"].map(MODEL_LABEL)
    for dia in DIALECT_ORDER:
        core_wide[dia] = core_wide[dia].map(lambda x: f"{x:.3f}")

    model_audit = model_audit.copy()
    model_audit["model_name"] = model_audit["model"].map(MODEL_LABEL)
    for col in ["choice_A_rate", "choice_B_rate", "dia_choice_rate", "mean_confidence", "cmnxdia_dia_rate", "diaxcmn_dia_rate"]:
        model_audit[col] = model_audit[col].map(lambda x: f"{x:.3f}")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    TABLE.mkdir(parents=True, exist_ok=True)
    raw, ok = read_m2()
    raw.to_csv(TABLE / "m2_raw_all_rows.csv", index=False, encoding="utf-8-sig")
    ok.to_csv(TABLE / "m2_ok_rows.csv", index=False, encoding="utf-8-sig")
    file_audit, model_audit, run_match = audit(raw, ok)
    unit, construct_unit = build_units(ok)
    summaries = summarise(unit, construct_unit)
    plot_position(model_audit)
    plot_core(summaries["m2_choice_by_model_construct_dialect.csv"])
    plot_qwen35_speaker(summaries["m2_qwen35_speaker_choice.csv"])
    write_report(model_audit, run_match, summaries)
    print(f"Wrote report: {OUT / 'M2_current_results_report.md'}")


if __name__ == "__main__":
    main()
