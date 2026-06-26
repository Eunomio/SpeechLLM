from __future__ import annotations

import math
import os
import sys
import warnings
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
M1_DIR = Path(r"E:\ECNU\SpeechLLM\M1")
OUT = ROOT / "outputs" / "m1_analysis"
FIG = OUT / "figures"
TABLE = OUT / "tables"
PYDEPS = ROOT / "analysis_pydeps"

(OUT / "mplconfig").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(OUT / "mplconfig"))
sys.path.insert(0, str(PYDEPS))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
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

DIALECT_LABEL = {
    "BEI": "Beijing",
    "CHD": "Chengdu",
    "JNN": "Jinan",
    "TYN": "Taiyuan",
    "WHN": "Wuhan",
    "XIA": "Xi'an",
}

MODEL_LABEL = {
    "qwen2.5": "Qwen2.5-Omni-7B",
    "qwen3": "Qwen3-Omni-Flash",
    "qwen3.5": "Qwen3.5-Omni-Plus",
}

MODEL_ORDER = ["qwen2.5", "qwen3", "qwen3.5"]
CONSTRUCT_ORDER = ["Ability", "Warmth", "Trust"]
DIALECT_ORDER = ["BEI", "CHD", "JNN", "TYN", "WHN", "XIA"]

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def ci95(values: pd.Series) -> pd.Series:
    x = values.dropna().to_numpy(dtype=float)
    n = len(x)
    mean = float(np.mean(x)) if n else np.nan
    if n < 2:
        return pd.Series({"mean": mean, "ci_low": np.nan, "ci_high": np.nan, "sd": np.nan, "n": n})
    se = float(np.std(x, ddof=1) / math.sqrt(n))
    tcrit = float(stats.t.ppf(0.975, n - 1))
    return pd.Series({"mean": mean, "ci_low": mean - tcrit * se, "ci_high": mean + tcrit * se, "sd": float(np.std(x, ddof=1)), "n": n})


def read_m1() -> tuple[pd.DataFrame, pd.DataFrame]:
    files = sorted(M1_DIR.rglob("m1_*.csv"))
    frames = []
    for f in files:
        model = f.parent.name
        df = pd.read_csv(f, encoding="utf-8-sig")
        if "run_id" not in df.columns:
            df["run_id"] = "1"
        df["run_id"] = df["run_id"].fillna("1").astype(str).replace({"": "1"})
        df["model"] = model
        df["source_file"] = f.name
        frames.append(df)
    raw = pd.concat(frames, ignore_index=True)
    raw["rating_num"] = pd.to_numeric(raw["rating"], errors="coerce")
    raw["ok_bool"] = raw["ok"].astype(str).eq("True")
    raw["construct"] = raw["dimension"].map(CONSTRUCT)
    raw["run_id"] = pd.to_numeric(raw["run_id"], errors="coerce").fillna(raw["run_id"]).astype(str)
    raw["run_id"] = raw["run_id"].str.replace(r"\.0$", "", regex=True)
    return raw, raw[raw["ok_bool"] & raw["rating_num"].notna()].copy()


def build_analysis_tables(ok: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # Qwen2.5 repeated runs are exact duplicates in this dataset, so retain run 1 to avoid
    # treating deterministic repeats as independent evidence.
    analysis = ok.copy()
    analysis = analysis[~((analysis["model"] == "qwen2.5") & (analysis["run_id"] != "1"))].copy()

    cell_cols = ["model", "speaker_id", "dialect", "guise", "text", "dimension", "construct"]
    cell = analysis.groupby(cell_cols, as_index=False)["rating_num"].mean()

    idx = ["model", "speaker_id", "dialect", "text", "dimension", "construct"]
    wide = cell.pivot_table(index=idx, columns="guise", values="rating_num", aggfunc="mean").reset_index()
    wide = wide.dropna(subset=["CMN", "DIA"]).copy()
    wide["bias"] = wide["DIA"] - wide["CMN"]
    wide["model_label"] = wide["model"].map(MODEL_LABEL)
    wide["dialect_label"] = wide["dialect"].map(DIALECT_LABEL)

    cu = (
        wide.groupby(["model", "speaker_id", "dialect", "text", "construct"], as_index=False)
        .agg(bias=("bias", "mean"), CMN=("CMN", "mean"), DIA=("DIA", "mean"))
    )
    cu["model_label"] = cu["model"].map(MODEL_LABEL)
    cu["dialect_label"] = cu["dialect"].map(DIALECT_LABEL)

    raw_construct = (
        cell.groupby(["model", "speaker_id", "dialect", "text", "construct", "guise"], as_index=False)
        .agg(rating=("rating_num", "mean"))
    )
    raw_construct["model_label"] = raw_construct["model"].map(MODEL_LABEL)
    raw_construct["dialect_label"] = raw_construct["dialect"].map(DIALECT_LABEL)
    return analysis, wide, cu, raw_construct


def save_data_audit(raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, source), g in raw.groupby(["model", "source_file"], sort=True):
        rows.append(
            {
                "model": model,
                "source_file": source,
                "rows": len(g),
                "ok_rows": int(g["ok_bool"].sum()),
                "unique_trials": int(g["trial_id"].nunique()),
                "duplicate_rows": int(len(g) - g["trial_id"].nunique()),
                "error_rows": int((~g["ok_bool"]).sum()),
            }
        )
    audit = pd.DataFrame(rows)

    match_rows = []
    for model, g in raw[raw["ok_bool"]].groupby("model"):
        by_run = {
            run: sub.drop_duplicates("trial_id").set_index("trial_id")["rating_num"]
            for run, sub in g.groupby("run_id")
        }
        runs = sorted(by_run)
        for a, b in zip(runs[:-1], runs[1:]):
            common = by_run[a].index.intersection(by_run[b].index)
            same = int((by_run[a].loc[common] == by_run[b].loc[common]).sum())
            match_rows.append(
                {
                    "model": model,
                    "run_a": a,
                    "run_b": b,
                    "common_trials": len(common),
                    "same_rating": same,
                    "same_rate": same / len(common) if len(common) else np.nan,
                }
            )
    run_match = pd.DataFrame(match_rows)
    audit.to_csv(TABLE / "m1_data_audit_by_file.csv", index=False, encoding="utf-8-sig")
    run_match.to_csv(TABLE / "m1_run_to_run_exact_match.csv", index=False, encoding="utf-8-sig")
    return audit, run_match


def summarise(wide: pd.DataFrame, cu: pd.DataFrame, raw_construct: pd.DataFrame) -> dict[str, pd.DataFrame]:
    core = (
        cu.groupby(["model", "construct", "dialect"])["bias"]
        .apply(ci95)
        .unstack()
        .reset_index()
    )
    overall = cu.groupby(["model", "construct"])["bias"].apply(ci95).unstack().reset_index()
    raw_summary = (
        raw_construct.groupby(["model", "construct", "guise"])["rating"]
        .apply(ci95)
        .unstack()
        .reset_index()
    )
    speaker = (
        cu[cu["model"] == "qwen3.5"]
        .groupby(["construct", "dialect", "speaker_id"])["bias"]
        .apply(ci95)
        .unstack()
        .reset_index()
    )
    sentence = (
        cu[cu["model"] == "qwen3.5"]
        .groupby(["construct", "text"])["bias"]
        .apply(ci95)
        .unstack()
        .reset_index()
    )
    dim = (
        wide[wide["model"] == "qwen3.5"]
        .groupby(["construct", "dimension", "dialect"])["bias"]
        .apply(ci95)
        .unstack()
        .reset_index()
    )

    outputs = {
        "core_bias_by_model_construct_dialect.csv": core,
        "overall_bias_by_model_construct.csv": overall,
        "raw_cmn_dia_by_model_construct.csv": raw_summary,
        "qwen35_speaker_bias.csv": speaker,
        "qwen35_sentence_bias.csv": sentence,
        "qwen35_dimension_bias.csv": dim,
    }
    for name, df in outputs.items():
        df.to_csv(TABLE / name, index=False, encoding="utf-8-sig")
    return outputs


def style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#e6e6e6", linewidth=0.8)
    ax.set_axisbelow(True)


def plot_core_heatmap(core: pd.DataFrame):
    pivot = core.copy()
    pivot["model"] = pd.Categorical(pivot["model"], MODEL_ORDER, ordered=True)
    pivot["construct"] = pd.Categorical(pivot["construct"], CONSTRUCT_ORDER, ordered=True)
    pivot["dialect"] = pd.Categorical(pivot["dialect"], DIALECT_ORDER, ordered=True)
    pivot = pivot.sort_values(["model", "construct", "dialect"])
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.8), sharey=True)
    vmin, vmax = -1.2, 0.35
    for ax, model in zip(axes, MODEL_ORDER):
        data = pivot[pivot["model"] == model].pivot(index="construct", columns="dialect", values="mean").loc[CONSTRUCT_ORDER, DIALECT_ORDER]
        im = ax.imshow(data.to_numpy(), cmap="RdBu_r", vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_title(MODEL_LABEL[model], fontsize=12, weight="bold")
        ax.set_xticks(range(len(DIALECT_ORDER)), DIALECT_ORDER, rotation=45, ha="right")
        ax.set_yticks(range(len(CONSTRUCT_ORDER)), CONSTRUCT_ORDER)
        for i, cons in enumerate(CONSTRUCT_ORDER):
            for j, dia in enumerate(DIALECT_ORDER):
                val = data.loc[cons, dia]
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=9, color="black")
    fig.subplots_adjust(right=0.88, wspace=0.16, top=0.82, bottom=0.18)
    cax = fig.add_axes([0.90, 0.23, 0.018, 0.55])
    fig.colorbar(im, cax=cax, label="Bias score (DIA - CMN)")
    fig.suptitle("M1 core bias by model, construct, and dialect", y=1.02, fontsize=14, weight="bold")
    fig.savefig(FIG / "m1_core_bias_heatmap.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_raw(raw_summary: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6), sharey=True)
    colors = {"CMN": "#4C78A8", "DIA": "#F58518"}
    for ax, model in zip(axes, MODEL_ORDER):
        sub = raw_summary[raw_summary["model"] == model]
        x = np.arange(len(CONSTRUCT_ORDER))
        width = 0.33
        for offset, guise in [(-width / 2, "CMN"), (width / 2, "DIA")]:
            g = sub[sub["guise"] == guise].set_index("construct").loc[CONSTRUCT_ORDER]
            y = g["mean"].to_numpy()
            lo = y - g["ci_low"].to_numpy()
            hi = g["ci_high"].to_numpy() - y
            ax.bar(x + offset, y, width, label=guise, color=colors[guise], alpha=0.88)
            ax.errorbar(x + offset, y, yerr=[lo, hi], fmt="none", ecolor="#333333", capsize=3, linewidth=1)
        ax.set_title(MODEL_LABEL[model], fontsize=12, weight="bold")
        ax.set_xticks(x, CONSTRUCT_ORDER)
        ax.set_ylim(4.3, 6.15)
        style_axes(ax)
    axes[0].set_ylabel("Mean rating (1-7)")
    axes[-1].legend(frameon=False, loc="upper right")
    fig.suptitle("Raw M1 ratings: Putonghua (CMN) vs dialect-accented speech (DIA)", y=1.02, fontsize=14, weight="bold")
    fig.tight_layout()
    fig.savefig(FIG / "m1_raw_cmn_vs_dia.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_speaker(speaker: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(14, 6.6), sharex=True)
    for ax, cons in zip(axes, CONSTRUCT_ORDER):
        sub = speaker[speaker["construct"] == cons].copy()
        sub["dialect"] = pd.Categorical(sub["dialect"], DIALECT_ORDER, ordered=True)
        sub = sub.sort_values(["dialect", "speaker_id"])
        y = np.arange(len(sub))
        ax.axvline(0, color="#555555", linewidth=1)
        ax.errorbar(
            sub["mean"], y,
            xerr=[sub["mean"] - sub["ci_low"], sub["ci_high"] - sub["mean"]],
            fmt="o", color="#2F5597", ecolor="#888888", capsize=2, markersize=4,
        )
        ax.set_yticks(y, sub["speaker_id"])
        ax.invert_yaxis()
        ax.set_title(cons, fontsize=12, weight="bold")
        ax.set_xlim(-1.8, 0.6)
        ax.grid(axis="x", color="#e6e6e6", linewidth=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_xlabel("Bias score (DIA - CMN)")
    axes[1].set_xlabel("Bias score (DIA - CMN)")
    axes[2].set_xlabel("Bias score (DIA - CMN)")
    fig.suptitle("Qwen3.5 speaker-level M1 bias", y=1.01, fontsize=14, weight="bold")
    fig.tight_layout()
    fig.savefig(FIG / "m1_qwen35_speaker_forest.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_sentence(sentence: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.8), sharex=True)
    for ax, cons in zip(axes, CONSTRUCT_ORDER):
        sub = sentence[sentence["construct"] == cons].sort_values("mean")
        y = np.arange(len(sub))
        ax.axvline(0, color="#555555", linewidth=1)
        ax.errorbar(
            sub["mean"], y,
            xerr=[sub["mean"] - sub["ci_low"], sub["ci_high"] - sub["mean"]],
            fmt="o", color="#7A3E9D", ecolor="#999999", capsize=2, markersize=4,
        )
        ax.set_yticks(y, sub["text"])
        ax.invert_yaxis()
        ax.set_title(cons, fontsize=12, weight="bold")
        ax.set_xlim(-1.35, 0.35)
        ax.grid(axis="x", color="#e6e6e6", linewidth=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    for ax in axes:
        ax.set_xlabel("Bias score (DIA - CMN)")
    fig.suptitle("Qwen3.5 sentence-level M1 bias", y=1.01, fontsize=14, weight="bold")
    fig.tight_layout()
    fig.savefig(FIG / "m1_qwen35_sentence_bias.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def fit_models(cu: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    model_df = cu.copy()
    for col, order in [("model", MODEL_ORDER), ("construct", CONSTRUCT_ORDER), ("dialect", DIALECT_ORDER)]:
        model_df[col] = pd.Categorical(model_df[col], categories=order, ordered=False)

    vc = {"text": "0 + C(text)"}
    formulas = {
        "full": "bias ~ C(model) * C(construct) * C(dialect)",
        "no_three_way": "bias ~ C(model) * C(construct) + C(model) * C(dialect) + C(construct) * C(dialect)",
        "no_model_construct": "bias ~ C(model) + C(construct) + C(dialect) + C(model) * C(dialect) + C(construct) * C(dialect)",
        "no_model_dialect": "bias ~ C(model) + C(construct) + C(dialect) + C(model) * C(construct) + C(construct) * C(dialect)",
        "no_construct_dialect": "bias ~ C(model) + C(construct) + C(dialect) + C(model) * C(construct) + C(model) * C(dialect)",
        "main_effects": "bias ~ C(model) + C(construct) + C(dialect)",
    }

    fits = {}
    fit_notes = []
    warnings.filterwarnings("ignore")
    for name, formula in formulas.items():
        try:
            md = smf.mixedlm(formula, model_df, groups=model_df["speaker_id"], re_formula="1", vc_formula=vc)
            fits[name] = md.fit(reml=False, method="lbfgs", maxiter=1000, disp=False)
        except Exception as exc:
            fit_notes.append(f"{name}: {type(exc).__name__}: {exc}")

    rows = []
    full = fits.get("full")
    for name, fit in fits.items():
        rows.append(
            {
                "model_spec": name,
                "nobs": int(fit.nobs),
                "df_modelwc": float(fit.df_modelwc),
                "llf": float(fit.llf),
                "aic": float(fit.aic),
                "bic": float(fit.bic),
                "converged": bool(fit.converged),
            }
        )
    fit_table = pd.DataFrame(rows)
    fit_table.to_csv(TABLE / "mixedlm_model_fit_indices.csv", index=False, encoding="utf-8-sig")

    lrt_rows = []
    if full is not None:
        for name, reduced in fits.items():
            if name == "full":
                continue
            lr = 2 * (full.llf - reduced.llf)
            df = full.df_modelwc - reduced.df_modelwc
            p = stats.chi2.sf(lr, df) if df > 0 else np.nan
            lrt_rows.append({"comparison": f"full vs {name}", "lr_chisq": lr, "df": df, "p": p})
    lrt = pd.DataFrame(lrt_rows)
    lrt.to_csv(TABLE / "mixedlm_likelihood_ratio_tests.csv", index=False, encoding="utf-8-sig")

    if full is not None:
        with open(TABLE / "mixedlm_full_summary.txt", "w", encoding="utf-8") as f:
            f.write(str(full.summary()))
    if fit_notes:
        with open(TABLE / "mixedlm_fit_notes.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(fit_notes))
    return lrt, "\n".join(fit_notes)


def fmt_num(x, digits=3):
    if pd.isna(x):
        return ""
    return f"{float(x):.{digits}f}"


def md_table(df: pd.DataFrame, columns: list[str]) -> str:
    d = df[columns].copy()
    d = d.fillna("")
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


def write_report(audit, run_match, summaries, lrt, fit_notes):
    core = summaries["core_bias_by_model_construct_dialect.csv"].copy()
    overall = summaries["overall_bias_by_model_construct.csv"].copy()
    raw_summary = summaries["raw_cmn_dia_by_model_construct.csv"].copy()
    speaker = summaries["qwen35_speaker_bias.csv"].copy()
    sentence = summaries["qwen35_sentence_bias.csv"].copy()
    dim = summaries["qwen35_dimension_bias.csv"].copy()

    overall["model_name"] = overall["model"].map(MODEL_LABEL)
    overall["bias_ci"] = overall.apply(lambda r: f"{r['mean']:.3f} [{r['ci_low']:.3f}, {r['ci_high']:.3f}]", axis=1)
    raw_summary["model_name"] = raw_summary["model"].map(MODEL_LABEL)
    raw_summary["rating_ci"] = raw_summary.apply(lambda r: f"{r['mean']:.3f} [{r['ci_low']:.3f}, {r['ci_high']:.3f}]", axis=1)

    core_wide = core.pivot_table(index=["model", "construct"], columns="dialect", values="mean").reset_index()
    core_wide["model_name"] = core_wide["model"].map(MODEL_LABEL)
    for dia in DIALECT_ORDER:
        core_wide[dia] = core_wide[dia].map(lambda x: f"{x:.3f}")

    fit_indices = pd.read_csv(TABLE / "mixedlm_model_fit_indices.csv")
    fit_indices["converged_text"] = fit_indices["converged"].map(lambda x: "yes" if bool(x) else "no")

    lrt_fmt = lrt.copy()
    if not lrt_fmt.empty:
        lrt_fmt["lr_chisq"] = lrt_fmt["lr_chisq"].map(lambda x: f"{x:.2f}")
        lrt_fmt["df"] = lrt_fmt["df"].map(lambda x: f"{x:.0f}")
        lrt_fmt["p"] = lrt_fmt["p"].map(lambda x: "<.001" if x < 0.001 else f"{x:.4f}")

    top_speaker = speaker[speaker["construct"] == "Ability"].sort_values("mean").head(6).copy()
    top_sentence = sentence[sentence["construct"] == "Ability"].sort_values("mean").head(5).copy()
    for df in [top_speaker, top_sentence]:
        df["bias_ci"] = df.apply(lambda r: f"{r['mean']:.3f} [{r['ci_low']:.3f}, {r['ci_high']:.3f}]", axis=1)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    TABLE.mkdir(parents=True, exist_ok=True)

    raw, ok = read_m1()
    audit, run_match = save_data_audit(raw)
    analysis, wide, cu, raw_construct = build_analysis_tables(ok)

    analysis.to_csv(TABLE / "m1_success_rows_for_analysis.csv", index=False, encoding="utf-8-sig")
    wide.to_csv(TABLE / "m1_dimension_pairs_clean.csv", index=False, encoding="utf-8-sig")
    cu.to_csv(TABLE / "m1_construct_units_clean.csv", index=False, encoding="utf-8-sig")
    raw_construct.to_csv(TABLE / "m1_raw_construct_ratings_clean.csv", index=False, encoding="utf-8-sig")

    summaries = summarise(wide, cu, raw_construct)
    plot_core_heatmap(summaries["core_bias_by_model_construct_dialect.csv"])
    plot_raw(summaries["raw_cmn_dia_by_model_construct.csv"])
    plot_speaker(summaries["qwen35_speaker_bias.csv"])
    plot_sentence(summaries["qwen35_sentence_bias.csv"])
    lrt, fit_notes = fit_models(cu)
    write_report(audit, run_match, summaries, lrt, fit_notes)
    print(f"Wrote report: {OUT / 'M1_further_analysis_report.md'}")


if __name__ == "__main__":
    main()
