from __future__ import annotations

import math
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
M1_DIR = Path(r"E:\ECNU\SpeechLLM\M1")
OUT = ROOT / "outputs" / "m1_analysis_no_audit"
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


def read_ok() -> pd.DataFrame:
    frames = []
    for f in sorted(M1_DIR.rglob("m1_*.csv")):
        df = pd.read_csv(f, encoding="utf-8-sig")
        if "run_id" not in df.columns:
            df["run_id"] = "1"
        df["run_id"] = df["run_id"].fillna("1").astype(str).replace({"": "1"})
        df["run_id"] = pd.to_numeric(df["run_id"], errors="coerce").fillna(df["run_id"]).astype(str)
        df["run_id"] = df["run_id"].str.replace(r"\.0$", "", regex=True)
        df["model"] = f.parent.name
        df["source_file"] = f.name
        frames.append(df)
    raw = pd.concat(frames, ignore_index=True)
    raw["rating_num"] = pd.to_numeric(raw["rating"], errors="coerce")
    raw["ok_bool"] = raw["ok"].astype(str).eq("True")
    raw["construct"] = raw["dimension"].map(CONSTRUCT)
    raw.to_csv(TABLE / "m1_no_audit_raw_all_rows.csv", index=False, encoding="utf-8-sig")
    ok = raw[raw["ok_bool"] & raw["rating_num"].notna()].copy()
    ok.to_csv(TABLE / "m1_no_audit_ok_rows.csv", index=False, encoding="utf-8-sig")
    return ok


def build_pairs(ok: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # No-audit mode: keep every successful run as an observation.
    idx = ["model", "run_id", "speaker_id", "dialect", "text", "dimension", "construct"]
    wide = ok.pivot_table(index=idx, columns="guise", values="rating_num", aggfunc="mean").reset_index()
    wide = wide.dropna(subset=["CMN", "DIA"]).copy()
    wide["bias"] = wide["DIA"] - wide["CMN"]

    cu = (
        wide.groupby(["model", "run_id", "speaker_id", "dialect", "text", "construct"], as_index=False)
        .agg(bias=("bias", "mean"), CMN=("CMN", "mean"), DIA=("DIA", "mean"))
    )

    raw_construct = (
        ok.groupby(["model", "run_id", "speaker_id", "dialect", "text", "construct", "guise"], as_index=False)
        .agg(rating=("rating_num", "mean"))
    )
    wide.to_csv(TABLE / "m1_no_audit_dimension_pairs.csv", index=False, encoding="utf-8-sig")
    cu.to_csv(TABLE / "m1_no_audit_construct_units.csv", index=False, encoding="utf-8-sig")
    raw_construct.to_csv(TABLE / "m1_no_audit_raw_construct_ratings.csv", index=False, encoding="utf-8-sig")
    return wide, cu, raw_construct


def summarise(wide: pd.DataFrame, cu: pd.DataFrame, raw_construct: pd.DataFrame) -> dict[str, pd.DataFrame]:
    core = cu.groupby(["model", "construct", "dialect"])["bias"].apply(ci95).unstack().reset_index()
    overall = cu.groupby(["model", "construct"])["bias"].apply(ci95).unstack().reset_index()
    raw_summary = raw_construct.groupby(["model", "construct", "guise"])["rating"].apply(ci95).unstack().reset_index()
    speaker = (
        cu[cu["model"] == "qwen3.5"]
        .groupby(["construct", "dialect", "speaker_id"])["bias"]
        .apply(ci95).unstack().reset_index()
    )
    sentence = (
        cu[cu["model"] == "qwen3.5"]
        .groupby(["construct", "text"])["bias"]
        .apply(ci95).unstack().reset_index()
    )
    outputs = {
        "no_audit_core_bias_by_model_construct_dialect.csv": core,
        "no_audit_overall_bias_by_model_construct.csv": overall,
        "no_audit_raw_cmn_dia_by_model_construct.csv": raw_summary,
        "no_audit_qwen35_speaker_bias.csv": speaker,
        "no_audit_qwen35_sentence_bias.csv": sentence,
    }
    for name, df in outputs.items():
        df.to_csv(TABLE / name, index=False, encoding="utf-8-sig")
    return outputs


def plot_core(core: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.8), sharey=True)
    vmin, vmax = -1.2, 0.35
    for ax, model in zip(axes, MODEL_ORDER):
        data = core[core["model"] == model].pivot(index="construct", columns="dialect", values="mean")
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
    fig.colorbar(im, cax=cax, label="Bias score (DIA - CMN)")
    fig.suptitle("M1 no-audit bias by model, construct, and dialect", y=1.02, fontsize=14, weight="bold")
    fig.savefig(FIG / "m1_no_audit_core_bias_heatmap.png", dpi=220, bbox_inches="tight")
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
        ax.grid(axis="y", color="#e6e6e6")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_ylabel("Mean rating (1-7)")
    axes[-1].legend(frameon=False, loc="upper right")
    fig.suptitle("M1 no-audit raw ratings: CMN vs DIA", y=1.02, fontsize=14, weight="bold")
    fig.tight_layout()
    fig.savefig(FIG / "m1_no_audit_raw_cmn_vs_dia.png", dpi=220, bbox_inches="tight")
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
        ax.grid(axis="x", color="#e6e6e6")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xlabel("Bias score (DIA - CMN)")
    fig.suptitle("Qwen3.5 no-audit speaker-level M1 bias", y=1.01, fontsize=14, weight="bold")
    fig.tight_layout()
    fig.savefig(FIG / "m1_no_audit_qwen35_speaker_forest.png", dpi=220, bbox_inches="tight")
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
        ax.set_xlabel("Bias score (DIA - CMN)")
    fig.suptitle("Qwen3.5 no-audit sentence-level M1 bias", y=1.01, fontsize=14, weight="bold")
    fig.tight_layout()
    fig.savefig(FIG / "m1_no_audit_qwen35_sentence_bias.png", dpi=220, bbox_inches="tight")
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


def fit_mixed_models(cu: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    model_df = cu.copy()
    for col, order in [("model", MODEL_ORDER), ("construct", CONSTRUCT_ORDER), ("dialect", DIALECT_ORDER)]:
        model_df[col] = pd.Categorical(model_df[col], categories=order, ordered=False)

    # Keep the model structure identical to the audited M1 analysis. The only
    # difference in this script is the data construction: repeated runs remain
    # as observations instead of being folded or averaged.
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
    notes = []
    for name, formula in formulas.items():
        try:
            md = smf.mixedlm(formula, model_df, groups=model_df["speaker_id"], re_formula="1", vc_formula=vc)
            fits[name] = md.fit(reml=False, method="lbfgs", maxiter=1000, disp=False)
        except Exception as exc:
            notes.append(f"{name}: {type(exc).__name__}: {exc}")

    fit_rows = []
    for name, fit in fits.items():
        fit_rows.append({
            "model_spec": name,
            "nobs": int(fit.nobs),
            "df_modelwc": float(fit.df_modelwc),
            "llf": float(fit.llf),
            "aic": float(fit.aic),
            "bic": float(fit.bic),
            "converged": bool(fit.converged),
        })
    fit_table = pd.DataFrame(fit_rows)
    fit_table.to_csv(TABLE / "no_audit_mixedlm_model_fit_indices.csv", index=False, encoding="utf-8-sig")

    lrt_rows = []
    full = fits.get("full")
    if full is not None:
        for name, reduced in fits.items():
            if name == "full":
                continue
            lr = 2 * (full.llf - reduced.llf)
            df = full.df_modelwc - reduced.df_modelwc
            p = stats.chi2.sf(lr, df) if df > 0 else np.nan
            lrt_rows.append({"comparison": f"full vs {name}", "lr_chisq": lr, "df": df, "p": p})
        with open(TABLE / "no_audit_mixedlm_full_summary.txt", "w", encoding="utf-8") as f:
            f.write(str(full.summary()))
    lrt = pd.DataFrame(lrt_rows)
    lrt.to_csv(TABLE / "no_audit_mixedlm_likelihood_ratio_tests.csv", index=False, encoding="utf-8-sig")
    if notes:
        (TABLE / "no_audit_mixedlm_fit_notes.txt").write_text("\n".join(notes), encoding="utf-8")
    return fit_table, lrt, "\n".join(notes)


def write_report(summaries: dict[str, pd.DataFrame], fit_table: pd.DataFrame, lrt: pd.DataFrame, fit_notes: str):
    overall = summaries["no_audit_overall_bias_by_model_construct.csv"].copy()
    overall["model_name"] = overall["model"].map(MODEL_LABEL)
    overall["bias_ci"] = overall.apply(lambda r: f"{r['mean']:.3f} [{r['ci_low']:.3f}, {r['ci_high']:.3f}]", axis=1)

    core = summaries["no_audit_core_bias_by_model_construct_dialect.csv"].copy()
    core_wide = core.pivot_table(index=["model", "construct"], columns="dialect", values="mean").reset_index()
    core_wide["model_name"] = core_wide["model"].map(MODEL_LABEL)
    for dia in DIALECT_ORDER:
        core_wide[dia] = core_wide[dia].map(lambda x: f"{x:.3f}")

    raw = summaries["no_audit_raw_cmn_dia_by_model_construct.csv"].copy()
    raw["model_name"] = raw["model"].map(MODEL_LABEL)
    raw["rating_ci"] = raw.apply(lambda r: f"{r['mean']:.3f} [{r['ci_low']:.3f}, {r['ci_high']:.3f}]", axis=1)

    speaker = summaries["no_audit_qwen35_speaker_bias.csv"].copy()
    top_speaker = speaker[speaker["construct"] == "Ability"].sort_values("mean").head(6).copy()
    top_speaker["bias_ci"] = top_speaker.apply(lambda r: f"{r['mean']:.3f} [{r['ci_low']:.3f}, {r['ci_high']:.3f}]", axis=1)

    sentence = summaries["no_audit_qwen35_sentence_bias.csv"].copy()
    top_sentence = sentence[sentence["construct"] == "Ability"].sort_values("mean").head(5).copy()
    top_sentence["bias_ci"] = top_sentence.apply(lambda r: f"{r['mean']:.3f} [{r['ci_low']:.3f}, {r['ci_high']:.3f}]", axis=1)

    fit_fmt = fit_table.copy()
    if not fit_fmt.empty:
        fit_fmt["converged_text"] = fit_fmt["converged"].map(lambda x: "yes" if bool(x) else "no")
        for col in ["aic", "bic"]:
            fit_fmt[col] = fit_fmt[col].map(lambda x: f"{x:.2f}")
    lrt_fmt = lrt.copy()
    if not lrt_fmt.empty:
        lrt_fmt["lr_chisq"] = lrt_fmt["lr_chisq"].map(lambda x: f"{x:.2f}")
        lrt_fmt["df"] = lrt_fmt["df"].map(lambda x: f"{x:.0f}")
        lrt_fmt["p"] = lrt_fmt["p"].map(lambda x: "<.001" if x < 0.001 else f"{x:.4f}")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    TABLE.mkdir(parents=True, exist_ok=True)
    ok = read_ok()
    wide, cu, raw_construct = build_pairs(ok)
    summaries = summarise(wide, cu, raw_construct)
    plot_core(summaries["no_audit_core_bias_by_model_construct_dialect.csv"])
    plot_raw(summaries["no_audit_raw_cmn_dia_by_model_construct.csv"])
    plot_speaker(summaries["no_audit_qwen35_speaker_bias.csv"])
    plot_sentence(summaries["no_audit_qwen35_sentence_bias.csv"])
    fit_table, lrt, fit_notes = fit_mixed_models(cu)
    write_report(summaries, fit_table, lrt, fit_notes)
    print(f"Wrote report: {OUT / 'M1_no_audit_report.md'}")


if __name__ == "__main__":
    main()
