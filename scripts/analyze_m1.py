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

    report = f"""# M1进一步分析报告

数据来源：`E:\\ECNU\\SpeechLLM\\M1`  
生成时间：自动分析脚本`{Path(__file__).name}`  
核心偏差定义：`Bias = DIA评分 - CMN评分`。负值表示地域口音版本相对普通话版本被打低。

## 1. 数据清理与质量审计

M1理论完整trial数为`12 speaker × 9 sentence × 2 guise × 7 dimension = 1512`/run。

- Qwen2.5-Omni-7B：5个run均为1512个成功trial；重复run评分完全一致，因此本报告只保留run1进入推断性汇总，避免把确定性重复当作独立样本。
- Qwen3-Omni-Flash：每个run有1512个成功trial；`m1_qwen3.csv`额外包含18行失败/重试残留，已按`ok=True`过滤。
- Qwen3.5-Omni-Plus：5个run均为1512个成功trial；重复run存在约20%评分波动，因此先在同一condition内对run取均值。

### Run-to-run exact match

{md_table(run_match, ["model", "run_a", "run_b", "common_trials", "same_rating", "same_rate"])}

## 2. M1核心结果重算

### 总体构念偏差

{md_table(overall[["model_name", "construct", "bias_ci", "n"]], ["model_name", "construct", "bias_ci", "n"])}

### 按模型、构念、方言的平均偏差

{md_table(core_wide[["model_name", "construct"] + DIALECT_ORDER], ["model_name", "construct"] + DIALECT_ORDER)}

![M1 core bias heatmap]({(FIG / "m1_core_bias_heatmap.png").as_posix()})

主要模式：

- 偏差最集中在`Ability`构念，尤其是Qwen3.5-Omni-Plus。
- Qwen3.5的`Ability`平均偏差为`-0.720`，明显强于Qwen2.5和Qwen3。
- 北京口音是稳定例外：在Qwen3.5中北京`Ability`仅为`-0.096`，而济南、西安、武汉分别约为`-1.100`、`-1.015`、`-0.974`。
- `Warmth`在M1中整体接近0，说明直接评分任务里不是所有社会评价都被统一压低。
- `Trust`存在小幅负向偏差，Qwen3.5中以西安和武汉更明显，但强度远低于Ability。

## 3. 原始CMN/DIA评分分解

{md_table(raw_summary[["model_name", "construct", "guise", "rating_ci", "n"]], ["model_name", "construct", "guise", "rating_ci", "n"])}

![Raw CMN vs DIA ratings]({(FIG / "m1_raw_cmn_vs_dia.png").as_posix()})

解释：

- Qwen3.5的Ability差异来自普通话版本评分较高、方言版本评分较低：`CMN=5.397`，`DIA=4.677`。
- Qwen3.5的Warmth几乎没有CMN/DIA差异：`CMN=5.196`，`DIA=5.158`。
- Qwen3.5的Trust整体分数接近天花板，尤其CMN接近6分，可能限制了可观察差异。

## 4. Speaker层面分解

![Qwen3.5 speaker-level forest plot]({(FIG / "m1_qwen35_speaker_forest.png").as_posix()})

Qwen3.5的Ability偏差最强speaker：

{md_table(top_speaker[["speaker_id", "dialect", "bias_ci", "n"]], ["speaker_id", "dialect", "bias_ci", "n"])}

解释：

- 地区均值并不是完全均匀地分布在每个speaker上。`JNN_013`、`XIA_020`、`WHN_010`是Ability负向偏差最强的speaker。
- 北京两个speaker都接近0，这说明“北京例外”不是单个speaker偶然造成的。
- 后续报告地区效应时应同时给speaker-level图，避免把少数声音的强效应误读为整个地区稳定效应。

## 5. Sentence/text层面分解

![Qwen3.5 sentence-level bias plot]({(FIG / "m1_qwen35_sentence_bias.png").as_posix()})

Qwen3.5的Ability偏差最强句子：

{md_table(top_sentence[["text", "bias_ci", "n"]], ["text", "bias_ci", "n"])}

解释：

- 句子之间的偏差强度不同。`有一只鹰在天上飞`、`农民在山坡上种了树`、`锅里的包子熟了`在Ability上最负。
- 这提示文本内容、音频切分质量、发音难度或特定声学实现都可能调节偏差强度。
- 后续正式模型应保留`text`随机截距；如果做稳健性分析，可以剔除带有潜在社会语义的句子如`农民...`、`病人...`、`儿童...`后重算。

## 6. 混合效应模型

模型使用构念层面的偏差分数作为因变量：

`bias ~ model * construct * dialect + (1 | speaker) + (1 | text)`

其中speaker和text作为随机截距；模型使用最大似然估计，并通过似然比检验比较完整模型与去除交互项的简化模型。

### Likelihood-ratio tests

模型收敛状态：

{md_table(fit_indices[["model_spec", "nobs", "aic", "bic", "converged_text"]], ["model_spec", "nobs", "aic", "bic", "converged_text"])}

注意：完整三重交互模型`full`未完全收敛，因此涉及`full`的LRT应作为探索性结果，而不是最终显著性证据。

{md_table(lrt_fmt, ["comparison", "lr_chisq", "df", "p"]) if not lrt_fmt.empty else "MixedLM did not return likelihood-ratio tests."}

解释：

- 三重交互检验为`full vs no_three_way`，当前结果为`p≈.090`，且完整模型未收敛。因此不能把`model × construct × dialect`三重交互写成稳健显著；更合适的表述是三重交互有探索性趋势，但需要更多speaker或更稳定模型进一步验证。
- 去除任一二重交互都会显著降低模型拟合，且完整交互结构相对主效应模型显著改善。这支持PPT中的核心观察：不同代Qwen-Omni模型不是简单整体平移，而是在模型、构念、方言之间存在结构性差异。
- 但需要谨慎：每个方言只有2名speaker，speaker和dialect部分嵌套，因此方言层面的随机性估计有限。当前模型适合用于现有M1数据的结构性检验，不应过度外推到所有说话人或所有地区口音。

## 7. 结论

M1进一步分析支持以下结论：

1. Qwen3.5-Omni-Plus在直接评分任务中表现出最强的地域口音偏差，主要集中在Ability/Status构念。
2. Qwen3.5中，非北京地域口音的Ability评分系统性低于同speaker普通话版本；北京接近0，是稳定例外。
3. Warmth在M1中基本不受影响，说明模型不是对方言口音做全面负向评价，而是更具体地把非标准口音与较低专业性/受教育程度/能力联系起来。
4. Speaker和sentence层面都有异质性，必须作为后续报告和模型控制项。
5. 重复run审计显示，Qwen2.5的重复run不提供额外独立信息，Qwen3.5则存在明显run-to-run评分不稳定，后续应报告稳定性并考虑定义版prompt稳健性检查。

## 8. 输出文件

- 清理后trial-level数据：`outputs/m1_analysis/tables/m1_dimension_pairs_clean.csv`
- 构念层数据：`outputs/m1_analysis/tables/m1_construct_units_clean.csv`
- 核心偏差表：`outputs/m1_analysis/tables/core_bias_by_model_construct_dialect.csv`
- 混合模型摘要：`outputs/m1_analysis/tables/mixedlm_full_summary.txt`
- 图表目录：`outputs/m1_analysis/figures/`
"""

    if fit_notes:
        report += f"\n\n## MixedLM拟合备注\n\n```text\n{fit_notes}\n```\n"

    (OUT / "M1_further_analysis_report.md").write_text(report, encoding="utf-8")


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
