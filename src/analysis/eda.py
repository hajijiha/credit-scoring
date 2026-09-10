"""EDA 그림 생성 (리포트 3장 · 4장용).

라벨은 영문으로 둔다. 이 환경의 한글 폰트는 Windows 마운트에만 있어
컨테이너에서 글리프가 깨진다(`src/models/report_assets.py` 와 같은 이유).

실행:
    python -m src.analysis.eda
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import norm

from src.config import load_config, resolve_path
from src.data.loader import load_gmsc
from src.data.preprocessor import Preprocessor
from src.data.simulator import (
    _FEATURE_KIND,
    _fit_feature,
    _nuisance_signal,
    _standardize,
    _transform,
    simulate,
)

logger = logging.getLogger(__name__)

SHORT_NAME = {
    "telecom_payment_rate": "Telecom rate",
    "utility_payment_rate": "Utility rate",
    "spending_consistency": "Spending consist.",
    "regular_payment_count": "Regular payments",
    "app_login_frequency": "App logins",
    "RevolvingUtilizationOfUnsecuredLines": "Revolving util.",
    "age": "Age",
    "NumberOfTime30-59DaysPastDueNotWorse": "Past due 30-59",
    "DebtRatio": "Debt ratio",
    "MonthlyIncome": "Monthly income",
    "NumberOfOpenCreditLinesAndLoans": "Open credit lines",
    "NumberOfTimes90DaysLate": "90+ days late",
    "NumberRealEstateLoansOrLines": "Real estate loans",
    "NumberOfTime60-89DaysPastDueNotWorse": "Past due 60-89",
    "NumberOfDependents": "Dependents",
}
TYPE_ORDER = ["none", "missing_only", "history_only", "both"]
TYPE_LABEL = {
    "none": "Non thin-filer", "missing_only": "Thin: missing only",
    "history_only": "Thin: history only", "both": "Thin: both",
}


def _style() -> None:
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams["figure.dpi"] = 120
    plt.rcParams["savefig.bbox"] = "tight"
    plt.rcParams["axes.unicode_minus"] = False


def _out_dir(config: dict[str, Any]) -> Path:
    path = resolve_path(config["explainer"]["report_assets_dir"])
    path.mkdir(parents=True, exist_ok=True)
    return path


# --- v1 재구성 -------------------------------------------------------------

def rebuild_v1_alternative(df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """ADR-001 **v1** 구조로 대안 5변수를 다시 만든다 (비교용).

    v1: `m_i = standardize(0.6·nuisance + 0.8·eps_i)` — nuisance 만 공유되고
    노이즈는 변수마다 독립이었다. 목표 상관 범위도 SPEC 요건 그대로 0.30~0.50 을 썼다.
    현재 코드(v2)는 공유 잠재요인을 별도로 두므로, 여기서는 v1 공식을 직접 재현한다.
    """
    sim = config["simulator"]
    features = list(sim["alternative_features"])
    lo = sim["target_correlation"]["min"]
    hi = sim["target_correlation"]["max"]

    target = df[config["data"]["target_column"]].astype("float64").to_numpy()
    t_z = _standardize(target)
    nuisance = _nuisance_signal(df)
    rng = np.random.default_rng(sim["random_state"])

    margin = 0.02 * (hi - lo)
    ladder = sorted(np.linspace(lo + margin, hi - margin, len(features)), reverse=True)
    biased = [f for f in features if f in sim["biased_features"]]
    ordered = biased + [f for f in features if f not in biased]
    goal_of = dict(zip(ordered, ladder))

    out = pd.DataFrame(index=df.index)
    for feature in features:
        eps = rng.normal(size=len(df))
        # v1 의 비타겟 성분: nuisance(공유) 0.6 + 개별 노이즈 0.8
        m = _standardize(0.6 * nuisance + 0.8 * eps)
        m = m - t_z * (float(m @ t_z) / float(t_z @ t_z))
        m = _standardize(m)
        rho = -goal_of[feature]
        values = _transform(rho * t_z + np.sqrt(max(1 - rho**2, 0.0)) * m,
                            _FEATURE_KIND[feature])
        # 변환 후 상관을 목표에 맞춘다(v2 와 같은 고정점 보정).
        for _ in range(30):
            achieved = abs(float(np.corrcoef(values, target)[0, 1]))
            if abs(achieved - goal_of[feature]) <= 1e-3:
                break
            rho = float(np.clip(rho * goal_of[feature] / max(achieved, 1e-6), -0.99, -0.01))
            m2 = _standardize(0.6 * nuisance + 0.8 * eps)
            m2 = m2 - t_z * (float(m2 @ t_z) / float(t_z @ t_z))
            m2 = _standardize(m2)
            values = _transform(rho * t_z + np.sqrt(max(1 - rho**2, 0.0)) * m2,
                                _FEATURE_KIND[feature])
        out[feature] = values
    return out


def _within_class_corr(frame: pd.DataFrame, features: list[str], y: np.ndarray, cls: int):
    block = frame.loc[y == cls, features]
    return block.corr()


def _joint_effect_size(frame: pd.DataFrame, features: list[str], y: np.ndarray) -> float:
    """**조건부 독립을 가정한** 결합 Cohen's d = sqrt(Σ d_i²).

    피처들이 타겟 조건부로 상관되어 있으면 이 값은 실제보다 크게 나온다.
    v1 과 v2 의 차이가 바로 그 지점이므로, 실제 값은 `_lda_effect_size` 로 따로 구한다.
    """
    total = 0.0
    for feature in features:
        a, b = frame.loc[y == 0, feature], frame.loc[y == 1, feature]
        pooled = np.sqrt(((len(a) - 1) * a.var() + (len(b) - 1) * b.var())
                         / (len(a) + len(b) - 2))
        total += (abs(a.mean() - b.mean()) / pooled) ** 2
    return float(np.sqrt(total))


def _lda_effect_size(frame: pd.DataFrame, features: list[str], y: np.ndarray) -> float:
    """피처 간 상관을 반영한 실제 결합 효과크기 (Mahalanobis 거리)."""
    x = frame[features].to_numpy()
    m0, m1 = x[y == 0].mean(0), x[y == 1].mean(0)
    cov = np.cov(x[y == 0].T)
    diff = m1 - m0
    return float(np.sqrt(diff @ np.linalg.solve(cov, diff)))


def figure_within_class_correlation(
    v1: pd.DataFrame, v2: pd.DataFrame, features: list[str], y: np.ndarray, out: Path
) -> Path:
    """4장 핵심: 타겟을 조건으로 한 대안 피처 간 상관 (v1 vs v2)."""
    labels = [SHORT_NAME[f] for f in features]
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    summary = {}
    for ax, (name, frame) in zip(axes, (("v1 — independent noise", v1),
                                        ("v2 — shared latent factor", v2))):
        corr = _within_class_corr(frame, features, y, 0)
        off = corr.to_numpy()[np.triu_indices(len(features), 1)]
        d_indep = _joint_effect_size(frame, features, y)
        d_actual = _lda_effect_size(frame, features, y)
        summary[name] = (off.mean(), d_indep, d_actual)
        sns.heatmap(corr, annot=True, fmt=".2f", vmin=0, vmax=1, cmap="Blues",
                    xticklabels=labels, yticklabels=labels, cbar=False, ax=ax,
                    annot_kws={"size": 11})
        ax.set_title(
            f"{name}\nmean off-diagonal {off.mean():.3f}\n"
            f"joint d: {d_indep:.2f} if independent  vs  {d_actual:.2f} actual\n"
            f"implied AUC: {norm.cdf(d_indep / np.sqrt(2)):.4f}  vs  "
            f"{norm.cdf(d_actual / np.sqrt(2)):.4f}",
            fontsize=11.5,
        )
        ax.tick_params(axis="x", labelrotation=35, labelsize=10)
        ax.tick_params(axis="y", labelsize=10)
        for label in ax.get_xticklabels():
            label.set_ha("right")
    fig.suptitle(
        "Correlation among alternative features within the non-default class\n"
        "v1: features are near-independent given the target, so information adds up",
        y=1.10, fontsize=13,
    )
    path = out / "within_class_correlation_v1_v2.png"
    fig.savefig(path)
    plt.close(fig)
    for name, (mean_corr, d_indep, d_actual) in summary.items():
        logger.info(
            "  %-28s 클래스내부 평균 상관 %.4f | 결합 d 독립가정 %.3f (AUC %.4f) "
            "/ 실제 %.3f (AUC %.4f)",
            name, mean_corr, d_indep, norm.cdf(d_indep / np.sqrt(2)),
            d_actual, norm.cdf(d_actual / np.sqrt(2)),
        )
    return path


# --- EDA 그림 --------------------------------------------------------------

def figure_target_distribution(df: pd.DataFrame, config: dict[str, Any], out: Path) -> Path:
    target = config["data"]["target_column"]
    counts = df[target].value_counts().sort_index()
    rate = df[target].mean()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    colors = ["#4C72B0", "#C44E52"]
    axes[0].bar(["Non-default (0)", "Default (1)"], counts.to_numpy(), color=colors)
    for i, value in enumerate(counts.to_numpy()):
        axes[0].text(i, value * 1.01, f"{value:,}", ha="center", fontsize=12)
    axes[0].set_ylabel("Rows")
    axes[0].set_title(f"Class counts (n = {len(df):,})")

    axes[1].barh(["Default rate"], [rate], color=colors[1])
    axes[1].barh(["Default rate"], [1 - rate], left=[rate], color=colors[0])
    axes[1].text(rate / 2, 0, f"{rate:.2%}", va="center", ha="center",
                 color="white", fontsize=13, fontweight="bold")
    axes[1].set_xlim(0, 1)
    axes[1].set_xlabel("Share")
    axes[1].set_title("Severe class imbalance")
    axes[1].grid(False)
    fig.suptitle("Target distribution — GMSC after ADR-007 age filter", y=1.02)
    path = out / "eda_target_distribution.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def figure_missing_pattern(raw: pd.DataFrame, simulated: pd.DataFrame,
                           config: dict[str, Any], out: Path) -> Path:
    """원본 결측(소득·부양가족)과 시뮬레이터가 만든 결측(연체 3컬럼)을 나눠 보여준다."""
    sim = config["simulator"]
    masked = sim["thin_filer_masked_columns"]

    natural = (raw.isna().mean() * 100).sort_values(ascending=False)
    natural = natural[natural > 0]
    induced = simulated[masked].isna().mean() * 100

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    axes[0].barh([SHORT_NAME.get(i, i) for i in natural.index][::-1],
                 natural.to_numpy()[::-1], color="#4C72B0")
    for i, value in enumerate(natural.to_numpy()[::-1]):
        axes[0].text(value + 0.4, i, f"{value:.2f}%", va="center", fontsize=11)
    axes[0].set_xlim(0, max(natural.max() * 1.25, 5))
    axes[0].set_xlabel("Missing rate (%)")
    axes[0].set_title("Original GMSC missingness")

    axes[1].barh([SHORT_NAME.get(i, i) for i in induced.index][::-1],
                 induced.to_numpy()[::-1], color="#C44E52")
    for i, value in enumerate(induced.to_numpy()[::-1]):
        axes[1].text(value + 0.4, i, f"{value:.2f}%", va="center", fontsize=11)
    axes[1].set_xlim(0, max(induced.max() * 1.3, 5))
    axes[1].set_xlabel("Missing rate (%)")
    axes[1].set_title("Simulator-induced missingness (thin filers, ADR-001/010)")
    fig.suptitle("Missing data: natural vs induced", y=1.02)
    path = out / "eda_missing_pattern.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def figure_feature_distributions(simulated: pd.DataFrame, config: dict[str, Any],
                                 out: Path) -> Path:
    traditional = ["age", "MonthlyIncome", "DebtRatio", "RevolvingUtilizationOfUnsecuredLines"]
    alternative = list(config["features"]["alternative"])
    target = config["data"]["target_column"]

    fig, axes = plt.subplots(2, 5, figsize=(23, 9))
    for ax, feature in zip(axes[0], traditional + ["NumberOfOpenCreditLinesAndLoans"]):
        series = simulated[feature].dropna()
        clipped = series.clip(upper=series.quantile(0.99))
        ax.hist(clipped, bins=40, color="#4C72B0", edgecolor="white", linewidth=0.4)
        ax.set_title(f"{SHORT_NAME[feature]}\n(clipped at p99)", fontsize=11)
        ax.set_yticks([])
    for ax, feature in zip(axes[1], alternative):
        for cls, color, label in ((0, "#4C72B0", "Non-default"), (1, "#C44E52", "Default")):
            ax.hist(simulated.loc[simulated[target] == cls, feature], bins=40, density=True,
                    alpha=0.6, color=color, label=label)
        corr = abs(float(np.corrcoef(simulated[feature], simulated[target])[0, 1]))
        ax.set_title(f"{SHORT_NAME[feature]}\n|corr| with target = {corr:.3f}", fontsize=11)
        ax.set_yticks([])
    axes[1][0].legend(fontsize=9)
    axes[0][0].set_ylabel("Traditional", fontsize=13)
    axes[1][0].set_ylabel("Alternative", fontsize=13)
    fig.suptitle("Feature distributions — traditional (top) vs alternative (bottom, split by class)",
                 y=1.01)
    path = out / "eda_feature_distributions.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def _profile_table(frame: pd.DataFrame, group: pd.Series, features: list[str]) -> pd.DataFrame:
    """집단별 표준화 평균(전체 평균·표준편차 기준)."""
    standardized = frame[features].apply(
        lambda col: (col - col.mean()) / (col.std(ddof=0) or 1.0)
    )
    return standardized.groupby(group).mean()


def figure_thin_filer_profile(simulated: pd.DataFrame, config: dict[str, Any],
                              out: Path) -> Path:
    sim = config["simulator"]
    flag = sim["thin_filer_flag_column"]
    features = ["RevolvingUtilizationOfUnsecuredLines", "NumberOfOpenCreditLinesAndLoans",
                "MonthlyIncome", "age", "credit_history_months"] + \
               list(config["features"]["alternative"])
    labels = [SHORT_NAME.get(f, "Credit history (mo)") for f in features]

    group = simulated[flag].map({1: "Thin filer", 0: "Non thin-filer"})
    profile = _profile_table(simulated, group, features)

    fig, ax = plt.subplots(figsize=(13, 6))
    width = 0.38
    positions = np.arange(len(features))
    for offset, name, color in ((-width / 2, "Non thin-filer", "#4C72B0"),
                                (width / 2, "Thin filer", "#C44E52")):
        ax.bar(positions + offset, profile.loc[name, features].to_numpy(),
               width=width, label=name, color=color)
    ax.axhline(0, color="#333333", lw=1)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=10)
    ax.set_ylabel("Standardized mean (z)")
    ax.set_title("Thin filer vs non thin-filer profile "
                 f"(thin = {int(simulated[flag].sum()):,} rows, "
                 f"{simulated[flag].mean():.1%})")
    ax.legend(fontsize=10)
    path = out / "eda_thin_filer_profile.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def figure_thin_filer_types(simulated: pd.DataFrame, config: dict[str, Any],
                            out: Path) -> Path:
    """유형 구성 + 정의 지표(이력 개월수·연체 관측률) + 나머지 피처 프로파일."""
    sim = config["simulator"]
    type_col = sim["thin_filer_type_column"]
    hist_col = sim["credit_history"]["column"]
    masked = sim["thin_filer_masked_columns"]
    colors = ["#8C8C8C", "#4C72B0", "#DD8452", "#C44E52"]

    counts = simulated[type_col].value_counts().reindex(TYPE_ORDER).fillna(0)
    months = simulated.groupby(type_col)[hist_col].mean().reindex(TYPE_ORDER)
    observed = (1 - simulated.groupby(type_col)[masked[0]].apply(
        lambda col: col.isna().mean())).reindex(TYPE_ORDER) * 100

    # 정의에 쓰이지 않는 피처만 프로파일에 넣는다(이력 개월수는 tautological 이라 제외).
    features = ["RevolvingUtilizationOfUnsecuredLines", "NumberOfOpenCreditLinesAndLoans",
                "MonthlyIncome", "telecom_payment_rate"]
    labels = [SHORT_NAME[f] for f in features]
    profile = _profile_table(simulated, simulated[type_col], features).reindex(TYPE_ORDER)

    fig, axes = plt.subplots(1, 3, figsize=(21, 6.4), layout="constrained")

    axes[0].bar([TYPE_LABEL[t] for t in TYPE_ORDER], counts.to_numpy(), color=colors)
    for i, value in enumerate(counts.to_numpy()):
        axes[0].text(i, value * 1.01, f"{int(value):,}", ha="center", fontsize=11)
    axes[0].set_ylabel("Rows")
    axes[0].set_title("Composition (ADR-010)")

    # 정의 지표. 이중축을 쓰면 옆 패널 라벨과 겹치므로, 연체 관측률은 막대 위 주석으로 쓴다.
    positions = np.arange(len(TYPE_ORDER))
    axes[1].bar(positions, months.to_numpy(), color=colors, width=0.6)
    for i, (value, rate) in enumerate(zip(months.to_numpy(), observed.to_numpy())):
        axes[1].text(i, value + 2, f"{value:.0f} mo", ha="center", fontsize=11)
        note = f"delinquency\nobserved {rate:.0f}%"
        if value > months.max() * 0.4:          # 막대가 충분히 높으면 안쪽에
            axes[1].text(i, value / 2, note, ha="center", va="center",
                         fontsize=9.5, color="white", fontweight="bold")
        else:                                    # 짧은 막대는 바깥에
            axes[1].text(i, value + 9, note, ha="center", va="bottom",
                         fontsize=9.5, color="#333333")
    axes[1].set_ylim(0, months.max() * 1.32)
    axes[1].set_xticks(positions)
    axes[1].set_xticklabels([TYPE_LABEL[t] for t in TYPE_ORDER])
    axes[1].set_ylabel("Credit history (months)")
    axes[1].set_title("Defining criteria (SPEC §4 OR condition)")

    width = 0.2
    positions = np.arange(len(features))
    for i, (kind, color) in enumerate(zip(TYPE_ORDER, colors)):
        axes[2].bar(positions + (i - 1.5) * width, profile.loc[kind, features].to_numpy(),
                    width=width, label=TYPE_LABEL[kind], color=color)
    axes[2].axhline(0, color="#333333", lw=1)
    axes[2].set_xticks(positions)
    axes[2].set_xticklabels(labels)
    axes[2].set_ylabel("Standardized mean (z)")
    span = float(np.abs(profile[features].to_numpy()).max())
    axes[2].set_title("Other features (definition-independent)\n"
                      f"all differences within ±{span:.3f} z — types differ only "
                      "in the defining criteria", fontsize=11)
    axes[2].legend(fontsize=9)

    for ax in (axes[0], axes[1]):
        ax.tick_params(axis="x", labelrotation=20, labelsize=9)
        for label in ax.get_xticklabels():
            label.set_ha("right")
    axes[2].tick_params(axis="x", labelrotation=20, labelsize=9)
    for label in axes[2].get_xticklabels():
        label.set_ha("right")

    fig.suptitle("Thin filer composition and profile")
    path = out / "eda_thin_filer_types.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def figure_demographics(simulated: pd.DataFrame, config: dict[str, Any], out: Path) -> Path:
    sim = config["simulator"]
    gender_col, age_col = sim["gender_column"], sim["age_group_column"]
    age_order = config["fairness"]["groups"]["age_group"]
    target = config["data"]["target_column"]

    fig, axes = plt.subplots(1, 3, figsize=(19, 5.5))
    counts = simulated[gender_col].value_counts()
    axes[0].bar(counts.index, counts.to_numpy(), color=["#C44E52", "#4C72B0"])
    for i, value in enumerate(counts.to_numpy()):
        axes[0].text(i, value * 1.01, f"{value:,}", ha="center", fontsize=11)
    axes[0].set_title("Gender (synthetic 50:50, ADR-002)")
    axes[0].set_ylabel("Rows")

    age_counts = simulated[age_col].value_counts().reindex(age_order)
    axes[1].bar(age_order, age_counts.to_numpy(), color="#55A868")
    for i, value in enumerate(age_counts.to_numpy()):
        axes[1].text(i, value * 1.01, f"{value:,}", ha="center", fontsize=11)
    axes[1].set_title("Age group (derived from actual age)")

    rates = simulated.groupby([age_col, gender_col])[target].mean().unstack()
    rates = rates.reindex(age_order)
    width = 0.38
    positions = np.arange(len(age_order))
    for offset, column, color in ((-width / 2, "female", "#C44E52"),
                                 (width / 2, "male", "#4C72B0")):
        axes[2].bar(positions + offset, rates[column].to_numpy() * 100,
                    width=width, label=column, color=color)
    axes[2].set_xticks(positions)
    axes[2].set_xticklabels(age_order)
    axes[2].set_ylabel("Default rate (%)")
    axes[2].set_title("Actual default rate by group\n(target untouched by bias injection)")
    axes[2].legend(fontsize=10)
    fig.suptitle("Protected attribute distribution", y=1.03)
    path = out / "eda_demographics.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def figure_correlation_matrix(simulated: pd.DataFrame, config: dict[str, Any],
                              out: Path) -> Path:
    traditional = list(config["features"]["traditional"])
    alternative = list(config["features"]["alternative"])
    target = config["data"]["target_column"]
    columns = traditional + alternative + [target]
    labels = [SHORT_NAME.get(c, "Default") for c in columns]

    corr = simulated[columns].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    fig, ax = plt.subplots(figsize=(13, 10.5))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                vmin=-1, vmax=1, xticklabels=labels, yticklabels=labels,
                annot_kws={"size": 8}, cbar_kws={"shrink": 0.7}, ax=ax)
    ax.axhline(len(traditional), color="black", lw=2)
    ax.axvline(len(traditional), color="black", lw=2)
    ax.set_title("Correlation matrix — traditional (upper-left) vs alternative (lower-right)")
    ax.tick_params(labelsize=9)
    path = out / "eda_correlation_matrix.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = load_config()
    _style()
    out = _out_dir(cfg)

    raw = load_gmsc(cfg)
    simulated = simulate(raw, cfg)
    target = cfg["data"]["target_column"]
    y = simulated[target].to_numpy()
    features = list(cfg["features"]["alternative"])

    logger.info("=== v1 재구성 vs v2 (클래스 내부 상관) ===")
    v1 = rebuild_v1_alternative(raw, cfg)
    v1[target] = simulated[target].to_numpy()
    v2 = simulated[features + [target]].copy()
    logger.info("  v1 개별 |corr|: %s",
                {f: round(abs(float(np.corrcoef(v1[f], y)[0, 1])), 4) for f in features})
    logger.info("  v2 개별 |corr|: %s",
                {f: round(abs(float(np.corrcoef(v2[f], y)[0, 1])), 4) for f in features})

    made = [
        figure_within_class_correlation(v1, v2, features, y, out),
        figure_target_distribution(simulated, cfg, out),
        figure_missing_pattern(raw, simulated, cfg, out),
        figure_feature_distributions(simulated, cfg, out),
        figure_thin_filer_profile(simulated, cfg, out),
        figure_thin_filer_types(simulated, cfg, out),
        figure_demographics(simulated, cfg, out),
        figure_correlation_matrix(simulated, cfg, out),
    ]
    logger.info("생성 %d개", len(made))
    for path in made:
        logger.info("  %s", path.relative_to(resolve_path(".")))


if __name__ == "__main__":
    main()
