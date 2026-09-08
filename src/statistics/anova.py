"""ANOVA 통계 검증 (SPEC §5, ADR-005, ADR-011).

- 검증 1: 피처 그룹별(traditional / alternative / combined) One-way ANOVA
- 검증 2: 알고리즘별(LR / XGBoost / LightGBM) One-way ANOVA

관측치는 `data/processed/fold_auc.csv` 의 fold AUC 다.
ADR-011 에 따라 각 검정에서 나머지 요인을 한 수준으로 고정해 교락을 제거한다.

실행:
    python -m src.statistics.anova
"""

from __future__ import annotations

import json
import logging
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd

from src.config import load_config, resolve_path

logger = logging.getLogger(__name__)


# --- 지표 계산 -------------------------------------------------------------

def eta_squared(groups: list[np.ndarray]) -> float:
    """η² = 집단간 제곱합 / 전체 제곱합."""
    values = np.concatenate(groups)
    grand_mean = values.mean()
    ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
    ss_total = ((values - grand_mean) ** 2).sum()
    return float(ss_between / ss_total) if ss_total > 0 else 0.0


def interpret_eta_squared(value: float, thresholds: dict[str, float]) -> str:
    """SPEC §5 의 효과크기 해석 기준."""
    if value >= thresholds["large"]:
        return "Large (strong)"
    if value >= thresholds["medium"]:
        return "Medium (moderate)"
    if value >= thresholds["small"]:
        return "Small (weak)"
    return "Negligible"


def normality_check(groups: dict[str, np.ndarray]) -> dict[str, Any]:
    """Shapiro-Wilk 정규성 검정. p < 0.05 이면 정규성 기각."""
    per_group = {
        name: {
            "statistic": float(stats.shapiro(values).statistic),
            "p_value": float(stats.shapiro(values).pvalue),
            "normal": bool(stats.shapiro(values).pvalue >= 0.05),
        }
        for name, values in groups.items()
    }
    return {
        "per_group": per_group,
        "all_normal": all(g["normal"] for g in per_group.values()),
    }


def homogeneity_check(groups: dict[str, np.ndarray]) -> dict[str, Any]:
    """Levene 등분산성 검정(중앙값 기준). p < 0.05 이면 등분산 기각."""
    statistic, p_value = stats.levene(*groups.values(), center="median")
    return {
        "statistic": float(statistic),
        "p_value": float(p_value),
        "equal_variance": bool(p_value >= 0.05),
    }


def tukey_hsd(df: pd.DataFrame, factor: str, alpha: float) -> list[dict[str, Any]]:
    """Tukey HSD 사후검정."""
    result = pairwise_tukeyhsd(df["auc"].to_numpy(), df[factor].to_numpy(), alpha=alpha)
    table = pd.DataFrame(result.summary().data[1:], columns=result.summary().data[0])
    return [
        {
            "comparison": f"{row['group1']} vs {row['group2']}",
            "mean_diff": float(row["meandiff"]),
            "p_adj": float(row["p-adj"]),
            "ci_lower": float(row["lower"]),
            "ci_upper": float(row["upper"]),
            "significant": bool(row["reject"]),
        }
        for _, row in table.iterrows()
    ]


# --- 검정 실행 -------------------------------------------------------------

def _diagnostics_advice(normality: dict[str, Any], homogeneity: dict[str, Any]) -> list[str]:
    """가정 위반 시 대안 검정을 제안한다."""
    advice: list[str] = []
    if not homogeneity["equal_variance"]:
        advice.append(
            "등분산 가정 위반(Levene p<0.05) — Welch ANOVA 결과를 함께 보고할 것. "
            "Welch 는 집단별 분산이 달라도 1종 오류율을 유지한다."
        )
    if not normality["all_normal"]:
        advice.append(
            "정규성 가정 위반(Shapiro-Wilk p<0.05) — 비모수 Kruskal-Wallis 결과를 함께 보고할 것. "
            "다만 n=20 이고 ANOVA 는 정규성 위반에 비교적 강건하므로 결론이 뒤집힐 가능성은 낮다."
        )
    if not advice:
        advice.append("정규성·등분산성 가정 모두 충족. 표준 One-way ANOVA 결과를 그대로 사용한다.")
    return advice


def _effect_size_note(eta: float, extreme: float) -> str | None:
    if eta < extreme:
        return None
    return (
        f"η²={eta:.4f} 로 극단적으로 크다(>{extreme}). 집단 간 분산이 전체 분산의 대부분을 설명한다는 뜻이며, "
        "집단 평균이 뚜렷이 분리되고 집단 내 fold 변동이 매우 작기 때문이다. "
        "이는 실험 설계상 각 집단이 서로 다른 피처 구성/알고리즘으로 고정돼 있고 "
        "동일 데이터에 반복 CV 를 적용해 fold 간 변동이 작게 나온 결과다. "
        "실무적 우월성의 크기를 나타내는 값으로 읽어서는 안 되며, "
        "집단 평균 차이(mean_diff)의 절대 크기로 판단할 것."
    )


def run_test(
    folds: pd.DataFrame, spec: dict[str, Any], config: dict[str, Any], scope: str = "controlled"
) -> dict[str, Any]:
    """One-way ANOVA 한 건을 수행한다.

    scope="controlled" : ADR-011 대로 나머지 요인을 고정
    scope="pooled"     : 전체 관측치 사용(교락 있음, 참고치)
    """
    acfg = config["anova"]
    alpha = acfg["alpha"]
    factor = spec["factor"]

    data = folds
    control = spec.get("control", {}) if scope == "controlled" else {}
    for column, value in control.items():
        data = data[data[column] == value]

    grouped = {name: g["auc"].to_numpy() for name, g in data.groupby(factor)}
    if len(grouped) < 2:
        raise ValueError(f"{spec['name']}: 비교할 집단이 부족합니다 ({list(grouped)}).")

    values = list(grouped.values())
    f_stat, p_value = stats.f_oneway(*values)
    eta = eta_squared(values)
    normality = normality_check(grouped)
    homogeneity = homogeneity_check(grouped)

    result: dict[str, Any] = {
        "test_name": spec["name"],
        "scope": scope,
        "factor": factor,
        "control": control,
        "groups": list(grouped),
        "group_n": {k: int(len(v)) for k, v in grouped.items()},
        "group_means": {k: float(v.mean()) for k, v in grouped.items()},
        "group_stds": {k: float(v.std(ddof=1)) for k, v in grouped.items()},
        "f_statistic": float(f_stat),
        "p_value": float(p_value),
        "effect_size_eta_squared": eta,
        "effect_size_interpretation": interpret_eta_squared(eta, acfg["eta_squared_thresholds"]),
        "is_significant": bool(p_value < alpha),
        "alpha": alpha,
        "diagnostics": {
            "normality_shapiro_wilk": normality,
            "homogeneity_levene": homogeneity,
            "advice": _diagnostics_advice(normality, homogeneity),
        },
    }

    # 가정 위반 시 대안 검정을 함께 싣는다.
    if not homogeneity["equal_variance"]:
        welch = stats.alexandergovern(*values)
        result["welch_anova"] = {
            "statistic": float(welch.statistic), "p_value": float(welch.pvalue),
        }
    if not normality["all_normal"]:
        kruskal = stats.kruskal(*values)
        result["kruskal_wallis"] = {
            "statistic": float(kruskal.statistic), "p_value": float(kruskal.pvalue),
        }

    if result["is_significant"]:
        result["post_hoc_tukey"] = tukey_hsd(data, factor, alpha)

    note = _effect_size_note(eta, acfg["eta_squared_extreme"])
    if note:
        result["effect_size_note"] = note

    return result


def load_fold_auc(config: dict[str, Any] | None = None) -> pd.DataFrame:
    cfg = config if config is not None else load_config()
    path = resolve_path(cfg["preprocess"]["output_dir"]) / cfg["anova"]["fold_auc_file"]
    if not path.exists():
        raise FileNotFoundError(
            f"{path} 가 없습니다. 먼저 `python -m src.models.train` 을 실행하세요."
        )
    return pd.read_csv(path)


def run_all(config: dict[str, Any] | None = None, save: bool = True) -> dict[str, Any]:
    """SPEC §5 의 두 검정을 수행하고 결과를 반환한다."""
    cfg = config if config is not None else load_config()
    folds = load_fold_auc(cfg)

    results: dict[str, Any] = {}
    for key, spec in cfg["anova"]["tests"].items():
        results[key] = run_test(folds, spec, cfg, scope="controlled")
        results[key]["pooled_reference"] = run_test(folds, spec, cfg, scope="pooled")

    if save:
        path = resolve_path(cfg["preprocess"]["output_dir"]) / cfg["anova"]["result_file"]
        path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("ANOVA 결과 저장: %s", path)
    return results


def format_report(results: dict[str, Any]) -> str:
    """콘솔·리포트용 텍스트."""
    lines: list[str] = []
    for key, res in results.items():
        lines.append(f"\n{'=' * 78}")
        lines.append(f"[{key}] {res['test_name']}")
        control = ", ".join(f"{k}={v}" for k, v in res["control"].items()) or "없음"
        lines.append(f"통제(ADR-011): {control}")
        lines.append(f"{'-' * 78}")
        lines.append(f"{'집단':<20} {'n':>5} {'평균 AUC':>12} {'표준편차':>12}")
        for group in res["groups"]:
            lines.append(f"{group:<20} {res['group_n'][group]:>5} "
                         f"{res['group_means'][group]:>12.4f} {res['group_stds'][group]:>12.4f}")
        lines.append("")
        lines.append(f"  F 통계량 = {res['f_statistic']:.4f}")
        lines.append(f"  p-value  = {res['p_value']:.6g}  (alpha={res['alpha']}, "
                     f"{'유의함' if res['is_significant'] else '유의하지 않음'})")
        lines.append(f"  η²       = {res['effect_size_eta_squared']:.4f}  "
                     f"[{res['effect_size_interpretation']}]")

        diag = res["diagnostics"]
        lines.append("\n  진단:")
        for name, item in diag["normality_shapiro_wilk"]["per_group"].items():
            mark = "정규성 충족" if item["normal"] else "정규성 위반"
            lines.append(f"    Shapiro-Wilk {name:<18} W={item['statistic']:.4f} "
                         f"p={item['p_value']:.4g}  {mark}")
        lev = diag["homogeneity_levene"]
        lines.append(f"    Levene (등분산)          stat={lev['statistic']:.4f} "
                     f"p={lev['p_value']:.4g}  "
                     f"{'등분산 충족' if lev['equal_variance'] else '등분산 위반'}")
        if "welch_anova" in res:
            lines.append(f"    Welch ANOVA              stat={res['welch_anova']['statistic']:.4f} "
                         f"p={res['welch_anova']['p_value']:.6g}")
        if "kruskal_wallis" in res:
            lines.append(f"    Kruskal-Wallis           H={res['kruskal_wallis']['statistic']:.4f} "
                         f"p={res['kruskal_wallis']['p_value']:.6g}")
        for advice in diag["advice"]:
            lines.append(f"    → {advice}")

        if "post_hoc_tukey" in res:
            lines.append("\n  사후검정 (Tukey HSD):")
            lines.append(f"    {'비교':<44} {'mean_diff':>11} {'p_adj':>10}  유의")
            for row in res["post_hoc_tukey"]:
                lines.append(f"    {row['comparison']:<44} {row['mean_diff']:>+11.4f} "
                             f"{row['p_adj']:>10.4g}  {'예' if row['significant'] else '아니오'}")

        if "effect_size_note" in res:
            lines.append(f"\n  [효과크기 해석 주의] {res['effect_size_note']}")

        pooled = res.get("pooled_reference")
        if pooled:
            lines.append(f"\n  참고(교락 있음, 전체 {sum(pooled['group_n'].values())}건): "
                         f"F={pooled['f_statistic']:.4f}, p={pooled['p_value']:.6g}, "
                         f"η²={pooled['effect_size_eta_squared']:.4f}")
    return "\n".join(lines)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    print(format_report(run_all()))


if __name__ == "__main__":
    main()
