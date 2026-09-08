"""공정성 지표 (SPEC §4).

- Disparate Impact Ratio: 보호집단 승인율 / 비보호집단 승인율
- Equalized Odds: 집단 간 TPR 차이, FPR 차이

용어
----
- 유리한 결정(favorable outcome) = **승인**. `p_default < approval_threshold`.
- 라벨 y = 1 은 **부도**. TPR/FPR 은 "부도로 예측" 기준으로 계산한다.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from src.config import load_config

logger = logging.getLogger(__name__)


def approval_mask(proba: np.ndarray, threshold: float) -> np.ndarray:
    """승인 여부. 부도 확률이 임계값 미만이면 승인."""
    return np.asarray(proba) < threshold


def group_rates(
    proba: np.ndarray, sensitive: pd.Series, threshold: float
) -> pd.Series:
    """집단별 승인율."""
    approved = approval_mask(proba, threshold)
    return pd.Series(approved, index=sensitive.index).groupby(sensitive).mean()


def disparate_impact(
    proba: np.ndarray,
    sensitive: pd.Series,
    threshold: float,
    disadvantaged: str,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """DI Ratio 를 계산하고 기준 위반 시 WARNING 을 남긴다.

    `di_ratio`         : 불이익 집단 승인율 / 나머지 집단(합산) 승인율
    `di_ratio_min_max` : 최저 승인율 집단 / 최고 승인율 집단 (관행적 정의)
    """
    cfg = config if config is not None else load_config()
    lo = cfg["thresholds"]["di_ratio_min"]
    hi = cfg["thresholds"]["di_ratio_max"]

    approved = pd.Series(approval_mask(proba, threshold), index=sensitive.index)
    rates = approved.groupby(sensitive).mean()
    counts = sensitive.value_counts()

    is_disadvantaged = sensitive == disadvantaged
    protected_rate = float(approved[is_disadvantaged].mean())
    reference_rate = float(approved[~is_disadvantaged].mean())
    ratio = protected_rate / reference_rate if reference_rate > 0 else float("nan")
    min_max = float(rates.min() / rates.max()) if rates.max() > 0 else float("nan")

    attribute = sensitive.name or "sensitive"
    if not (lo <= ratio <= hi):
        logger.warning(
            "[공정성 위반] %s DI Ratio %.4f 가 기준 범위 [%.2f, %.2f] 밖입니다 "
            "(%s 승인율 %.4f vs 나머지 %.4f)",
            attribute, ratio, lo, hi, disadvantaged, protected_rate, reference_rate,
        )
    else:
        logger.info("%s DI Ratio %.4f (기준 %.2f~%.2f 충족)", attribute, ratio, lo, hi)

    return {
        "attribute": attribute,
        "disadvantaged_group": disadvantaged,
        "approval_rates": {k: float(v) for k, v in rates.items()},
        "group_sizes": {k: int(v) for k, v in counts.items()},
        "protected_approval_rate": protected_rate,
        "reference_approval_rate": reference_rate,
        "di_ratio": ratio,
        "di_ratio_min_max": min_max,
        "threshold_min": lo,
        "threshold_max": hi,
        "passes": bool(lo <= ratio <= hi),
    }


def _rates_by_group(
    y_true: np.ndarray, y_pred: np.ndarray, sensitive: pd.Series
) -> tuple[pd.Series, pd.Series]:
    """집단별 TPR, FPR. y=1 은 부도, y_pred=1 은 '부도로 예측'."""
    frame = pd.DataFrame({"y": np.asarray(y_true), "p": np.asarray(y_pred),
                          "g": sensitive.to_numpy()})
    tpr, fpr = {}, {}
    for group, block in frame.groupby("g"):
        positives = block[block["y"] == 1]
        negatives = block[block["y"] == 0]
        tpr[group] = float(positives["p"].mean()) if len(positives) else float("nan")
        fpr[group] = float(negatives["p"].mean()) if len(negatives) else float("nan")
    return pd.Series(tpr), pd.Series(fpr)


def equalized_odds(
    y_true: np.ndarray,
    proba: np.ndarray,
    sensitive: pd.Series,
    threshold: float,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Equalized Odds: 집단 간 TPR 차이와 FPR 차이 (최대-최소)."""
    cfg = config if config is not None else load_config()
    limit = cfg["thresholds"]["eo_diff_max"]

    y_pred = (~approval_mask(proba, threshold)).astype(int)   # 부도로 예측 = 거절
    tpr, fpr = _rates_by_group(y_true, y_pred, sensitive)
    tpr_diff = float(np.nanmax(tpr) - np.nanmin(tpr))
    fpr_diff = float(np.nanmax(fpr) - np.nanmin(fpr))

    attribute = sensitive.name or "sensitive"
    for name, value in (("TPR", tpr_diff), ("FPR", fpr_diff)):
        if value > limit:
            logger.warning(
                "[공정성 위반] %s Equalized Odds %s 차이 %.4f 가 기준 %.2f 를 초과합니다",
                attribute, name, value, limit,
            )
    if tpr_diff <= limit and fpr_diff <= limit:
        logger.info("%s Equalized Odds TPR차 %.4f / FPR차 %.4f (기준 %.2f 충족)",
                    attribute, tpr_diff, fpr_diff, limit)

    return {
        "attribute": attribute,
        "tpr_by_group": {k: float(v) for k, v in tpr.items()},
        "fpr_by_group": {k: float(v) for k, v in fpr.items()},
        "equalized_odds_tpr_diff": tpr_diff,
        "equalized_odds_fpr_diff": fpr_diff,
        "threshold_max": limit,
        "passes": bool(tpr_diff <= limit and fpr_diff <= limit),
    }


def evaluate(
    y_true: np.ndarray,
    proba: np.ndarray,
    sensitive_frame: pd.DataFrame,
    config: dict[str, Any] | None = None,
    threshold: float | None = None,
) -> dict[str, Any]:
    """보호 속성별로 DI Ratio 와 Equalized Odds 를 계산한다."""
    cfg = config if config is not None else load_config()
    thr = cfg["thresholds"]["approval_threshold"] if threshold is None else threshold
    disadvantaged = cfg["fairness"]["disadvantaged_groups"]

    results: dict[str, Any] = {
        "threshold": thr,
        "overall_approval_rate": float(approval_mask(proba, thr).mean()),
    }
    for attribute in cfg["fairness"]["sensitive_attributes"]:
        column = sensitive_frame[attribute].rename(attribute)
        results[attribute] = {
            "disparate_impact": disparate_impact(
                proba, column, thr, disadvantaged[attribute], cfg
            ),
            "equalized_odds": equalized_odds(y_true, proba, column, thr, cfg),
        }
    return results
