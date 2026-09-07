"""평가 지표: AUC-ROC, KS 통계량, PSI (SPEC §2 성능 목표)."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

logger = logging.getLogger(__name__)


def auc_score(y_true: np.ndarray | pd.Series, y_score: np.ndarray) -> float:
    """AUC-ROC."""
    return float(roc_auc_score(y_true, y_score))


def ks_statistic(y_true: np.ndarray | pd.Series, y_score: np.ndarray) -> float:
    """KS 통계량 = max|TPR - FPR|.

    양성/음성 점수 분포의 최대 누적분포 차이와 같다.
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype="float64")
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")

    thresholds = np.sort(np.unique(y_score))
    # 각 임계값에서의 누적분포. searchsorted 로 한 번에 계산한다.
    cdf_pos = np.searchsorted(np.sort(pos), thresholds, side="right") / len(pos)
    cdf_neg = np.searchsorted(np.sort(neg), thresholds, side="right") / len(neg)
    return float(np.max(np.abs(cdf_pos - cdf_neg)))


def psi(
    expected: np.ndarray | pd.Series,
    actual: np.ndarray | pd.Series,
    n_bins: int = 10,
    epsilon: float = 1e-6,
) -> float:
    """Population Stability Index.

    기준(expected, 보통 train) 분포의 분위수로 구간을 나누고,
    구간별 비율 차이를 sum((a - e) * ln(a / e)) 로 집계한다.
    해석: < 0.1 안정 / 0.1~0.25 주의 / > 0.25 불안정.
    """
    expected = np.asarray(expected, dtype="float64")
    actual = np.asarray(actual, dtype="float64")

    quantiles = np.linspace(0, 1, n_bins + 1)
    edges = np.unique(np.quantile(expected, quantiles))
    if len(edges) < 3:                      # 사실상 상수 분포
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf

    e_share = np.histogram(expected, bins=edges)[0] / len(expected)
    a_share = np.histogram(actual, bins=edges)[0] / len(actual)
    e_share = np.clip(e_share, epsilon, None)
    a_share = np.clip(a_share, epsilon, None)
    return float(np.sum((a_share - e_share) * np.log(a_share / e_share)))


def psi_by_feature(
    expected: pd.DataFrame, actual: pd.DataFrame, columns: list[str] | None = None
) -> pd.Series:
    """피처별 PSI. 리포트의 분포 안정성 검증용."""
    cols = columns if columns is not None else list(expected.columns)
    return pd.Series({c: psi(expected[c], actual[c]) for c in cols}).sort_values(ascending=False)


def classification_metrics(
    y_true: np.ndarray | pd.Series, y_score: np.ndarray, threshold: float
) -> dict[str, float]:
    """임계값 기반 지표 묶음. threshold 는 config 의 approval_threshold 를 쓴다."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = (np.asarray(y_score) >= threshold).astype(int)
    return {
        "auc": auc_score(y_true, y_score),
        "ks": ks_statistic(y_true, y_score),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def check_targets(metrics: dict[str, float], config: dict[str, Any]) -> dict[str, bool]:
    """SPEC §2 성능 목표 충족 여부."""
    return {
        "auc>=0.78": metrics.get("auc", 0.0) >= 0.78,
        "ks>=0.28": metrics.get("ks", 0.0) >= 0.28,
    }
