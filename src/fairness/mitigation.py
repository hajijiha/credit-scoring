"""Bias Mitigation (SPEC §4, ADR-004).

Fairlearn `ThresholdOptimizer(constraints="equalized_odds")` 를 적용하고
완화 전/후의 공정성 지표와 성능 지표를 비교한다.

학습 규칙
--------
ThresholdOptimizer 는 **valid 세트**로 적합하고 **test 세트**로 평가한다.
테스트 라벨을 임계값 조정에 쓰면 SPEC §7 위반이다.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from fairlearn.postprocessing import ThresholdOptimizer
from sklearn.base import BaseEstimator, ClassifierMixin

from src.config import load_config
from src.fairness.metrics import approval_mask, disparate_impact, equalized_odds
from src.models.evaluate import auc_score

logger = logging.getLogger(__name__)


class Float64ProbaEstimator(ClassifierMixin, BaseEstimator):
    """학습된 모델을 감싸 predict_proba 를 float64 로 돌려주는 어댑터.

    두 가지 문제를 함께 해결한다.
    1) XGBoost 는 float32 확률을 돌려주는데, fairlearn 내부에서 float32 Series 에
       float64 배열을 대입하다 pandas 3.x 가 TypeError 를 낸다.
    2) sklearn 의 check_is_fitted 는 BaseEstimator 계열을 요구한다(__sklearn_tags__).
    """

    def __init__(self, model: Any = None) -> None:
        self.model = model

    def fit(self, x, y=None):                      # prefit=True 라 실제로는 호출되지 않는다
        self.classes_ = np.asarray(getattr(self.model, "classes_", [0, 1]))
        return self

    def __sklearn_is_fitted__(self) -> bool:
        return self.model is not None

    def predict_proba(self, x) -> np.ndarray:
        return np.asarray(self.model.predict_proba(x), dtype="float64")

    def predict(self, x) -> np.ndarray:
        return np.asarray(self.model.predict(x))


def fit_threshold_optimizer(
    model: Any,
    columns: list[str],
    x_valid: pd.DataFrame,
    y_valid: pd.Series,
    sensitive_valid: pd.Series,
    config: dict[str, Any] | None = None,
) -> ThresholdOptimizer:
    """valid 세트로 ThresholdOptimizer 를 적합한다."""
    cfg = config if config is not None else load_config()
    optimizer = ThresholdOptimizer(
        estimator=Float64ProbaEstimator(model),
        constraints=cfg["fairness"]["mitigation"]["constraints"],
        objective="accuracy_score",
        prefit=True,
        predict_method="predict_proba",
    )
    optimizer.fit(
        x_valid[columns], np.asarray(y_valid), sensitive_features=np.asarray(sensitive_valid)
    )
    return optimizer


def adjusted_thresholds(optimizer: ThresholdOptimizer) -> dict[str, Any]:
    """집단별로 선택된 임계값(보간 규칙)을 사람이 읽을 형태로 뽑는다."""
    out: dict[str, Any] = {}
    for group, rule in optimizer.interpolated_thresholder_.interpolation_dict.items():
        entry: dict[str, Any] = {}
        for key, value in rule.items():
            if hasattr(value, "threshold"):        # ThresholdOperation
                entry[key] = {
                    "operator": value.operator, "threshold": float(value.threshold),
                }
            else:
                entry[key] = float(value)
        out[str(group)] = entry
    return out


def compare(
    y_true: np.ndarray,
    proba: np.ndarray,
    mitigated_pred: np.ndarray,
    sensitive: pd.Series,
    disadvantaged: str,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """완화 전/후 지표 비교.

    before: 고정 임계값(config 의 approval_threshold) 기반 결정
    after : ThresholdOptimizer 가 낸 집단별 결정 (1 = 부도 예측 = 거절)
    """
    cfg = config if config is not None else load_config()
    thr = cfg["thresholds"]["approval_threshold"]
    lo, hi = cfg["thresholds"]["di_ratio_min"], cfg["thresholds"]["di_ratio_max"]
    limit = cfg["thresholds"]["eo_diff_max"]

    before = {
        "disparate_impact": disparate_impact(proba, sensitive, thr, disadvantaged, cfg),
        "equalized_odds": equalized_odds(y_true, proba, sensitive, thr, cfg),
        "approval_rate": float(approval_mask(proba, thr).mean()),
        "auc": auc_score(y_true, proba),
        "accuracy": float(((~approval_mask(proba, thr)).astype(int) == y_true).mean()),
    }

    # 완화 후에는 확률이 아니라 결정이 나오므로, 승인(=0)을 확률 0, 거절(=1)을 확률 1 로 두고
    # 같은 임계값 로직을 그대로 태운다.
    pseudo = np.asarray(mitigated_pred, dtype="float64")
    after = {
        "disparate_impact": disparate_impact(pseudo, sensitive, 0.5, disadvantaged, cfg),
        "equalized_odds": equalized_odds(y_true, pseudo, sensitive, 0.5, cfg),
        "approval_rate": float((pseudo < 0.5).mean()),
        "auc": auc_score(y_true, pseudo),          # 이진 결정이라 순위 정보가 사라진다
        "accuracy": float((pseudo.astype(int) == y_true).mean()),
    }

    return {
        "attribute": sensitive.name,
        "before": before,
        "after": after,
        "delta": {
            "di_ratio": after["disparate_impact"]["di_ratio"]
                        - before["disparate_impact"]["di_ratio"],
            "eo_tpr_diff": after["equalized_odds"]["equalized_odds_tpr_diff"]
                           - before["equalized_odds"]["equalized_odds_tpr_diff"],
            "eo_fpr_diff": after["equalized_odds"]["equalized_odds_fpr_diff"]
                           - before["equalized_odds"]["equalized_odds_fpr_diff"],
            "approval_rate": after["approval_rate"] - before["approval_rate"],
            "auc": after["auc"] - before["auc"],
        },
        "criteria": {"di_min": lo, "di_max": hi, "eo_diff_max": limit},
    }


def run_mitigation(
    model: Any,
    columns: list[str],
    valid: pd.DataFrame,
    test: pd.DataFrame,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """보호 속성별로 ThresholdOptimizer 를 적합하고 전/후를 비교한다."""
    cfg = config if config is not None else load_config()
    target = cfg["data"]["target_column"]
    seed = cfg["project"]["random_state"]
    disadvantaged = cfg["fairness"]["disadvantaged_groups"]

    proba_test = model.predict_proba(test[columns])[:, 1]
    y_test = test[target].to_numpy()

    results: dict[str, Any] = {}
    for attribute in cfg["fairness"]["sensitive_attributes"]:
        optimizer = fit_threshold_optimizer(
            model, columns, valid, valid[target], valid[attribute].rename(attribute), cfg
        )
        predictions = optimizer.predict(
            test[columns], sensitive_features=test[attribute].to_numpy(), random_state=seed
        )
        comparison = compare(
            y_test, proba_test, predictions, test[attribute].rename(attribute),
            disadvantaged[attribute], cfg,
        )
        comparison["adjusted_thresholds"] = adjusted_thresholds(optimizer)
        results[attribute] = comparison
        logger.info(
            "[%s] DI %.4f -> %.4f | EO TPR차 %.4f -> %.4f | FPR차 %.4f -> %.4f | "
            "승인율 %.4f -> %.4f | AUC %.4f -> %.4f",
            attribute,
            comparison["before"]["disparate_impact"]["di_ratio"],
            comparison["after"]["disparate_impact"]["di_ratio"],
            comparison["before"]["equalized_odds"]["equalized_odds_tpr_diff"],
            comparison["after"]["equalized_odds"]["equalized_odds_tpr_diff"],
            comparison["before"]["equalized_odds"]["equalized_odds_fpr_diff"],
            comparison["after"]["equalized_odds"]["equalized_odds_fpr_diff"],
            comparison["before"]["approval_rate"], comparison["after"]["approval_rate"],
            comparison["before"]["auc"], comparison["after"]["auc"],
        )
    return results
