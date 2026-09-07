"""학습 파이프라인 구성 요소 검증 (가벼운 단위 테스트만)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from imblearn.pipeline import Pipeline as ImbPipeline

from src.config import load_config
from src.models.train import (
    algorithms,
    build_estimator,
    cross_validate_auc,
    imbalance_methods,
)


@pytest.fixture(scope="module")
def config() -> dict:
    return load_config()


@pytest.fixture(scope="module")
def toy():
    rng = np.random.default_rng(0)
    n = 1200
    x = pd.DataFrame(rng.normal(size=(n, 4)), columns=[f"f{i}" for i in range(4)])
    y = pd.Series((x["f0"] + rng.normal(scale=1.0, size=n) > 1.4).astype(int))
    return x, y


def test_all_combinations_build(config):
    """설정에 있는 알고리즘 × 불균형 처리 조합이 모두 생성돼야 한다."""
    built = 0
    for algorithm in algorithms(config):
        for imbalance in imbalance_methods(config):
            model = build_estimator(algorithm, imbalance, config, 13.0)
            assert hasattr(model, "fit") and hasattr(model, "predict_proba")
            built += 1
    assert built == len(algorithms(config)) * len(imbalance_methods(config)) == 6


def test_smote_is_pipeline_only_for_smote(config):
    """SMOTE 조합만 imblearn 파이프라인이어야 한다(리샘플링을 학습 fold 안에 가둔다)."""
    for algorithm in algorithms(config):
        assert isinstance(build_estimator(algorithm, "smote", config, 13.0), ImbPipeline)
        assert not isinstance(build_estimator(algorithm, "class_weight", config, 13.0), ImbPipeline)


def test_unknown_algorithm_raises(config):
    with pytest.raises(ValueError):
        build_estimator("randomforest", "class_weight", config, 1.0)


def test_hyperparameters_come_from_config(config):
    """하이퍼파라미터는 config.models 에서 와야 한다(하드코딩 금지)."""
    model = build_estimator("xgboost", "class_weight", config, 13.0)
    assert model.n_estimators == config["models"]["xgboost"]["n_estimators"]
    assert model.max_depth == config["models"]["xgboost"]["max_depth"]
    assert model.n_jobs == config["models"]["n_jobs"]


def test_cross_validate_returns_expected_fold_count(config, toy):
    """ADR-005: fold 개수 = n_splits × n_repeats."""
    x, y = toy
    scores = cross_validate_auc(
        lambda: build_estimator("logistic_regression", "class_weight", config, 1.0),
        x, y, n_splits=3, n_repeats=2, seed=42,
    )
    assert len(scores) == 6
    assert all(0.0 <= s <= 1.0 for s in scores)
    assert np.mean(scores) > 0.7, "신호가 있는 데이터인데 AUC 가 너무 낮다"


def test_cross_validate_is_reproducible(config, toy):
    x, y = toy
    kwargs = dict(x=x, y=y, n_splits=3, n_repeats=1, seed=42)
    factory = lambda: build_estimator("logistic_regression", "class_weight", config, 1.0)
    assert cross_validate_auc(factory, **kwargs) == cross_validate_auc(factory, **kwargs)
