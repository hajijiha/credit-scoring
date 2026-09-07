"""평가 지표 검증."""

from __future__ import annotations

import numpy as np
import pytest

from src.models.evaluate import auc_score, ks_statistic, psi


def test_auc_perfect_and_random():
    y = np.array([0, 0, 1, 1])
    assert auc_score(y, np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
    assert auc_score(y, np.array([0.9, 0.8, 0.2, 0.1])) == 0.0


def test_ks_perfect_separation():
    """완전 분리면 KS = 1."""
    y = np.array([0] * 50 + [1] * 50)
    score = np.concatenate([np.zeros(50), np.ones(50)])
    assert ks_statistic(y, score) == pytest.approx(1.0)


def test_ks_no_separation():
    """양성/음성 점수 분포가 같으면 KS ≈ 0."""
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 4000)
    score = rng.random(4000)
    assert ks_statistic(y, score) < 0.1


def test_ks_between_zero_and_one():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 1000)
    score = rng.random(1000) * 0.5 + y * 0.3
    ks = ks_statistic(y, score)
    assert 0.0 <= ks <= 1.0


def test_psi_identical_distributions():
    """같은 분포면 PSI ≈ 0."""
    rng = np.random.default_rng(2)
    base = rng.normal(size=20_000)
    other = rng.normal(size=20_000)
    assert psi(base, other) < 0.01


def test_psi_shifted_distribution():
    """평균이 크게 이동하면 PSI 가 커진다."""
    rng = np.random.default_rng(3)
    base = rng.normal(size=20_000)
    shifted = rng.normal(loc=1.5, size=20_000)
    assert psi(base, shifted) > 0.25


def test_psi_monotone_in_shift():
    rng = np.random.default_rng(4)
    base = rng.normal(size=20_000)
    values = [psi(base, rng.normal(loc=s, size=20_000)) for s in (0.0, 0.3, 1.0)]
    assert values[0] < values[1] < values[2]
