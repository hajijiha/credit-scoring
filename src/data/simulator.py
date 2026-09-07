"""대안 데이터 시뮬레이터 (ADR-001, ADR-002).

GMSC 원본에 다음을 합성한다.

1. 인구통계 컬럼: gender(male/female, 무작위 50:50), age_group(실제 age 기반)
2. 대안 데이터 5변수: target · 소득 · 연령을 신호원으로 생성하고, target 과의
   상관계수 절댓값이 config 의 simulator.target_correlation 범위(0.3~0.5)에 들도록 맞춘다.
3. 씬파일러 처리: thin_filer_ratio 비율을 무작위 선정해 연체 3컬럼을 결측 처리하고
   RevolvingUtilization / NumberOfOpenCreditLinesAndLoans 에 강한 노이즈를 준다.
4. 편향 주입: female / 55+ 집단의 telecom·utility_payment_rate 를 bias_ratio × 0.1 만큼
   하향한다. 타겟은 절대 건드리지 않는다.

모든 파라미터는 config/config.yaml 에서 읽는다.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm

from src.config import load_config

logger = logging.getLogger(__name__)

# 각 대안 피처의 값 변환 방식. 모두 "값이 클수록 성실한 금융행동" 방향이므로
# target(부도=1) 과는 음의 상관을 갖도록 생성한다.
_FEATURE_KIND = {
    "telecom_payment_rate": "rate",
    "utility_payment_rate": "rate",
    "spending_consistency": "unit",
    "regular_payment_count": "count24",
    "app_login_frequency": "count40",
}


def _standardize(x: np.ndarray) -> np.ndarray:
    sd = x.std(ddof=0)
    return (x - x.mean()) / sd if sd > 0 else np.zeros_like(x)


def _transform(v: np.ndarray, kind: str) -> np.ndarray:
    """표준정규 잠재변수를 피처별 실제 스케일로 옮긴다(단조 변환)."""
    u = norm.cdf(v)
    if kind == "rate":            # 납부율: 0.60 ~ 1.00
        return 0.60 + 0.40 * u
    if kind == "unit":            # 소비 일관성 지수: 0 ~ 1
        return u
    if kind == "count24":         # 연간 정기결제 건수: 1 ~ 24
        return np.round(1 + 23 * u)
    if kind == "count40":         # 월간 앱 접속 횟수: 0 ~ 40
        return np.round(40 * u)
    raise ValueError(f"알 수 없는 변환 종류: {kind}")


def _nuisance_signal(df: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
    """소득·연령 기반 신호. 소득 결측은 중앙값으로 채워 신호 계산에만 쓴다(원본 미변경)."""
    income = df["MonthlyIncome"].astype("float64")
    income = income.fillna(income.median())
    income_z = _standardize(np.log1p(np.clip(income.to_numpy(), 0, None)))
    age_z = _standardize(df["age"].astype("float64").to_numpy())
    return 0.7 * income_z + 0.3 * age_z


def _latent_with_corr(t_z: np.ndarray, nuisance: np.ndarray, eps: np.ndarray, rho: float) -> np.ndarray:
    """target 과의 상관이 정확히 rho 인 표준정규 잠재변수를 만든다.

    나머지 성분(소득·연령·노이즈)을 t_z 에 대해 직교화한 뒤 결합하므로
    corr(v, t_z) = rho 가 (표본 기준) 정확히 성립한다.
    """
    m = _standardize(0.6 * nuisance + 0.8 * eps)
    m = m - t_z * (float(m @ t_z) / float(t_z @ t_z))   # t_z 성분 제거
    m = _standardize(m)
    return rho * t_z + np.sqrt(max(1.0 - rho**2, 0.0)) * m


def _fit_feature(
    t_z: np.ndarray,
    target: np.ndarray,
    nuisance: np.ndarray,
    eps: np.ndarray,
    kind: str,
    goal: float,
    max_iter: int = 30,
    tol: float = 1e-3,
) -> tuple[np.ndarray, float]:
    """변환 후의 |corr(feature, target)| 이 goal 이 되도록 잠재 상관을 보정한다.

    norm.cdf 와 반올림은 비선형이라 잠재 상관이 그대로 유지되지 않는다.
    고정점 반복으로 변환 후 상관을 목표에 맞춘다.
    """
    rho = -goal          # 값이 클수록 우량 → 부도와 음의 상관
    values = _transform(_latent_with_corr(t_z, nuisance, eps, rho), kind)
    achieved = float(np.corrcoef(values, target)[0, 1])
    for _ in range(max_iter):
        if abs(abs(achieved) - goal) <= tol:
            break
        rho = float(np.clip(rho * goal / max(abs(achieved), 1e-6), -0.99, -0.01))
        values = _transform(_latent_with_corr(t_z, nuisance, eps, rho), kind)
        achieved = float(np.corrcoef(values, target)[0, 1])
    return values, achieved


def add_demographics(df: pd.DataFrame, config: dict[str, Any] | None = None) -> pd.DataFrame:
    """gender(무작위 50:50)와 age_group(실제 age 기반) 컬럼을 추가한다."""
    cfg = config if config is not None else load_config()
    sim = cfg["simulator"]
    rng = np.random.default_rng(sim["random_state"])
    df = df.copy()

    # GMSC 에는 성별이 없다. 공정성 분석용으로 무작위 합성한다(타겟과 독립).
    df[sim["gender_column"]] = rng.choice(cfg["fairness"]["groups"]["gender"], size=len(df))

    bins = sim["age_group_bins"]
    edges = [b["min"] for b in bins] + [np.inf if bins[-1]["max"] is None else bins[-1]["max"]]
    labels = [b["label"] for b in bins]
    df[sim["age_group_column"]] = pd.cut(
        df["age"], bins=edges, labels=labels, right=False, include_lowest=True
    ).astype("object")

    n_unassigned = int(df[sim["age_group_column"]].isna().sum())
    if n_unassigned:
        raise ValueError(f"age_group 배정 실패 {n_unassigned}건 — ADR-007 필터를 확인하세요.")
    logger.info("인구통계 생성: gender %s / age_group %s",
                df[sim["gender_column"]].value_counts().to_dict(),
                df[sim["age_group_column"]].value_counts().to_dict())
    return df


def add_alternative_features(df: pd.DataFrame, config: dict[str, Any] | None = None) -> pd.DataFrame:
    """대안 데이터 5변수를 생성한다 (ADR-001)."""
    cfg = config if config is not None else load_config()
    sim = cfg["simulator"]
    features = sim["alternative_features"]
    lo = sim["target_correlation"]["min"]
    hi = sim["target_correlation"]["max"]

    df = df.copy()
    target = df[cfg["data"]["target_column"]].astype("float64").to_numpy()
    t_z = _standardize(target)
    nuisance = _nuisance_signal(df, np.random.default_rng(sim["random_state"]))
    rng = np.random.default_rng(sim["random_state"])

    # 목표 상관은 설정 범위 [min, max] 안에 여유를 두고 균등 배치한다.
    # 편향 주입 대상 피처(telecom/utility)는 apply_bias 에서 값이 하향되면서 상관이 함께
    # 떨어지므로, 여유가 큰 상단 목표값을 배정해 bias_ratio 를 올려도 범위를 벗어나지 않게 한다.
    margin = 0.02 * (hi - lo)
    ladder = sorted(np.linspace(lo + margin, hi - margin, len(features)), reverse=True)
    biased = [f for f in features if f in sim["biased_features"]]
    ordered = biased + [f for f in features if f not in biased]
    goal_of = dict(zip(ordered, ladder))

    for feature in features:
        goal = goal_of[feature]
        eps = rng.normal(size=len(df))
        values, achieved = _fit_feature(
            t_z, target, nuisance, eps, _FEATURE_KIND[feature], float(goal)
        )
        df[feature] = values
        logger.info("%-24s 목표 |corr|=%.3f -> 실제 corr=%+.4f", feature, goal, achieved)
    return df


def apply_thin_filer(
    df: pd.DataFrame,
    config: dict[str, Any] | None = None,
    thin_filer_ratio: float | None = None,
) -> pd.DataFrame:
    """씬파일러를 무작위 선정해 전통 피처의 정보량을 줄인다 (ADR-001).

    - 연체 3컬럼 결측 처리
    - RevolvingUtilization / NumberOfOpenCreditLinesAndLoans 에 강한 노이즈
    - is_thin_filer 는 선정 마스크가 아니라 '연체 관련 컬럼 2개 이상 결측' 정의로 산출한다.
    """
    cfg = config if config is not None else load_config()
    sim = cfg["simulator"]
    ratio = sim["thin_filer_ratio"] if thin_filer_ratio is None else thin_filer_ratio
    rng = np.random.default_rng(sim["random_state"])
    df = df.copy()

    n = len(df)
    n_thin = int(round(n * ratio))
    selected = np.zeros(n, dtype=bool)
    if n_thin:
        selected[rng.choice(n, size=n_thin, replace=False)] = True

    masked = sim["thin_filer_masked_columns"]
    for col in masked:
        df[col] = df[col].astype("float64")
        df.loc[selected, col] = np.nan

    noise = sim["thin_filer_noise"]
    rev_col, open_col = sim["thin_filer_noised_columns"]
    if selected.any():
        idx = df.index[selected]
        factor = np.exp(rng.normal(0.0, noise["revolving_lognormal_sigma"], size=selected.sum()))
        df.loc[idx, rev_col] = df.loc[idx, rev_col].to_numpy() * factor
        shift = np.round(rng.normal(0.0, noise["open_credit_lines_sigma"], size=selected.sum()))
        df.loc[idx, open_col] = np.clip(df.loc[idx, open_col].to_numpy() + shift, 0, None)

    flag_col = sim["thin_filer_flag_column"]
    df[flag_col] = (df[masked].isna().sum(axis=1) >= sim["thin_filer_min_missing"]).astype(int)

    logger.info(
        "씬파일러 처리: 선정 %s건(%.3f) / %s=1 %s건(%.3f) / 연체 3컬럼 결측률 %s",
        f"{int(selected.sum()):,}", selected.mean(),
        flag_col, f"{int(df[flag_col].sum()):,}", df[flag_col].mean(),
        {c: round(float(df[c].isna().mean()), 4) for c in masked},
    )
    return df


def apply_bias(
    df: pd.DataFrame,
    config: dict[str, Any] | None = None,
    bias_ratio: float | None = None,
) -> pd.DataFrame:
    """female / 55+ 집단의 telecom·utility_payment_rate 를 하향한다 (ADR-002).

    하향폭 = bias_ratio × bias_shift_per_ratio. 두 집단에 모두 속하면 가산 적용된다.
    타겟은 건드리지 않는다.
    """
    cfg = config if config is not None else load_config()
    sim = cfg["simulator"]
    ratio = sim["bias_ratio"] if bias_ratio is None else bias_ratio
    shift = ratio * sim["bias_shift_per_ratio"]
    df = df.copy()
    if shift == 0:
        logger.info("편향 주입: bias_ratio=0 — 변경 없음")
        return df

    disadvantaged = cfg["fairness"]["disadvantaged_groups"]
    penalty = np.zeros(len(df), dtype="float64")
    for attr, group in disadvantaged.items():
        col = sim["gender_column"] if attr == "gender" else sim["age_group_column"]
        penalty += (df[col] == group).to_numpy().astype("float64") * shift

    for feature in sim["biased_features"]:
        df[feature] = np.clip(df[feature].to_numpy() - penalty, 0.0, 1.0)

    logger.info("편향 주입: bias_ratio=%.3f, 집단당 하향폭 %.4f, 대상 %s",
                ratio, shift, sim["biased_features"])
    return df


def simulate(
    df: pd.DataFrame,
    config: dict[str, Any] | None = None,
    thin_filer_ratio: float | None = None,
    bias_ratio: float | None = None,
) -> pd.DataFrame:
    """전체 시뮬레이션 파이프라인."""
    cfg = config if config is not None else load_config()
    out = add_demographics(df, cfg)
    out = add_alternative_features(out, cfg)
    out = apply_thin_filer(out, cfg, thin_filer_ratio=thin_filer_ratio)
    out = apply_bias(out, cfg, bias_ratio=bias_ratio)
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    from src.data.loader import load_gmsc

    cfg = load_config()
    simulate(load_gmsc(cfg), cfg)
