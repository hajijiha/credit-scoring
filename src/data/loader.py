"""데이터 로더.

규칙:
- 경로·파일명·타겟 컬럼은 config/config.yaml 에서만 읽는다. 코드에 하드코딩하지 않는다.
- 모든 경로는 프로젝트 루트 기준 상대경로다.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

logger = logging.getLogger(__name__)

# src/data/loader.py -> src/data -> src -> 프로젝트 루트
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ADR-007: 성년 미만은 age_group(20-34/35-54/55+) 배정이 불가능하므로 제거한다.
MIN_AGE = 18


def load_config(config_path: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    """config.yaml 을 읽어 dict 로 반환한다.

    경로 우선순위: 인자 > 환경변수 CONFIG_PATH > 기본값 config/config.yaml
    상대경로는 프로젝트 루트 기준으로 해석한다.
    """
    raw = config_path or os.getenv("CONFIG_PATH") or "config/config.yaml"
    path = Path(raw)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(relative: str) -> Path:
    """config 의 상대경로를 프로젝트 루트 기준 절대경로로 변환한다."""
    return PROJECT_ROOT / relative


def _log_frame_profile(df: pd.DataFrame, name: str) -> None:
    """컬럼명 / dtype / 결측 비율을 로그로 남긴다."""
    logger.info("[%s] shape=%s", name, df.shape)
    logger.info("[%s] %-42s %-10s %10s %8s", name, "column", "dtype", "missing", "ratio")
    for col in df.columns:
        n_missing = int(df[col].isna().sum())
        ratio = n_missing / len(df) if len(df) else 0.0
        logger.info(
            "[%s] %-42s %-10s %10s %7.2f%%",
            name, col, str(df[col].dtype), f"{n_missing:,}", ratio * 100,
        )


def load_gmsc(config: dict[str, Any] | None = None) -> pd.DataFrame:
    """GMSC(Give Me Some Credit) 원본을 로드한다. ADR-006 의 메인 데이터셋.

    - 이름 없는 인덱스 컬럼 제거
    - age < 18 행 제거 (ADR-007), 제거 건수 로그
    - 컬럼명 / dtype / 결측 비율 로그
    """
    cfg = config if config is not None else load_config()
    csv_path = resolve_path(cfg["paths"]["data_raw"]) / cfg["data"]["gmsc_file"]
    target = cfg["data"]["target_column"]

    if not csv_path.exists():
        raise FileNotFoundError(
            f"GMSC 원본을 찾을 수 없습니다: {csv_path}\n"
            "README.md 의 '데이터 내려받기' 절차를 먼저 실행하세요."
        )

    df = pd.read_csv(csv_path)
    logger.info("GMSC 로드: %s (%s행)", csv_path.relative_to(PROJECT_ROOT), f"{len(df):,}")

    # 원본 첫 컬럼은 이름 없는 행 번호다.
    unnamed = [c for c in df.columns if str(c).startswith("Unnamed")]
    if unnamed:
        df = df.drop(columns=unnamed)
        logger.info("인덱스 컬럼 제거: %s", unnamed)

    if target not in df.columns:
        raise KeyError(f"타겟 컬럼 '{target}' 이 데이터에 없습니다. 실제 컬럼: {list(df.columns)}")

    # ADR-007
    n_before = len(df)
    dropped = df[df["age"] < MIN_AGE]
    if len(dropped):
        logger.info(
            "ADR-007 age<%d 제거: %s건 (제거된 age 값: %s)",
            MIN_AGE, f"{len(dropped):,}", sorted(dropped["age"].unique().tolist()),
        )
    df = df[df["age"] >= MIN_AGE].reset_index(drop=True)
    logger.info("age 필터: %s행 -> %s행 (%s건 제거)",
                f"{n_before:,}", f"{len(df):,}", f"{n_before - len(df):,}")

    _log_frame_profile(df, "gmsc")
    pos_rate = df[target].mean()
    logger.info("타겟 '%s' 양성 %s건 / %s행 = %.3f%%",
                target, f"{int(df[target].sum()):,}", f"{len(df):,}", pos_rate * 100)

    return df


def load_german_credit() -> pd.DataFrame:
    """German Credit (UCI id=144) 로더.

    ADR-006 에 따라 메인 분석에는 사용하지 않는다. 스펙 대응 및 보조 검증용으로만 둔다.
    네트워크 접근이 필요하다.
    """
    from ucimlrepo import fetch_ucirepo  # 선택적 의존성 취급: 호출 시점에만 import

    logger.info("German Credit(UCI id=144) 로드 — ADR-006 에 따라 메인 분석에는 미사용")
    dataset = fetch_ucirepo(id=144)
    df = pd.concat([dataset.data.features, dataset.data.targets], axis=1)
    _log_frame_profile(df, "german_credit")
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    load_gmsc()
