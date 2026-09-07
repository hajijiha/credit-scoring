"""데이터 로더.

규칙:
- 경로·파일명·타겟 컬럼은 config/config.yaml 에서만 읽는다. 코드에 하드코딩하지 않는다.
- 모든 경로는 프로젝트 루트 기준 상대경로다.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from src.config import PROJECT_ROOT, load_config, resolve_path

logger = logging.getLogger(__name__)

# ADR-007: 성년 미만은 age_group(20-34/35-54/55+) 배정이 불가능하므로 제거한다.
MIN_AGE = 18


def frame_profile(df: pd.DataFrame) -> str:
    """컬럼명 / dtype / 결측 건수 / 결측 비율 표를 문자열로 만든다."""
    name_w = max([len(str(c)) for c in df.columns] + [len("column")])
    type_w = max([len(str(t)) for t in df.dtypes] + [len("dtype")])
    header = f"{'column':<{name_w}}  {'dtype':<{type_w}}  {'missing':>10}  {'ratio':>7}"
    lines = [header, "-" * len(header)]
    n_rows = len(df)
    for col in df.columns:
        n_missing = int(df[col].isna().sum())
        ratio = (n_missing / n_rows * 100) if n_rows else 0.0
        lines.append(
            f"{col!s:<{name_w}}  {df[col].dtype!s:<{type_w}}  {n_missing:>10,}  {ratio:>6.2f}%"
        )
    return "\n".join(lines)


def _log_frame_profile(df: pd.DataFrame, name: str) -> None:
    """컬럼 프로파일을 로그로 남긴다.

    표 전체를 한 개의 로그 레코드로 내보낸다. 행마다 따로 찍으면 로그 핸들러의
    프리픽스·줄바꿈 처리에 따라 표가 어긋나 보일 수 있기 때문이다.
    """
    logger.info("[%s] shape=%s\n%s", name, df.shape, frame_profile(df))


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
