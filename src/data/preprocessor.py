"""전처리 및 Train/Valid/Test 분할.

핵심 규칙 (SPEC §1 전처리·분할, §7 제약 사항):
- 결측 대체값·스케일러·인코더는 **train 에만 fit** 하고 valid/test 는 transform 만 한다(누수 방지).
- 분할은 stratify=y 를 적용한 층화 분할이며 비율은 config 의 split 에서 읽는다.
- 씬파일러의 연체 3컬럼 결측은 중앙값으로 채우지 않는다. '정보 없음' 자체가 신호이므로
  센티넬 값(-1) 으로 바꾸고 별도 결측 플래그 컬럼을 남긴다.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from src.config import load_config, resolve_path

logger = logging.getLogger(__name__)


@dataclass
class Preprocessor:
    """train 에서 학습한 전처리 상태를 담고, valid/test 에 그대로 적용한다."""

    config: dict[str, Any]
    medians: dict[str, float] = field(default_factory=dict)
    categories: dict[str, list[str]] = field(default_factory=dict)
    feature_names: list[str] = field(default_factory=list)
    scaler: Any = None
    numeric_columns: list[str] = field(default_factory=list)
    fitted: bool = False

    # --- 설정 접근 ---------------------------------------------------------
    @property
    def _pre(self) -> dict[str, Any]:
        return self.config["preprocess"]

    @property
    def target_column(self) -> str:
        return self.config["data"]["target_column"]

    # --- 내부 변환 ---------------------------------------------------------
    def _add_missing_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """연체 컬럼: 결측 플래그를 만들고 센티넬 값으로 대체한다(중앙값 대체 금지)."""
        pre = self._pre
        out = df.copy()
        for col in pre["missing_indicator_columns"]:
            if col not in out.columns:
                continue
            flag = f"{col}{pre['missing_flag_suffix']}"
            out[flag] = out[col].isna().astype(int)
            out[col] = out[col].fillna(pre["missing_sentinel"])
        return out

    def _impute_median(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for col, value in self.medians.items():
            if col in out.columns:
                out[col] = out[col].fillna(value)
        return out

    def _one_hot(self, df: pd.DataFrame) -> pd.DataFrame:
        """train 에서 확정한 카테고리 목록으로 One-Hot 인코딩한다(컬럼 구성 고정)."""
        out = df.copy()
        for col, cats in self.categories.items():
            for cat in cats:
                out[f"{col}_{cat}"] = (out[col] == cat).astype(int)
            out = out.drop(columns=[col])
        return out

    # --- fit / transform ---------------------------------------------------
    def fit(self, x_train: pd.DataFrame) -> "Preprocessor":
        pre = self._pre
        work = self._add_missing_indicators(x_train)

        # 중앙값은 train 에서만 계산한다.
        self.medians = {
            col: float(work[col].median())
            for col in pre["median_impute_columns"]
            if col in work.columns
        }
        work = self._impute_median(work)

        # 카테고리 목록도 train 기준으로 고정한다.
        self.categories = {
            col: sorted(work[col].dropna().unique().tolist())
            for col in pre["categorical_columns"]
            if col in work.columns
        }
        work = self._one_hot(work)

        passthrough = [c for c in pre["passthrough_columns"] if c in work.columns]
        one_hot_cols = [f"{c}_{v}" for c, vals in self.categories.items() for v in vals]
        flag_cols = [
            f"{c}{pre['missing_flag_suffix']}"
            for c in pre["missing_indicator_columns"]
            if f"{c}{pre['missing_flag_suffix']}" in work.columns
        ]
        excluded = set(passthrough) | set(one_hot_cols) | set(flag_cols)
        self.numeric_columns = [c for c in work.columns if c not in excluded]

        scaler_kind = pre["scaler"]
        self.scaler = StandardScaler() if scaler_kind == "standard" else MinMaxScaler()
        self.scaler.fit(work[self.numeric_columns])

        self.feature_names = list(work.columns)
        self.fitted = True
        logger.info(
            "전처리 fit 완료: 수치형 %d개 스케일링(%s), One-Hot %d개, 결측플래그 %d개, 통과 %d개",
            len(self.numeric_columns), scaler_kind, len(one_hot_cols), len(flag_cols), len(passthrough),
        )
        logger.info("train 중앙값: %s", {k: round(v, 4) for k, v in self.medians.items()})
        return self

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted:
            raise RuntimeError("fit 을 먼저 호출해야 합니다. valid/test 는 transform 만 씁니다.")
        work = self._add_missing_indicators(x)
        work = self._impute_median(work)          # train 중앙값 사용
        work = self._one_hot(work)                # train 카테고리 사용
        work = work.reindex(columns=self.feature_names, fill_value=0)
        work[self.numeric_columns] = self.scaler.transform(work[self.numeric_columns])
        return work

    def fit_transform(self, x_train: pd.DataFrame) -> pd.DataFrame:
        return self.fit(x_train).transform(x_train)

    # --- 저장 / 로드 -------------------------------------------------------
    def save(self, path: str | Path | None = None) -> Path:
        pre = self._pre
        target = Path(path) if path else resolve_path(pre["output_dir"]) / pre["scaler_file"]
        target.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, target)
        logger.info("전처리기 저장: %s", target)
        return target

    @staticmethod
    def load(path: str | Path) -> "Preprocessor":
        return joblib.load(Path(path))


def feature_columns(config: dict[str, Any], group: str) -> list[str]:
    """피처 그룹별 학습 컬럼 목록 (ADR-008, ADR-009).

    group: "traditional_only" | "alternative_only" | "combined"
    보호 속성(gender, age_group)과 세그먼트 컬럼(is_thin_filer, credit_history_months)은
    어떤 그룹에도 포함되지 않는다.
    """
    feats = config["features"]
    pre = config["preprocess"]
    flags = [
        f"{c}{pre['missing_flag_suffix']}" for c in pre["missing_indicator_columns"]
    ]
    traditional = list(feats["traditional"])
    if feats.get("include_missing_flags_in") == "traditional":
        traditional += flags

    groups = {
        "traditional_only": traditional,
        "alternative_only": list(feats["alternative"]),
        "combined": traditional + list(feats["alternative"]),
    }
    if group not in groups:
        raise ValueError(f"알 수 없는 피처 그룹: {group} (가능: {list(groups)})")
    return groups[group]


FEATURE_GROUPS = ("traditional_only", "alternative_only", "combined")


def split_data(
    df: pd.DataFrame, config: dict[str, Any] | None = None
) -> dict[str, tuple[pd.DataFrame, pd.Series]]:
    """Train/Valid/Test 층화 분할. 비율과 random_state 는 config 에서 읽는다."""
    cfg = config if config is not None else load_config()
    split = cfg["split"]
    target = cfg["data"]["target_column"]
    seed = split["random_state"]

    x = df.drop(columns=[target])
    y = df[target]
    stratify_1 = y if split.get("stratify", True) else None

    # 1단계: train vs (valid + test)
    holdout_size = split["valid"] + split["test"]
    x_train, x_hold, y_train, y_hold = train_test_split(
        x, y, test_size=holdout_size, random_state=seed, stratify=stratify_1
    )
    # 2단계: holdout 을 valid / test 로 쪼갠다.
    test_share = split["test"] / holdout_size
    stratify_2 = y_hold if split.get("stratify", True) else None
    x_valid, x_test, y_valid, y_test = train_test_split(
        x_hold, y_hold, test_size=test_share, random_state=seed, stratify=stratify_2
    )

    parts = {
        "train": (x_train, y_train),
        "valid": (x_valid, y_valid),
        "test": (x_test, y_test),
    }
    total = len(df)
    for name, (x_part, y_part) in parts.items():
        logger.info(
            "%-5s %s행 (%.4f) / 양성률 %.4f%%",
            name, f"{len(x_part):,}", len(x_part) / total, y_part.mean() * 100,
        )
    return parts


def prepare(
    df: pd.DataFrame,
    config: dict[str, Any] | None = None,
    save: bool = False,
) -> tuple[dict[str, tuple[pd.DataFrame, pd.Series]], Preprocessor]:
    """분할 → train 에만 fit → 전 분할 transform. 필요 시 결과와 전처리기를 저장한다."""
    cfg = config if config is not None else load_config()
    parts = split_data(df, cfg)

    pre = Preprocessor(cfg)
    pre.fit(parts["train"][0])

    processed = {name: (pre.transform(x), y) for name, (x, y) in parts.items()}

    if save:
        out_dir = resolve_path(cfg["preprocess"]["output_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)
        for name, (x, y) in processed.items():
            frame = x.copy()
            frame[cfg["data"]["target_column"]] = y.to_numpy()
            path = out_dir / f"{name}.parquet"
            frame.to_parquet(path, index=False)
            logger.info("전처리 데이터 저장: %s (%s행 x %d열)", path, f"{len(frame):,}", frame.shape[1])
        pre.save()

    return processed, pre


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    from src.data.loader import load_gmsc
    from src.data.simulator import simulate

    cfg = load_config()
    data = simulate(load_gmsc(cfg), cfg)
    prepare(data, cfg, save=True)
