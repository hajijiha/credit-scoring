"""SHAP 기반 설명가능성 모듈 (SPEC §3).

- 전역: Summary Plot (상위 15개), Dependence Plot (대안 1 + 전통 1)
- 지역: 개별 고객 Waterfall Plot
- 거절 사유 Top 5 자동 생성 (자연어 설명 포함)

부호 규약
--------
모델은 **부도 확률**을 예측한다. 따라서 SHAP 값이 양수면 부도 확률을 올린다(고객에게 불리).
스펙 §8 의 응답 예시는 `shap_contribution: -0.15` 처럼 **신용도 기여도** 관점(음수 = 불리)이므로,
`shap_contribution` 필드는 부호를 뒤집어 담는다. 원본 값은 `shap_value_default`에 함께 싣는다.
"""

from __future__ import annotations

import logging
import math
import shutil
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from src.config import load_config, load_yaml, resolve_path

logger = logging.getLogger(__name__)

# 자연어 문장에서 값을 어떻게 적을지. unit -> (포맷 함수, 비교 형용사 쌍)
# count 는 counter(회/건/개/명)를 붙여 별도로 처리한다.
_FORMATTERS = {
    "percent": (lambda v: f"{v * 100:.0f}%", ("높습니다", "낮습니다")),
    "ratio": (lambda v: f"{v:.2f}", ("높습니다", "낮습니다")),
    "amount": (lambda v: f"{v:,.0f}", ("많습니다", "적습니다")),
    "years": (lambda v: f"{v:.0f}세", ("많습니다", "적습니다")),
    "months": (lambda v: f"{v:.0f}개월", ("깁니다", "짧습니다")),
    "flag": (lambda v: "해당" if v >= 0.5 else "비해당", ("해당합니다", "해당하지 않습니다")),
}

# 숫자를 한글로 읽었을 때의 받침. 조사 선택용.
# 받침 있음: 0 영(ㅇ) 1 일(ㄹ) 3 삼(ㅁ) 6 육(ㄱ) 7 칠(ㄹ) 8 팔(ㄹ)
# 받침 없음: 2 이 4 사 5 오 9 구
_DIGITS_WITH_FINAL = set("013678")
# '로' 를 취하는 자릿수(모음 또는 ㄹ 받침): 1 일, 2 이, 4 사, 5 오, 7 칠, 8 팔, 9 구
_DIGITS_TAKING_RO = set("124578 9".replace(" ", ""))


def _has_final_consonant(word: str) -> bool:
    """마지막 글자를 한글로 읽었을 때 받침이 있는지. 조사 선택용."""
    if not word:
        return False
    last = word[-1]
    if "가" <= last <= "힣":
        return (ord(last) - 0xAC00) % 28 != 0
    if last.isdigit():
        return last in _DIGITS_WITH_FINAL
    if last == "%":                                # 퍼센트 -> 받침 없음
        return False
    return False


def _subject_particle(word: str) -> str:
    """이/가."""
    return "이" if _has_final_consonant(word) else "가"


def _instrumental_particle(word: str) -> str:
    """로/으로. 모음이나 ㄹ 받침 뒤에는 '로', 그 외 받침 뒤에는 '으로'."""
    if not word:
        return "로"
    last = word[-1]
    if "가" <= last <= "힣":
        final = (ord(last) - 0xAC00) % 28
        return "로" if final in (0, 8) else "으로"   # 8 = ㄹ
    if last.isdigit():
        return "로" if last in _DIGITS_TAKING_RO else "으로"
    return "로"


def _strip_negative_zero(text: str) -> str:
    """'-0회' 처럼 반올림으로 생긴 음의 0 표기를 정리한다.

    역스케일 왕복에서 0 이 -0.0004 로 돌아오는 경우가 있다.
    """
    if text.startswith("-") and not any(c in "123456789" for c in text):
        return text[1:]
    return text


def _format_value(value: float, meta: dict[str, Any]) -> str:
    """단위에 맞춰 값을 문자열로 만든다."""
    unit = meta.get("unit", "ratio")
    if unit == "count":
        return _strip_negative_zero(f"{value:.0f}") + meta.get("counter", "회")
    formatter, _ = _FORMATTERS.get(unit, _FORMATTERS["ratio"])
    return _strip_negative_zero(formatter(value))


def _comparison_words(meta: dict[str, Any]) -> tuple[str, str]:
    unit = meta.get("unit", "ratio")
    if unit == "count":
        return ("많습니다", "적습니다")
    _, words = _FORMATTERS.get(unit, _FORMATTERS["ratio"])
    return words


def shap_values_2d(explanation: Any) -> np.ndarray:
    """SHAP 값을 (n_samples, n_features) 2차원으로 정규화한다.

    shap 0.52 는 이진 분류 LightGBM/XGBoost 에서 (n, features) 2D 를 돌려주지만,
    구현·버전에 따라 (n, features, 2) 3D 가 나오기도 한다. 양쪽을 모두 받는다.
    """
    values = np.asarray(explanation.values if hasattr(explanation, "values") else explanation)
    if values.ndim == 3:
        return values[..., 1]           # 양성 클래스(부도)
    if values.ndim == 2:
        return values
    if values.ndim == 1:
        return values.reshape(1, -1)
    raise ValueError(f"예상하지 못한 SHAP 값 차원: {values.shape}")


def base_value_scalar(explanation: Any) -> float:
    """base value 를 스칼라로 뽑는다(클래스 축이 있으면 양성 클래스)."""
    base = np.asarray(explanation.base_values)
    if base.ndim == 2:
        base = base[:, 1] if base.shape[1] > 1 else base[:, 0]
    return float(np.ravel(base)[0])


class CreditExplainer:
    """챔피언 모델에 대한 SHAP 설명기."""

    def __init__(
        self,
        model: Any,
        feature_columns: list[str],
        reference: pd.DataFrame,
        config: dict[str, Any] | None = None,
    ) -> None:
        """
        model            : 학습된 트리 모델 (XGBoost / LightGBM)
        feature_columns  : 모델 입력 컬럼 순서
        reference        : 원 단위(역스케일)로 되돌린 참조 데이터. age_group 컬럼 포함
        """
        self.config = config if config is not None else load_config()
        self.ecfg = self.config["explainer"]
        self.names = load_yaml(self.ecfg["feature_names_file"])["features"]
        self.feature_columns = list(feature_columns)
        self.reference = reference
        self.group_column = self.ecfg["reference_group_column"]
        self.explainer = shap.TreeExplainer(model)

        # 동일 연령대 평균 (reference_value). 원 단위 기준.
        # 씬파일러의 연체 컬럼은 센티넬(-1)로 채워진 '정보 없음'이므로 평균에서 제외한다.
        # 포함하면 "동일 연령대 평균 -0회" 같은 값이 나와 설명이 성립하지 않는다.
        numeric = [c for c in self.feature_columns if c in reference.columns]
        masked = self._mask_missing_sentinels(reference[numeric + [self.group_column]])
        self._group_means = masked.groupby(self.group_column)[numeric].mean()
        self._overall_means = masked[numeric].mean()

    def _mask_missing_sentinels(self, frame: pd.DataFrame) -> pd.DataFrame:
        """결측 플래그가 선 행의 연체 컬럼 값을 NaN 으로 만들어 평균 계산에서 뺀다."""
        pre = self.config["preprocess"]
        suffix = pre["missing_flag_suffix"]
        out = frame.copy()
        for column in pre["missing_indicator_columns"]:
            flag = f"{column}{suffix}"
            if column in out.columns and flag in self.reference.columns:
                out.loc[self.reference[flag] == 1, column] = np.nan
        return out

    # --- SHAP 계산 ---------------------------------------------------------
    def explain(self, x: pd.DataFrame) -> tuple[np.ndarray, float]:
        """(shap_values 2D, base_value) 반환."""
        explanation = self.explainer(x[self.feature_columns])
        return shap_values_2d(explanation), base_value_scalar(explanation)

    def reference_value(self, feature: str, age_group: str) -> float:
        """동일 연령대 평균. 해당 집단이 없으면 전체 평균으로 대체."""
        if age_group in self._group_means.index and feature in self._group_means.columns:
            return float(self._group_means.loc[age_group, feature])
        return float(self._overall_means.get(feature, np.nan))

    # --- 자연어 설명 -------------------------------------------------------
    def _describe(
        self, feature: str, current: float, reference: float, contribution: float,
        missing: bool = False,
    ) -> tuple[str, bool]:
        """스펙 §3 형식의 자연어 설명과 부호-방향 일치 여부를 반환한다.

        예: "통신비 정상납부율이 78%로, 동일 연령대 평균(92%) 대비 낮습니다 (기여도: -0.15)"
        """
        meta = self.names.get(feature, {})
        label = meta.get("kr", feature)
        unit = meta.get("unit", "ratio")
        higher_is_better = meta.get("higher_is_better", True)
        above, below = _comparison_words(meta)
        current_text = _format_value(current, meta)
        reference_text = _format_value(reference, meta)

        if missing:
            # 씬파일러: 값 자체가 '정보 없음'이다. 비교 문장을 만들면 거짓이 된다.
            text = (
                f"{label} 기록이 없습니다. 동일 연령대 평균은 {reference_text}입니다 "
                f"(기여도: {contribution:+.2f})"
            )
            return text, True

        is_above = current >= reference
        comparison = above if is_above else below
        particle = _subject_particle(label)

        # 부호-방향 일치 검증:
        # contribution 은 신용도 기여도(음수 = 불리). 값이 유리한 방향인데 기여도가 음수면 불일치.
        favourable_value = is_above if higher_is_better else not is_above
        consistent = (contribution >= 0) == favourable_value

        if unit == "flag":
            text = (
                f"{label} 항목에 {'해당' if current >= 0.5 else '비해당'}합니다 "
                f"(기여도: {contribution:+.2f})"
            )
        else:
            text = (
                f"{label}{particle} {current_text}{_instrumental_particle(current_text)}, "
                f"동일 연령대 평균({reference_text}) 대비 "
                f"{comparison} (기여도: {contribution:+.2f})"
            )
        if not consistent:
            # 단순 비교로는 설명되지 않는 기여(상호작용·비선형)임을 밝힌다.
            text += " — 다른 항목과의 조합에 따른 영향입니다"
        return text, consistent

    def contributing_factors(
        self,
        row: pd.Series,
        age_group: str,
        adverse: bool = True,
        top_n: int | None = None,
        display_row: pd.Series | None = None,
    ) -> list[dict[str, Any]]:
        """기여도 상위 요인 Top N.

        adverse=True  : 부도 확률을 올린 요인 (거절 사유)
        adverse=False : 부도 확률을 낮춘 요인 (승인 사유)
        """
        n = top_n if top_n is not None else self.ecfg["top_rejection_reasons"]
        shown = display_row if display_row is not None else row
        frame = row[self.feature_columns].to_frame().T.astype("float64")
        shap_row = self.explain(frame)[0][0]

        order = np.argsort(-shap_row if adverse else shap_row)
        reasons: list[dict[str, Any]] = []
        for rank, idx in enumerate(order[:n], start=1):
            feature = self.feature_columns[idx]
            shap_default = float(shap_row[idx])
            contribution = -shap_default            # 신용도 기여도 (스펙 부호 규약)
            raw_value = shown[feature]
            # '기록 없음'은 두 형태로 들어온다. API 요청은 null(None/NaN) 그대로 오고,
            # 배치 경로(run_explain)는 전처리를 거친 센티넬(-1) + 결측 플래그 1 로 온다.
            flag = f"{feature}{self.config['preprocess']['missing_flag_suffix']}"
            flagged = flag in shown.index and shown[flag] == 1
            missing = flagged or raw_value is None or (
                isinstance(raw_value, float) and math.isnan(raw_value)
            )
            # 값이 없으면 전처리와 같은 센티넬로 표기한다(설명 문장은 '기록 없음'으로 낸다).
            current = (
                float(self.config["preprocess"]["missing_sentinel"]) if missing
                else float(raw_value)
            )
            reference = self.reference_value(feature, age_group)
            text, consistent = self._describe(
                feature, current, reference, contribution, missing=missing
            )
            reasons.append({
                "rank": rank,
                "feature": feature,
                "feature_name_kr": self.names.get(feature, {}).get("kr", feature),
                "current_value": round(current, 4),
                "reference_value": round(reference, 4),
                "shap_contribution": round(contribution, 4),
                "shap_value_default": round(shap_default, 4),
                "explanation": text,
                "direction_consistent": consistent,
            })
        return reasons

    def rejection_reasons(
        self,
        row: pd.Series,
        age_group: str,
        top_n: int | None = None,
        display_row: pd.Series | None = None,
    ) -> list[dict[str, Any]]:
        """부도 확률을 가장 크게 올린 요인 Top N 을 거절 사유로 반환한다.

        row         : **모델 입력 그대로**(전처리·스케일링된) 값. SHAP 계산에 쓴다.
        display_row : 자연어 설명에 쓸 원 단위 값. 생략하면 row 를 그대로 쓴다.

        두 값을 나누는 이유: SHAP 은 학습 때와 같은 스케일의 입력에서 계산해야 하고,
        설명 문장에는 "78%" 같은 원래 단위가 필요하기 때문이다.
        """
        return self.contributing_factors(
            row, age_group, adverse=True, top_n=top_n, display_row=display_row
        )

    def approval_factors(
        self,
        row: pd.Series,
        age_group: str,
        top_n: int | None = None,
        display_row: pd.Series | None = None,
    ) -> list[dict[str, Any]]:
        """부도 확률을 가장 크게 낮춘 요인 Top N (승인 사유)."""
        return self.contributing_factors(
            row, age_group, adverse=False, top_n=top_n, display_row=display_row
        )

    # --- 그림 -------------------------------------------------------------
    def summary_plot(self, x: pd.DataFrame, path: Path) -> Path:
        values, _ = self.explain(x)
        plt.figure()
        shap.summary_plot(
            values, x[self.feature_columns], max_display=self.ecfg["summary_max_display"],
            show=False,
        )
        plt.title(f"SHAP summary (top {self.ecfg['summary_max_display']})")
        plt.savefig(path, dpi=110, bbox_inches="tight")
        plt.close()
        return path

    def dependence_plot(self, feature: str, x: pd.DataFrame, path: Path) -> Path:
        values, _ = self.explain(x)
        plt.figure()
        shap.dependence_plot(
            feature, values, x[self.feature_columns], show=False, interaction_index="auto"
        )
        plt.savefig(path, dpi=110, bbox_inches="tight")
        plt.close()
        return path

    def waterfall_plot(self, row: pd.Series, path: Path, title: str = "") -> Path:
        frame = row[self.feature_columns].to_frame().T.astype("float64")
        values, base = self.explain(frame)
        explanation = shap.Explanation(
            values=values[0], base_values=base,
            data=frame.iloc[0].to_numpy(), feature_names=self.feature_columns,
        )
        plt.figure()
        shap.plots.waterfall(explanation, max_display=12, show=False)
        if title:
            plt.title(title, fontsize=11)
        plt.savefig(path, dpi=110, bbox_inches="tight")
        plt.close()
        return path

    def global_importance(self, x: pd.DataFrame) -> pd.DataFrame:
        """평균 |SHAP| 기준 전역 중요도."""
        values, _ = self.explain(x)
        importance = pd.DataFrame({
            "feature": self.feature_columns,
            "mean_abs_shap": np.abs(values).mean(axis=0),
        })
        alternative = set(self.config["features"]["alternative"])
        importance["feature_type"] = np.where(
            importance["feature"].isin(alternative), "alternative", "traditional"
        )
        importance["feature_name_kr"] = importance["feature"].map(
            lambda f: self.names.get(f, {}).get("kr", f)
        )
        return importance.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)


def copy_to_report_assets(paths: list[Path], config: dict[str, Any]) -> list[Path]:
    """산출 이미지를 리포트 자료 폴더에 복사한다."""
    target = resolve_path(config["explainer"]["report_assets_dir"])
    target.mkdir(parents=True, exist_ok=True)
    copied = []
    for path in paths:
        destination = target / path.name
        shutil.copy2(path, destination)
        copied.append(destination)
    return copied
