"""FastAPI 서버 (SPEC §6).

7개 엔드포인트: /health, /predict, /explain, /fairness, /fairness/mitigation, /metrics, /anova

설계 규칙
--------
- 모델·전처리기·SHAP 설명기는 **기동 시 1회** 로드한다. 요청마다 로드하지 않는다.
- `/fairness`, `/fairness/mitigation`, `/anova`, `/metrics` 는 **사전 계산 결과를 읽어** 반환한다.
- `/predict`, `/explain` 입력은 **원본 단위**로 받고 서버가 전처리한다.
  (F3 에서 겪은 스케일 불일치 버그를 구조적으로 막는다.)
- 설정값은 config.yaml / 환경변수에서만 읽는다.

실행:
    uvicorn src.api.main:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, status

from src.api import schemas
from src.config import load_config, resolve_path
from src.data.preprocessor import Preprocessor, feature_columns
from src.explainer.explainer import CreditExplainer
from src.models.train import champion_combination

logger = logging.getLogger(__name__)


class ServiceState:
    """기동 시 한 번 로드하는 서빙 자원."""

    def __init__(self) -> None:
        self.config: dict[str, Any] = {}
        self.model: Any = None
        self.preprocessor: Preprocessor | None = None
        self.explainer: CreditExplainer | None = None
        self.champion: tuple[str, str, str] | None = None
        self.feature_columns: list[str] = []
        self.fairness: dict[str, Any] = {}
        self.anova: dict[str, Any] = {}
        self.metrics: dict[str, float] = {}
        self.errors: dict[str, str] = {}

    @property
    def model_version(self) -> str:
        return self.config.get("models", {}).get("model_version", "unknown")

    @property
    def loaded(self) -> dict[str, bool]:
        return {
            "model": self.model is not None,
            "preprocessor": self.preprocessor is not None,
            "explainer": self.explainer is not None,
            "fairness": bool(self.fairness),
            "anova": bool(self.anova),
            "metrics": bool(self.metrics),
        }

    def load(self, config: dict[str, Any] | None = None) -> None:
        self.config = config if config is not None else load_config()
        cfg = self.config
        processed = resolve_path(cfg["preprocess"]["output_dir"])

        try:
            self.champion = champion_combination(cfg)
            model_path = resolve_path(cfg["paths"]["models"]) / (
                f"champion_{self.champion[0]}_{cfg['models']['model_version']}.pkl"
            )
            self.model = joblib.load(model_path)
            self.feature_columns = feature_columns(cfg, self.champion[1])
            self.preprocessor = Preprocessor.load(processed / cfg["preprocess"]["scaler_file"])
            logger.info("모델 로드: %s (%s)", model_path.name, "|".join(self.champion))
        except Exception as exc:                       # noqa: BLE001 - 기동은 계속한다
            self.errors["model"] = f"{type(exc).__name__}: {exc}"
            logger.error("모델 로드 실패: %s", exc)

        if self.model is not None:
            try:
                train_raw = self.preprocessor.inverse_transform_numeric(
                    pd.read_parquet(processed / "train.parquet")
                )
                self.explainer = CreditExplainer(
                    self.model, self.feature_columns, train_raw, cfg
                )
                logger.info("SHAP 설명기 초기화 완료")
            except Exception as exc:                   # noqa: BLE001
                self.errors["explainer"] = f"{type(exc).__name__}: {exc}"
                logger.error("설명기 초기화 실패: %s", exc)

        pre = cfg["api"]["precomputed"]
        self.fairness = self._read_json(processed / pre["fairness"], "fairness")
        self.anova = self._read_json(processed / pre["anova"], "anova")
        self._load_metrics(processed / pre["model_results"])

    def _read_json(self, path: Path, name: str) -> dict[str, Any]:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:                       # noqa: BLE001
            self.errors[name] = f"{type(exc).__name__}: {exc}"
            logger.warning("%s 사전 계산 결과를 읽지 못했습니다: %s", name, path)
            return {}

    def _load_metrics(self, path: Path) -> None:
        try:
            results = pd.read_csv(path)
            key = self.champion or champion_combination(self.config, results)
            row = results[
                (results["algorithm"] == key[0])
                & (results["feature_group"] == key[1])
                & (results["imbalance"] == key[2])
            ].iloc[0]
            self.metrics = {
                "auc": float(row["test_auc"]), "ks": float(row["test_ks"]),
                "precision": float(row["test_precision"]), "recall": float(row["test_recall"]),
                "f1": float(row["test_f1"]), "psi": float(row["psi_train_test"]),
            }
        except Exception as exc:                       # noqa: BLE001
            self.errors["metrics"] = f"{type(exc).__name__}: {exc}"
            logger.warning("성능 지표를 읽지 못했습니다: %s", path)


state = ServiceState()


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _require(resource: Any, name: str) -> Any:
    if not resource:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"{name} 를 사용할 수 없습니다. 먼저 학습·분석 파이프라인을 실행하세요. "
                   f"({state.errors.get(name, '사유 미상')})",
        )
    return resource


def derive_age_group(age: int, config: dict[str, Any]) -> str:
    for entry in config["simulator"]["age_group_bins"]:
        upper = entry["max"]
        if age >= entry["min"] and (upper is None or age < upper):
            return entry["label"]
    return config["simulator"]["age_group_bins"][-1]["label"]


def to_raw_frame(payload: schemas.CustomerFeatures, config: dict[str, Any]) -> pd.DataFrame:
    """요청을 전처리기가 기대하는 **원본 단위** 데이터프레임으로 바꾼다."""
    data = payload.model_dump(by_alias=True, exclude_none=False)
    data.pop("customer_id", None)

    sim = config["simulator"]
    age_group = data.get("age_group") or derive_age_group(payload.age, config)
    data["age_group"] = age_group
    data["gender"] = data.get("gender") or config["fairness"]["groups"]["gender"][0]

    # 거래 이력 개월수. 주지 않으면 '정상 범위 하한'으로 두어 씬파일러로 오분류되지 않게 한다.
    hist = sim["credit_history"]
    if data.get(hist["column"]) is None:
        data[hist["column"]] = hist["normal_range"][0]
    frame = pd.DataFrame([data])

    # is_thin_filer 는 SPEC §4 정의대로 산출한다(학습 피처는 아니지만 통과 컬럼이다).
    masked = sim["thin_filer_masked_columns"]
    by_missing = frame[masked].isna().sum(axis=1) >= sim["thin_filer_min_missing"]
    by_history = frame[hist["column"]] < hist["min_months_for_normal"]
    frame[sim["thin_filer_flag_column"]] = (by_missing | by_history).astype(int)
    frame[sim["thin_filer_type_column"]] = "none"
    return frame


def score(payload: schemas.CustomerFeatures) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    """(원본 단위 행, 전처리된 행, 부도 확률)."""
    _require(state.model, "model")
    raw = to_raw_frame(payload, state.config)
    processed = state.preprocessor.transform(raw)
    proba = float(state.model.predict_proba(processed[state.feature_columns])[:, 1][0])
    return raw, processed, proba


def grade_and_status(proba: float, config: dict[str, Any]) -> tuple[int, str, str]:
    scoring = config["scoring"]
    credit_score = int(round(scoring["score_max"] * (1 - proba)))
    grade = "E"
    for name, cutoff in sorted(scoring["grade_cutoffs"].items(), key=lambda kv: -kv[1]):
        if credit_score >= cutoff:
            grade = name
            break
    approved = proba < config["thresholds"]["approval_threshold"]
    return credit_score, ("APPROVED" if approved else "REJECTED"), grade


cfg_boot = load_config()
app = FastAPI(
    title=cfg_boot["api"]["title"],
    version=cfg_boot["api"]["version"],
    description=cfg_boot["api"]["description"],
)


@app.on_event("startup")
def startup() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    state.load()
    logger.info("서비스 기동 완료: %s", state.loaded)


@app.get("/health", response_model=schemas.HealthResponse, tags=["system"])
def health() -> schemas.HealthResponse:
    loaded = state.loaded
    return schemas.HealthResponse(
        status="healthy" if all(loaded.values()) else "degraded",
        model_version=state.model_version,
        timestamp=_now(),
        champion="|".join(state.champion) if state.champion else None,
        artifacts_loaded=loaded,
    )


@app.post("/predict", response_model=schemas.PredictResponse, tags=["scoring"])
def predict(payload: schemas.CustomerFeatures) -> schemas.PredictResponse:
    _, _, proba = score(payload)
    credit_score, approval, grade = grade_and_status(proba, state.config)
    return schemas.PredictResponse(
        customer_id=payload.customer_id,
        credit_score=credit_score,
        approval_status=approval,
        default_probability=round(proba, 6),
        risk_grade=grade,
        model_version=state.model_version,
        threshold=state.config["thresholds"]["approval_threshold"],
        timestamp=_now(),
    )


@app.post("/explain", response_model=schemas.ExplainResponse, tags=["scoring"])
def explain(payload: schemas.CustomerFeatures) -> schemas.ExplainResponse:
    _require(state.explainer, "explainer")
    raw, processed, proba = score(payload)
    _, approval, _ = grade_and_status(proba, state.config)
    age_group = raw["age_group"].iloc[0]

    reasons = state.explainer.contributing_factors(
        processed.iloc[0], age_group,
        adverse=(approval == "REJECTED"), display_row=raw.iloc[0],
    )
    values, base = state.explainer.explain(processed)
    return schemas.ExplainResponse(
        customer_id=payload.customer_id,
        approval_status=approval,
        default_probability=round(proba, 6),
        rejection_reasons=[schemas.RejectionReason(**r) for r in reasons],
        total_rejection_factors=len(reasons),
        shap_values={
            name: round(float(v), 6)
            for name, v in zip(state.feature_columns, values[0])
        },
        base_value=round(float(base), 6),
        model_version=state.model_version,
        timestamp=_now(),
    )


@app.get("/metrics", response_model=schemas.MetricsResponse, tags=["analysis"])
def metrics() -> schemas.MetricsResponse:
    _require(state.metrics, "metrics")
    return schemas.MetricsResponse(model_version=state.model_version, **state.metrics)


def _di_payload(block: dict[str, Any]) -> schemas.DisparateImpact:
    return schemas.DisparateImpact(
        attribute=block["attribute"],
        disadvantaged_group=block["disadvantaged_group"],
        approval_rates=block["approval_rates"],
        di_ratio=block["di_ratio"],
        di_ratio_min_max=block["di_ratio_min_max"],
        passes=block["passes"],
    )


@app.get("/fairness", response_model=schemas.FairnessResponse, tags=["fairness"])
def fairness() -> schemas.FairnessResponse:
    _require(state.fairness, "fairness")
    before = state.fairness["before"]
    thresholds = state.config["thresholds"]
    attributes = state.config["fairness"]["sensitive_attributes"]

    by_attribute = {
        name: schemas.FairnessAttribute(
            disparate_impact=_di_payload(before[name]["disparate_impact"]),
            equalized_odds=schemas.EqualizedOdds(**{
                k: v for k, v in before[name]["equalized_odds"].items()
                if k in schemas.EqualizedOdds.model_fields
            }),
        )
        for name in attributes
    }
    # 최상위 값은 4/5 규칙에서 가장 나쁜 속성 기준
    worst = min(attributes, key=lambda a: before[a]["disparate_impact"]["di_ratio_min_max"])
    return schemas.FairnessResponse(
        di_ratio=before[worst]["disparate_impact"]["di_ratio"],
        di_ratio_min_max=before[worst]["disparate_impact"]["di_ratio_min_max"],
        equalized_odds_tpr_diff=before[worst]["equalized_odds"]["equalized_odds_tpr_diff"],
        equalized_odds_fpr_diff=before[worst]["equalized_odds"]["equalized_odds_fpr_diff"],
        worst_attribute=worst,
        threshold=before["threshold"],
        overall_approval_rate=before["overall_approval_rate"],
        criteria={
            "di_ratio_min": thresholds["di_ratio_min"],
            "di_ratio_max": thresholds["di_ratio_max"],
            "eo_diff_max": thresholds["eo_diff_max"],
        },
        by_attribute=by_attribute,
    )


def _phase(block: dict[str, Any]) -> schemas.MitigationPhase:
    return schemas.MitigationPhase(
        di_ratio=block["disparate_impact"]["di_ratio"],
        di_ratio_min_max=block["disparate_impact"]["di_ratio_min_max"],
        equalized_odds_tpr_diff=block["equalized_odds"]["equalized_odds_tpr_diff"],
        equalized_odds_fpr_diff=block["equalized_odds"]["equalized_odds_fpr_diff"],
        approval_rate=block["approval_rate"],
        auc=block["auc"],
        accuracy=block["accuracy"],
    )


@app.get("/fairness/mitigation", response_model=schemas.MitigationResponse, tags=["fairness"])
def fairness_mitigation() -> schemas.MitigationResponse:
    _require(state.fairness, "fairness")
    mitigation = state.fairness["mitigation"]
    settings = state.config["fairness"]["mitigation"]
    return schemas.MitigationResponse(
        method=settings["method"],
        constraints=settings["constraints"],
        objective=settings.get("objective", "balanced_accuracy_score"),
        before={name: _phase(res["before"]) for name, res in mitigation.items()},
        after={name: _phase(res["after"]) for name, res in mitigation.items()},
        by_attribute={
            name: schemas.MitigationAttribute(
                before=_phase(res["before"]), after=_phase(res["after"]),
                delta=res["delta"], adjusted_thresholds=res["adjusted_thresholds"],
            )
            for name, res in mitigation.items()
        },
        note=(
            "완화 후 AUC 하락은 출력이 확률에서 이진 결정으로 바뀌어 순위 정보가 소실된 결과이며 "
            "모델 성능 저하가 아니다. 전/후 비교는 DI Ratio, Equalized Odds, 승인율로 판단한다."
        ),
    )


@app.get("/anova", response_model=schemas.AnovaResponse, tags=["analysis"])
def anova() -> schemas.AnovaResponse:
    _require(state.anova, "anova")

    def build(key: str) -> schemas.AnovaTest:
        block = state.anova[key]
        allowed = set(schemas.AnovaTest.model_fields)
        return schemas.AnovaTest(**{k: v for k, v in block.items() if k in allowed})

    return schemas.AnovaResponse(
        test_1=build("test_1_feature_group"),
        test_2=build("test_2_algorithm"),
    )
