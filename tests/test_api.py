"""FastAPI 엔드포인트 검증 (SPEC §6)."""

from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from src.api.main import app, derive_age_group, to_raw_frame
from src.api.schemas import CustomerFeatures
from src.config import load_config

BASE_PAYLOAD = {
    "customer_id": "C001234",
    "RevolvingUtilizationOfUnsecuredLines": 0.72,
    "age": 29,
    "DebtRatio": 0.41,
    "MonthlyIncome": 3200,
    "NumberOfOpenCreditLinesAndLoans": 4,
    "NumberRealEstateLoansOrLines": 0,
    "NumberOfDependents": 1,
    "NumberOfTime30-59DaysPastDueNotWorse": 2,
    "NumberOfTimes90DaysLate": 1,
    "NumberOfTime60-89DaysPastDueNotWorse": 0,
    "telecom_payment_rate": 0.62,
    "utility_payment_rate": 0.65,
    "spending_consistency": 0.30,
    "regular_payment_count": 3,
    "app_login_frequency": 8,
    "gender": "female",
    "credit_history_months": 8,
}

GOOD_PAYLOAD = {
    **BASE_PAYLOAD,
    "customer_id": "C009999",
    "RevolvingUtilizationOfUnsecuredLines": 0.05,
    "age": 58,
    "DebtRatio": 0.12,
    "MonthlyIncome": 9000,
    "NumberOfTime30-59DaysPastDueNotWorse": 0,
    "NumberOfTimes90DaysLate": 0,
    "NumberOfTime60-89DaysPastDueNotWorse": 0,
    "telecom_payment_rate": 0.99,
    "utility_payment_rate": 0.98,
    "spending_consistency": 0.92,
    "regular_payment_count": 18,
    "app_login_frequency": 27,
    "gender": "male",
    "credit_history_months": 90,
}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="module")
def config() -> dict:
    return load_config()


# --- /health --------------------------------------------------------------

def test_health(client, config):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] in {"healthy", "degraded"}
    assert body["model_version"] == config["models"]["model_version"]
    assert "timestamp" in body
    assert set(body["artifacts_loaded"]) >= {"model", "preprocessor", "explainer"}


def test_health_reports_all_artifacts_loaded(client):
    """산출물이 모두 준비된 상태라면 healthy 여야 한다."""
    body = client.get("/health").json()
    missing = [k for k, v in body["artifacts_loaded"].items() if not v]
    assert not missing, f"로드되지 않은 자원: {missing}"
    assert body["status"] == "healthy"


# --- /predict -------------------------------------------------------------

def test_predict_schema(client, config):
    response = client.post("/predict", json=BASE_PAYLOAD)
    assert response.status_code == 200
    body = response.json()
    for field in ("credit_score", "approval_status", "default_probability", "risk_grade"):
        assert field in body
    assert 0 <= body["credit_score"] <= config["scoring"]["score_max"]
    assert body["approval_status"] in {"APPROVED", "REJECTED"}
    assert 0.0 <= body["default_probability"] <= 1.0
    assert body["risk_grade"] in {"A", "B", "C", "D", "E"}
    assert body["customer_id"] == "C001234"


def test_predict_score_matches_probability(client, config):
    """ADR-003: score = 1000 x (1 - p_default)."""
    body = client.post("/predict", json=BASE_PAYLOAD).json()
    expected = round(config["scoring"]["score_max"] * (1 - body["default_probability"]))
    assert body["credit_score"] == expected


def test_predict_threshold_decides_status(client, config):
    """승인 여부는 approval_threshold 로 갈린다."""
    threshold = config["thresholds"]["approval_threshold"]
    for payload in (BASE_PAYLOAD, GOOD_PAYLOAD):
        body = client.post("/predict", json=payload).json()
        expected = "APPROVED" if body["default_probability"] < threshold else "REJECTED"
        assert body["approval_status"] == expected


def test_predict_separates_good_and_bad_customer(client):
    """우량 고객의 부도 확률이 더 낮아야 한다."""
    bad = client.post("/predict", json=BASE_PAYLOAD).json()
    good = client.post("/predict", json=GOOD_PAYLOAD).json()
    assert good["default_probability"] < bad["default_probability"]
    assert good["credit_score"] > bad["credit_score"]


def test_predict_response_time_under_two_seconds(client):
    """SPEC §6: 단일 요청 2초 이내."""
    durations = []
    for _ in range(10):
        started = time.perf_counter()
        response = client.post("/predict", json=BASE_PAYLOAD)
        durations.append(time.perf_counter() - started)
        assert response.status_code == 200
    assert max(durations) < 2.0, f"최대 응답 {max(durations):.3f}초"


def test_predict_accepts_missing_delinquency(client):
    """연체 컬럼 null(기록 없음, 씬파일러)도 처리돼야 한다."""
    payload = {k: v for k, v in BASE_PAYLOAD.items()
               if not k.startswith(("NumberOfTime", "NumberOfTimes90"))}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200


def test_predict_rejects_invalid_input(client):
    """잘못된 입력은 422."""
    cases = [
        {**BASE_PAYLOAD, "age": 5},                              # 하한 위반
        {**BASE_PAYLOAD, "telecom_payment_rate": 1.5},           # 0~1 범위 위반
        {**BASE_PAYLOAD, "gender": "unknown"},                   # Literal 위반
        {**BASE_PAYLOAD, "RevolvingUtilizationOfUnsecuredLines": -1},
        {k: v for k, v in BASE_PAYLOAD.items() if k != "age"},   # 필수 필드 누락
        {**BASE_PAYLOAD, "정체불명필드": 1},                        # extra="forbid"
    ]
    for payload in cases:
        assert client.post("/predict", json=payload).status_code == 422, payload


# --- /explain -------------------------------------------------------------

def test_explain_schema(client, config):
    response = client.post("/explain", json=BASE_PAYLOAD)
    assert response.status_code == 200
    body = response.json()
    assert body["total_rejection_factors"] == config["explainer"]["top_rejection_reasons"] == 5
    assert len(body["rejection_reasons"]) == 5
    for reason in body["rejection_reasons"]:
        for field in ("feature", "current_value", "shap_contribution", "explanation"):
            assert field in reason
        assert "기여도" in reason["explanation"]
    assert body["shap_values"] and isinstance(body["shap_values"], dict)
    assert "base_value" in body


def test_explain_accepts_missing_delinquency(client, config):
    """연체 '기록 없음'(null) 입력에서 /explain 이 500 없이 '기록 없음' 문장을 내야 한다.

    회귀 방지: 초기 구현은 float(None) 에서 500 을 냈다. 씬파일러가 그대로 걸리는 경로다.
    대시보드 기본 입력(경계 거절 사례, 씬파일러 both 유형)은 연체 3컬럼이 null 이고
    그중 30-59일 연체가 거절 사유 Top 5 에 든다.
    """
    payload = dict(config["dashboard"]["default_input"])
    response = client.post("/explain", json=payload)
    assert response.status_code == 200
    texts = [r["explanation"] for r in response.json()["rejection_reasons"]]
    assert any("기록이 없습니다" in t for t in texts)
    assert not any("-1회" in t for t in texts)


def test_explain_shap_values_cover_model_features(client):
    from src.api.main import state
    body = client.post("/explain", json=BASE_PAYLOAD).json()
    assert set(body["shap_values"]) == set(state.feature_columns)


def test_explain_matches_predict_probability(client):
    """같은 입력이면 /predict 와 /explain 의 부도 확률이 같아야 한다."""
    predicted = client.post("/predict", json=BASE_PAYLOAD).json()["default_probability"]
    explained = client.post("/explain", json=BASE_PAYLOAD).json()["default_probability"]
    assert predicted == pytest.approx(explained)


def test_explain_rejects_invalid_input(client):
    assert client.post("/explain", json={**BASE_PAYLOAD, "age": 200}).status_code == 422


# --- /metrics -------------------------------------------------------------

def test_metrics(client):
    response = client.get("/metrics")
    assert response.status_code == 200
    body = response.json()
    for field in ("auc", "ks", "precision", "recall", "f1"):
        assert field in body and 0.0 <= body[field] <= 1.0
    assert body["auc"] >= 0.78, "SPEC §2 목표 미달"
    assert body["ks"] >= 0.28
    assert body["psi"] < 0.1


# --- /fairness ------------------------------------------------------------

def test_fairness(client, config):
    response = client.get("/fairness")
    assert response.status_code == 200
    body = response.json()
    for field in ("di_ratio", "equalized_odds_tpr_diff", "equalized_odds_fpr_diff"):
        assert field in body
    assert "di_ratio_min_max" in body, "양 정의를 모두 반환해야 한다"
    assert set(body["by_attribute"]) == set(config["fairness"]["sensitive_attributes"])
    for attribute in body["by_attribute"].values():
        di = attribute["disparate_impact"]
        assert "di_ratio" in di and "di_ratio_min_max" in di
        assert di["approval_rates"]
        eo = attribute["equalized_odds"]
        assert eo["tpr_by_group"] and eo["fpr_by_group"]


def test_fairness_worst_attribute_is_the_lowest_min_max(client):
    body = client.get("/fairness").json()
    ratios = {
        name: attribute["disparate_impact"]["di_ratio_min_max"]
        for name, attribute in body["by_attribute"].items()
    }
    assert body["worst_attribute"] == min(ratios, key=ratios.get)


def test_fairness_mitigation(client, config):
    response = client.get("/fairness/mitigation")
    assert response.status_code == 200
    body = response.json()
    assert "before" in body and "after" in body
    assert body["constraints"] == config["fairness"]["mitigation"]["constraints"]
    assert body["objective"] == config["fairness"]["mitigation"]["objective"]
    for attribute, block in body["by_attribute"].items():
        before, after = block["before"], block["after"]
        # 완화 후 Equalized Odds 최대 차이가 줄어야 한다.
        assert max(after["equalized_odds_tpr_diff"], after["equalized_odds_fpr_diff"]) < max(
            before["equalized_odds_tpr_diff"], before["equalized_odds_fpr_diff"]
        ), attribute
        assert block["adjusted_thresholds"]


# --- /anova ---------------------------------------------------------------

def test_anova(client):
    response = client.get("/anova")
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"test_1", "test_2"}
    for test in body.values():
        for field in ("f_statistic", "p_value", "effect_size_eta_squared"):
            assert field in test
        assert test["is_significant"] is True
        assert test["post_hoc_tukey"], "p<0.05 이면 Tukey HSD 가 있어야 한다"
        assert set(test["group_n"].values()) == {20}, "ADR-011: 그룹당 n=20"


# --- 내부 헬퍼 -------------------------------------------------------------

@pytest.mark.parametrize(
    "age,expected", [(20, "20-34"), (34, "20-34"), (35, "35-54"), (54, "35-54"), (55, "55+"), (90, "55+")]
)
def test_derive_age_group(config, age, expected):
    assert derive_age_group(age, config) == expected


def test_to_raw_frame_derives_age_group_and_thin_filer(config):
    payload = CustomerFeatures(**{k: v for k, v in BASE_PAYLOAD.items()
                                  if k not in ("gender", "credit_history_months")})
    frame = to_raw_frame(payload, config)
    assert frame["age_group"].iloc[0] == "20-34"
    # 이력 개월수를 안 주면 정상 범위 하한으로 채우므로 씬파일러가 아니다.
    assert frame[config["simulator"]["thin_filer_flag_column"]].iloc[0] == 0


def test_to_raw_frame_flags_thin_filer_by_history(config):
    payload = CustomerFeatures(**{**BASE_PAYLOAD, "credit_history_months": 6})
    frame = to_raw_frame(payload, config)
    assert frame[config["simulator"]["thin_filer_flag_column"]].iloc[0] == 1


def test_openapi_documents_all_endpoints(client):
    """Swagger UI 가 참조하는 OpenAPI 문서에 7개 엔드포인트가 있어야 한다."""
    schema = client.get("/openapi.json").json()
    expected = {
        ("/health", "get"), ("/predict", "post"), ("/explain", "post"),
        ("/fairness", "get"), ("/fairness/mitigation", "get"),
        ("/metrics", "get"), ("/anova", "get"),
    }
    actual = {(path, method) for path, item in schema["paths"].items() for method in item}
    assert expected <= actual
    assert len(expected) == 7


def test_swagger_ui_served(client):
    response = client.get("/docs")
    assert response.status_code == 200
    assert "swagger" in response.text.lower()
