"""대시보드 검증 (SPEC §7).

핵심은 **MSA 제약**이다. 대시보드는 모델 파일을 직접 로드해서는 안 되고,
API 가 죽어 있어도 크래시 없이 안내를 띄워야 한다.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import requests

from src.config import load_config
from src.dashboard.api_client import ApiError, CreditScoringApi, base_url

DASHBOARD_DIR = Path(__file__).resolve().parents[1] / "src" / "dashboard"
APP_PATH = DASHBOARD_DIR / "app.py"
DEAD_API = "http://127.0.0.1:59999"      # 아무도 듣지 않는 포트


@pytest.fixture(scope="module")
def config() -> dict:
    return load_config()


# --- import 가능 여부 -------------------------------------------------------

def test_dashboard_modules_import():
    import src.dashboard as package
    from src.dashboard import api_client

    assert package.CreditScoringApi is api_client.CreditScoringApi


def test_app_module_is_valid_python():
    """app.py 는 실행 시 Streamlit 이 필요하므로 구문 파싱으로만 확인한다."""
    ast.parse(APP_PATH.read_text(encoding="utf-8"))


# --- SPEC §7: 모델 파일 직접 로드 금지 --------------------------------------

FORBIDDEN_IMPORTS = {"joblib", "pickle", "cloudpickle", "xgboost", "lightgbm", "shap", "mlflow"}
FORBIDDEN_PATTERNS = (
    r"\.pkl\b", r"\.joblib\b", r"joblib\.load", r"pickle\.load",
    r"load_champion", r"Preprocessor", r"predict_proba",
)


def _dashboard_sources() -> list[Path]:
    return sorted(DASHBOARD_DIR.glob("*.py"))


def test_dashboard_does_not_import_model_libraries():
    """모델·직렬화 라이브러리를 import 하지 않아야 한다(AST 로 검사)."""
    offenders: list[str] = []
    for path in _dashboard_sources():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name.split(".")[0] for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [(node.module or "").split(".")[0]]
            else:
                continue
            for name in names:
                if name in FORBIDDEN_IMPORTS:
                    offenders.append(f"{path.name}: import {name}")
    assert not offenders, f"대시보드가 모델 관련 모듈을 import 한다: {offenders}"


def test_dashboard_has_no_model_loading_code():
    """코드 라인(주석 제외)에 모델 파일 로드 흔적이 없어야 한다."""
    offenders: list[str] = []
    for path in _dashboard_sources():
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            stripped = line.strip()
            if stripped.startswith("#") or not stripped:
                continue
            for pattern in FORBIDDEN_PATTERNS:
                if re.search(pattern, line):
                    offenders.append(f"{path.name}:{number}: {stripped[:80]}")
    # 도큐스트링 안의 설명 문구는 제외한다.
    offenders = [o for o in offenders if "절대 로드하지 않는다" not in o and "직접 로드하지" not in o]
    assert not offenders, f"모델 로드로 보이는 코드: {offenders}"


def test_dashboard_only_talks_to_api():
    """추론 경로가 api_client 를 거치는지: app.py 는 CreditScoringApi 만 쓴다."""
    source = APP_PATH.read_text(encoding="utf-8")
    assert "CreditScoringApi" in source
    assert "requests.post" not in source, "app.py 가 API 클라이언트를 우회한다"


# --- API 주소 --------------------------------------------------------------

def test_base_url_prefers_environment_variable(config, monkeypatch):
    monkeypatch.setenv("API_BASE_URL", "http://api:8000/")
    assert base_url(config) == "http://api:8000"


def test_base_url_falls_back_to_config_port(config, monkeypatch):
    monkeypatch.delenv("API_BASE_URL", raising=False)
    assert base_url(config) == f"http://localhost:{config['ports']['api']}"


# --- API 클라이언트 (mock) --------------------------------------------------

def _mock_response(payload: dict, status: int = 200) -> Mock:
    response = Mock()
    response.status_code = status
    response.json.return_value = payload
    response.text = str(payload)
    return response


def test_client_calls_expected_endpoints(config):
    api = CreditScoringApi(config, url="http://testapi:8000")
    calls = {
        "health": ("GET", "/health"),
        "metrics": ("GET", "/metrics"),
        "fairness": ("GET", "/fairness"),
        "fairness_mitigation": ("GET", "/fairness/mitigation"),
        "anova": ("GET", "/anova"),
    }
    for method_name, (verb, path) in calls.items():
        with patch("src.dashboard.api_client.requests.request",
                   return_value=_mock_response({"ok": True})) as request:
            assert getattr(api, method_name)() == {"ok": True}
            args, kwargs = request.call_args
            assert args[0] == verb
            assert args[1] == f"http://testapi:8000{path}"
            assert kwargs["timeout"] == config["dashboard"]["request_timeout_seconds"]


def test_client_posts_payload(config):
    api = CreditScoringApi(config, url="http://testapi:8000")
    payload = {"age": 30}
    for method_name, path in (("predict", "/predict"), ("explain", "/explain")):
        with patch("src.dashboard.api_client.requests.request",
                   return_value=_mock_response({"ok": True})) as request:
            getattr(api, method_name)(payload)
            args, kwargs = request.call_args
            assert args[0] == "POST" and args[1].endswith(path)
            assert kwargs["json"] == payload


def test_client_raises_api_error_on_connection_failure(config):
    api = CreditScoringApi(config, url=DEAD_API)
    with patch("src.dashboard.api_client.requests.request",
               side_effect=requests.exceptions.ConnectionError("refused")):
        with pytest.raises(ApiError) as excinfo:
            api.health()
    assert "연결할 수 없습니다" in str(excinfo.value)
    assert excinfo.value.url.endswith("/health")


def test_client_raises_api_error_on_timeout(config):
    api = CreditScoringApi(config, url=DEAD_API)
    with patch("src.dashboard.api_client.requests.request",
               side_effect=requests.exceptions.Timeout("slow")):
        with pytest.raises(ApiError, match="오지 않았습니다"):
            api.health()


def test_client_raises_api_error_on_http_error(config):
    api = CreditScoringApi(config, url="http://testapi:8000")
    with patch("src.dashboard.api_client.requests.request",
               return_value=_mock_response({"detail": "모델 없음"}, status=503)):
        with pytest.raises(ApiError) as excinfo:
            api.predict({})
    assert "503" in str(excinfo.value)
    assert excinfo.value.detail == "모델 없음"


def test_client_really_cannot_connect_to_dead_port(config):
    """mock 없이도 실제 연결 실패가 ApiError 로 감싸지는지."""
    api = CreditScoringApi(config, url=DEAD_API)
    api.timeout = 2
    with pytest.raises(ApiError):
        api.health()


# --- 위험도 색상 -----------------------------------------------------------

def test_risk_level_colors(config):
    from src.dashboard.app import risk_level

    colors = config["dashboard"]["risk_colors"]
    assert risk_level(0.05, config) == ("저위험", colors["low"])
    assert risk_level(0.20, config) == ("중위험", colors["medium"])
    assert risk_level(0.80, config) == ("고위험", colors["high"])
    # 승인 임계값과 고위험 경계가 일치해야 한다.
    assert config["dashboard"]["risk_bands"]["medium_max"] == \
        config["thresholds"]["approval_threshold"]


# --- Streamlit AppTest -----------------------------------------------------

def _app_test(timeout: int = 180):
    from streamlit.testing.v1 import AppTest

    return AppTest.from_file(str(APP_PATH), default_timeout=timeout)


def test_app_renders_five_tabs_when_api_is_down(monkeypatch):
    """API 가 죽어 있어도 크래시 없이 안내를 띄운다."""
    monkeypatch.setenv("API_BASE_URL", DEAD_API)
    app = _app_test().run()
    assert not app.exception, f"API 미기동 시 예외 발생: {app.exception}"
    assert len(app.tabs) == 5
    messages = [e.value for e in app.error]
    assert messages, "연결 실패 안내가 없다"
    assert any("연결할 수 없습니다" in m for m in messages)


def test_app_default_input_matches_config(config):
    """폼 기본값이 config 의 borderline 샘플과 일치해야 한다."""
    defaults = config["dashboard"]["default_input"]
    assert defaults["credit_history_months"] < \
        config["simulator"]["credit_history"]["min_months_for_normal"], "씬파일러 샘플이어야 한다"
    assert all(defaults[c] is None for c in config["preprocess"]["missing_indicator_columns"])


def test_disclaimer_present():
    """SPEC §7: 면책 조항 필수."""
    from src.dashboard.app import DISCLAIMER

    assert "교육 목적" in DISCLAIMER and "사용할 수 없습니다" in DISCLAIMER
    assert DISCLAIMER in APP_PATH.read_text(encoding="utf-8")


def test_tab_names_match_spec():
    from src.dashboard.app import TAB_NAMES

    assert TAB_NAMES == ["고객 심사", "XAI 분석", "공정성", "모델 성능", "통계 검증"]
