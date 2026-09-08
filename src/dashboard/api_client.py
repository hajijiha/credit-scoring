"""대시보드 → FastAPI HTTP 클라이언트 (SPEC §7 MSA 강제).

**이 모듈과 대시보드는 모델 파일(.pkl/.joblib)을 절대 로드하지 않는다.**
모든 추론과 지표 조회는 FastAPI 서버와의 HTTP 통신으로만 수행한다.
API 주소는 환경변수 `API_BASE_URL` 에서 읽는다(Docker 네트워크 대비).
"""

from __future__ import annotations

import logging
import os
from typing import Any

import requests

from src.config import load_config

logger = logging.getLogger(__name__)


class ApiError(RuntimeError):
    """API 호출 실패. 대시보드는 이 예외를 잡아 안내 문구를 띄운다."""

    def __init__(self, message: str, *, url: str = "", detail: str = "") -> None:
        super().__init__(message)
        self.url = url
        self.detail = detail


def base_url(config: dict[str, Any] | None = None) -> str:
    """API 주소. 환경변수 > config 의 포트 기반 기본값."""
    from_env = os.getenv("API_BASE_URL")
    if from_env:
        return from_env.rstrip("/")
    cfg = config if config is not None else load_config()
    return f"http://localhost:{cfg['ports']['api']}"


class CreditScoringApi:
    """FastAPI 엔드포인트 래퍼."""

    def __init__(self, config: dict[str, Any] | None = None, url: str | None = None) -> None:
        self.config = config if config is not None else load_config()
        self.url = (url or base_url(self.config)).rstrip("/")
        self.timeout = self.config["dashboard"]["request_timeout_seconds"]

    # --- 내부 ---------------------------------------------------------------
    def _request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        endpoint = f"{self.url}{path}"
        try:
            response = requests.request(method, endpoint, timeout=self.timeout, **kwargs)
        except requests.exceptions.ConnectionError as exc:
            raise ApiError(
                f"API 서버에 연결할 수 없습니다: {endpoint}", url=endpoint, detail=str(exc)
            ) from exc
        except requests.exceptions.Timeout as exc:
            raise ApiError(
                f"API 응답이 {self.timeout}초 안에 오지 않았습니다: {endpoint}",
                url=endpoint, detail=str(exc),
            ) from exc
        except requests.exceptions.RequestException as exc:
            raise ApiError(f"API 요청 실패: {endpoint}", url=endpoint, detail=str(exc)) from exc

        if response.status_code >= 400:
            detail = ""
            try:
                detail = str(response.json().get("detail", response.text))
            except Exception:                          # noqa: BLE001
                detail = response.text[:500]
            raise ApiError(
                f"API 오류 {response.status_code}: {endpoint}", url=endpoint, detail=detail
            )
        return response.json()

    # --- 엔드포인트 ---------------------------------------------------------
    def health(self) -> dict[str, Any]:
        return self._request("GET", "/health")

    def predict(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._request("POST", "/predict", json=payload)

    def explain(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._request("POST", "/explain", json=payload)

    def metrics(self) -> dict[str, Any]:
        return self._request("GET", "/metrics")

    def fairness(self) -> dict[str, Any]:
        return self._request("GET", "/fairness")

    def fairness_mitigation(self) -> dict[str, Any]:
        return self._request("GET", "/fairness/mitigation")

    def anova(self) -> dict[str, Any]:
        return self._request("GET", "/anova")
