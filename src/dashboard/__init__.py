"""심사역용 Streamlit 대시보드 (SPEC §7).

모델 파일을 직접 로드하지 않는다. 모든 추론은 FastAPI 를 경유한다.
"""

from src.dashboard.api_client import ApiError, CreditScoringApi, base_url

__all__ = ["CreditScoringApi", "ApiError", "base_url"]
