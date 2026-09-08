"""FastAPI Request / Response 스키마 (SPEC §6).

데이터셋 컬럼명(`NumberOfTime30-59DaysPastDueNotWorse` 등)은 파이썬 식별자가 될 수 없으므로
필드는 snake_case 로 두고 `alias` 로 원본 컬럼명을 붙인다. 둘 다로 입력할 수 있다.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class CustomerFeatures(BaseModel):
    """심사 요청. **원본 단위**(전처리 전)로 받는다. 스케일링은 서버가 한다."""

    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    customer_id: str | None = Field(default=None, description="고객 식별자(선택)")

    # --- 전통 신용 피처 ---
    revolving_utilization: float = Field(
        alias="RevolvingUtilizationOfUnsecuredLines", ge=0,
        description="무담보 신용한도 소진율",
    )
    age: int = Field(ge=18, le=120, description="연령(만 나이)")
    debt_ratio: float = Field(alias="DebtRatio", ge=0, description="부채비율")
    monthly_income: float | None = Field(
        default=None, alias="MonthlyIncome", ge=0,
        description="월 소득. null 이면 학습 중앙값으로 대체",
    )
    open_credit_lines: int = Field(
        alias="NumberOfOpenCreditLinesAndLoans", ge=0, description="보유 신용계좌 수",
    )
    real_estate_loans: int = Field(
        alias="NumberRealEstateLoansOrLines", ge=0, description="부동산 담보대출 건수",
    )
    dependents: float | None = Field(
        default=None, alias="NumberOfDependents", ge=0,
        description="부양가족 수. null 이면 학습 중앙값으로 대체",
    )

    # 연체 3컬럼: null 은 '기록 없음'(씬파일러). 중앙값으로 채우지 않고 결측 플래그로 보존한다.
    past_due_30_59: int | None = Field(
        default=None, alias="NumberOfTime30-59DaysPastDueNotWorse", ge=0,
        description="30-59일 연체 횟수. null = 기록 없음",
    )
    past_due_60_89: int | None = Field(
        default=None, alias="NumberOfTime60-89DaysPastDueNotWorse", ge=0,
        description="60-89일 연체 횟수. null = 기록 없음",
    )
    times_90_days_late: int | None = Field(
        default=None, alias="NumberOfTimes90DaysLate", ge=0,
        description="90일 이상 연체 횟수. null = 기록 없음",
    )

    # --- 대안 데이터 ---
    telecom_payment_rate: float = Field(ge=0, le=1, description="통신비 정상납부율 (0~1)")
    utility_payment_rate: float = Field(ge=0, le=1, description="공과금 납부율 (0~1)")
    spending_consistency: float = Field(ge=0, le=1, description="소비 일관성 점수 (0~1)")
    regular_payment_count: float = Field(ge=0, description="정기결제 건수")
    app_login_frequency: float = Field(ge=0, description="앱 접속 빈도")

    # --- 보호 속성 (학습 피처 아님. 공정성 평가용, ADR-008) ---
    gender: Literal["male", "female"] | None = Field(
        default=None, description="성별. 학습에 사용되지 않는다",
    )
    age_group: Literal["20-34", "35-54", "55+"] | None = Field(
        default=None, description="연령대. 생략하면 age 로부터 계산한다",
    )

    # --- 세그먼트 (학습 피처 아님, ADR-009/010) ---
    credit_history_months: int | None = Field(
        default=None, ge=0, description="신용카드 거래 이력 개월수",
    )


class PredictResponse(BaseModel):
    customer_id: str | None = None
    credit_score: int = Field(description="0~1000. score = 1000 x (1 - p_default)")
    approval_status: Literal["APPROVED", "REJECTED"]
    default_probability: float = Field(ge=0, le=1)
    risk_grade: Literal["A", "B", "C", "D", "E"]
    model_version: str
    threshold: float
    timestamp: datetime


class RejectionReason(BaseModel):
    rank: int
    feature: str
    feature_name_kr: str
    current_value: float
    reference_value: float
    shap_contribution: float = Field(description="신용도 기여도. 음수일수록 불리")
    shap_value_default: float = Field(description="부도 확률에 대한 원본 SHAP 값")
    explanation: str
    direction_consistent: bool


class ExplainResponse(BaseModel):
    customer_id: str | None = None
    approval_status: Literal["APPROVED", "REJECTED"]
    default_probability: float
    rejection_reasons: list[RejectionReason]
    total_rejection_factors: int
    shap_values: dict[str, float] = Field(description="피처별 SHAP 값(부도 확률 기준)")
    base_value: float
    model_version: str
    timestamp: datetime


class HealthResponse(BaseModel):
    status: Literal["healthy", "degraded"]
    model_version: str
    timestamp: datetime
    champion: str | None = None
    artifacts_loaded: dict[str, bool] = Field(default_factory=dict)


class MetricsResponse(BaseModel):
    auc: float
    ks: float
    precision: float
    recall: float
    f1: float
    psi: float
    model_version: str
    dataset: str = "test"


class DisparateImpact(BaseModel):
    attribute: str
    disadvantaged_group: str
    approval_rates: dict[str, float]
    di_ratio: float = Field(description="SPEC 정의: 보호 집단 / 나머지 집단 승인율")
    di_ratio_min_max: float = Field(description="관행 정의: 최저 집단 / 최고 집단 승인율")
    passes: bool


class EqualizedOdds(BaseModel):
    attribute: str
    tpr_by_group: dict[str, float]
    fpr_by_group: dict[str, float]
    equalized_odds_tpr_diff: float
    equalized_odds_fpr_diff: float
    passes: bool


class FairnessAttribute(BaseModel):
    disparate_impact: DisparateImpact
    equalized_odds: EqualizedOdds


class FairnessResponse(BaseModel):
    """SPEC §6 의 최소 필드(di_ratio, tpr_diff, fpr_diff)를 최상위에 두고,

    보호 속성별 상세를 `by_attribute` 에 함께 싣는다.
    최상위 값은 기준 위반이 가장 큰 속성의 값이다.
    """

    di_ratio: float
    di_ratio_min_max: float
    equalized_odds_tpr_diff: float
    equalized_odds_fpr_diff: float
    worst_attribute: str
    threshold: float
    overall_approval_rate: float
    criteria: dict[str, float]
    by_attribute: dict[str, FairnessAttribute]


class MitigationPhase(BaseModel):
    di_ratio: float
    di_ratio_min_max: float
    equalized_odds_tpr_diff: float
    equalized_odds_fpr_diff: float
    approval_rate: float
    auc: float
    accuracy: float


class MitigationAttribute(BaseModel):
    before: MitigationPhase
    after: MitigationPhase
    delta: dict[str, float]
    adjusted_thresholds: dict[str, Any]


class MitigationResponse(BaseModel):
    method: str
    constraints: str
    objective: str
    before: dict[str, MitigationPhase]
    after: dict[str, MitigationPhase]
    by_attribute: dict[str, MitigationAttribute]
    note: str


class TukeyRow(BaseModel):
    comparison: str
    mean_diff: float
    p_adj: float
    significant: bool


class AnovaTest(BaseModel):
    test_name: str
    factor: str
    control: dict[str, str]
    groups: list[str]
    group_n: dict[str, int]
    group_means: dict[str, float]
    f_statistic: float
    p_value: float
    effect_size_eta_squared: float
    effect_size_interpretation: str
    is_significant: bool
    post_hoc_tukey: list[TukeyRow] = Field(default_factory=list)
    diagnostics: dict[str, Any] = Field(default_factory=dict)
    effect_size_note: str | None = None


class AnovaResponse(BaseModel):
    test_1: AnovaTest
    test_2: AnovaTest


class ErrorResponse(BaseModel):
    detail: str
