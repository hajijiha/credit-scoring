"""심사역용 Streamlit 대시보드 (SPEC §7).

5개 탭: 고객 심사 / XAI 분석 / 공정성 / 모델 성능 / 통계 검증

**제약 (SPEC §7)**
- 모델 파일(.pkl / .joblib)을 직접 로드하지 않는다. 이 파일에는 joblib·pickle import 가 없다.
- 모든 추론과 지표 조회는 FastAPI 서버와 HTTP 통신으로 수행한다.
- API 주소는 환경변수 `API_BASE_URL` 에서 읽는다.

실행:
    streamlit run src/dashboard/app.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.config import load_config, resolve_path
from src.dashboard.api_client import ApiError, CreditScoringApi, base_url

DISCLAIMER = "본 시스템은 교육 목적으로 개발되었으며, 실제 금융 의사결정에 사용할 수 없습니다."

TAB_NAMES = ["고객 심사", "XAI 분석", "공정성", "모델 성능", "통계 검증"]

DELINQUENCY_FIELDS = {
    "NumberOfTime30-59DaysPastDueNotWorse": "30-59일 연체 횟수",
    "NumberOfTime60-89DaysPastDueNotWorse": "60-89일 연체 횟수",
    "NumberOfTimes90DaysLate": "최근 90일 연체 횟수",
}
ALTERNATIVE_FIELDS = {
    "telecom_payment_rate": "통신비 정상납부율",
    "utility_payment_rate": "공과금 납부율",
    "spending_consistency": "소비 일관성 점수",
}
COUNT_FIELDS = {
    "regular_payment_count": "정기결제 건수",
    "app_login_frequency": "앱 접속 빈도(월)",
}


# --- 공통 ------------------------------------------------------------------

def get_config() -> dict[str, Any]:
    return load_config()


def get_api(config: dict[str, Any]) -> CreditScoringApi:
    return CreditScoringApi(config)


def risk_level(probability: float, config: dict[str, Any]) -> tuple[str, str]:
    """(등급 라벨, 색상). SPEC §7 위험도별 색상 구분."""
    bands = config["dashboard"]["risk_bands"]
    colors = config["dashboard"]["risk_colors"]
    if probability < bands["low_max"]:
        return "저위험", colors["low"]
    if probability < bands["medium_max"]:
        return "중위험", colors["medium"]
    return "고위험", colors["high"]


def show_api_error(error: ApiError, config: dict[str, Any]) -> None:
    """API 연결 실패 시 크래시 대신 안내를 띄운다."""
    st.error(f"⚠️ {error}")
    with st.expander("해결 방법", expanded=True):
        st.markdown(
            f"""
현재 API 주소: `{base_url(config)}`  (환경변수 `API_BASE_URL` 로 바꿀 수 있습니다)

**API 서버를 먼저 띄우세요.**

```bash
# 로컬 실행
uvicorn src.api.main:app --host 0.0.0.0 --port {config['ports']['api']}

# 또는 통합 실행
docker-compose up
```

서버가 떠 있는데도 실패한다면 `{base_url(config)}/health` 를 직접 열어 상태를 확인하세요.
"""
        )
        if error.detail:
            st.caption(f"상세: {error.detail[:300]}")


def static_asset(config: dict[str, Any], name: str) -> Path | None:
    path = resolve_path(config["dashboard"]["static_assets_dir"]) / name
    return path if path.exists() else None


def show_asset(config: dict[str, Any], name: str, caption: str) -> None:
    path = static_asset(config, name)
    if path is None:
        st.info(f"이미지가 아직 생성되지 않았습니다: `{name}`\n\n"
                "`python -m src.models.report_assets`, `python -m src.explainer.run_explain` 실행 후 표시됩니다.")
        return
    st.image(str(path), caption=caption, width="stretch")


# --- 입력 폼 ---------------------------------------------------------------

def customer_form(config: dict[str, Any]) -> dict[str, Any] | None:
    """심사 입력 폼. 기본값은 config 의 borderline 샘플."""
    defaults = config["dashboard"]["default_input"]
    groups = config["fairness"]["groups"]

    with st.form("customer"):
        st.caption("기본값은 경계 사례(부도 확률 0.375, 씬파일러) 입니다. XAI 설명이 잘 드러납니다.")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**기본 정보**")
            age = st.number_input("연령", 18, 120, int(defaults["age"]))
            gender = st.selectbox("성별 (학습에 사용되지 않음)", groups["gender"],
                                  index=groups["gender"].index(defaults["gender"]))
            income = st.number_input("월 소득", 0.0, 1e7, float(defaults["MonthlyIncome"]), step=100.0)
            dependents = st.number_input("부양가족 수", 0, 20, int(defaults["NumberOfDependents"]))
        with col2:
            st.markdown("**전통 신용 정보**")
            revolving = st.number_input("무담보 신용한도 소진율", 0.0, 50.0,
                                        float(defaults["RevolvingUtilizationOfUnsecuredLines"]),
                                        step=0.01, format="%.4f")
            debt_ratio = st.number_input("부채비율", 0.0, 1e5, float(defaults["DebtRatio"]),
                                         step=0.01, format="%.4f")
            open_lines = st.number_input("보유 신용계좌 수", 0, 100,
                                         int(defaults["NumberOfOpenCreditLinesAndLoans"]))
            real_estate = st.number_input("부동산 담보대출 건수", 0, 50,
                                          int(defaults["NumberRealEstateLoansOrLines"]))
            history_months = st.number_input(
                "신용카드 거래 이력(개월)", 0, 600, int(defaults["credit_history_months"]),
                help="12개월 미만이면 씬파일러로 분류됩니다 (ADR-009)",
            )
        with col3:
            st.markdown("**연체 이력**")
            st.caption("'기록 없음'은 씬파일러 신호입니다. 중앙값으로 대체하지 않습니다.")
            delinquency: dict[str, Any] = {}
            for field, label in DELINQUENCY_FIELDS.items():
                unknown = st.checkbox(f"{label} 기록 없음", value=defaults[field] is None,
                                      key=f"unknown_{field}")
                value = st.number_input(label, 0, 30, int(defaults[field] or 0),
                                        key=f"value_{field}", disabled=unknown)
                delinquency[field] = None if unknown else int(value)

        st.markdown("**대안 데이터**")
        alt_cols = st.columns(5)
        alternative: dict[str, Any] = {}
        for col, (field, label) in zip(alt_cols, ALTERNATIVE_FIELDS.items()):
            with col:
                alternative[field] = st.slider(label, 0.0, 1.0, float(defaults[field]), 0.01)
        for col, (field, label) in zip(alt_cols[3:], COUNT_FIELDS.items()):
            with col:
                alternative[field] = st.number_input(label, 0, 200, int(defaults[field]),
                                                     key=f"count_{field}")

        submitted = st.form_submit_button("심사 실행", type="primary", width="stretch")

    if not submitted:
        return None
    return {
        "customer_id": "DASHBOARD-001",
        "RevolvingUtilizationOfUnsecuredLines": revolving,
        "age": int(age),
        "DebtRatio": debt_ratio,
        "MonthlyIncome": income,
        "NumberOfOpenCreditLinesAndLoans": int(open_lines),
        "NumberRealEstateLoansOrLines": int(real_estate),
        "NumberOfDependents": float(dependents),
        "gender": gender,
        "credit_history_months": int(history_months),
        **delinquency,
        **alternative,
    }


# --- 탭 1: 고객 심사 --------------------------------------------------------

def render_decision(result: dict[str, Any], config: dict[str, Any]) -> None:
    probability = result["default_probability"]
    label, color = risk_level(probability, config)
    approved = result["approval_status"] == "APPROVED"

    st.markdown(
        f"""
<div style="background:{color};color:white;padding:18px 24px;border-radius:10px;margin-bottom:12px">
  <span style="font-size:26px;font-weight:700">{'승인' if approved else '거절'}
  ({result['approval_status']})</span>
  <span style="font-size:18px;margin-left:16px">· {label}</span>
</div>
""",
        unsafe_allow_html=True,
    )
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("신용 점수", f"{result['credit_score']} / 1000")
    col2.metric("리스크 등급", result["risk_grade"])
    col3.metric("부도 확률", f"{probability:.2%}")
    col4.metric("승인 임계값", f"{result['threshold']:.2f}")
    st.caption(f"모델 버전 {result['model_version']} · 응답 시각 {result['timestamp']}")


def tab_screening(api: CreditScoringApi, config: dict[str, Any]) -> None:
    st.subheader("고객 심사")
    payload = customer_form(config)
    if payload is not None:
        st.session_state["payload"] = payload
        try:
            st.session_state["predict"] = api.predict(payload)
            st.session_state["explain"] = api.explain(payload)
        except ApiError as error:
            st.session_state.pop("predict", None)
            show_api_error(error, config)
            return

    result = st.session_state.get("predict")
    if result is None:
        st.info("좌측 폼에 고객 정보를 입력하고 **심사 실행** 을 누르세요.")
        return
    render_decision(result, config)


# --- 탭 2: XAI 분석 ---------------------------------------------------------

def waterfall_figure(explanation: dict[str, Any], config: dict[str, Any], top_n: int = 12):
    """API 가 준 SHAP 값으로 워터폴을 그린다(모델 로드 없음)."""
    values = pd.Series(explanation["shap_values"]).sort_values(key=abs, ascending=False).head(top_n)
    values = values.sort_values()
    colors_cfg = config["dashboard"]["risk_colors"]
    colors = [colors_cfg["high"] if v > 0 else colors_cfg["low"] for v in values]

    fig, ax = plt.subplots(figsize=(9, 0.42 * len(values) + 1.6))
    ax.barh(values.index, values.to_numpy(), color=colors)
    ax.axvline(0, color="#333333", lw=1)
    # matplotlib 라벨은 영문으로 둔다. 컨테이너에 한글 폰트가 없으면 글리프가 깨진다
    # (Dockerfile 에 fonts-nanum 을 넣어두었으나 로컬 실행 환경까지 보장할 수는 없다).
    ax.set_xlabel("SHAP value (toward default; positive = adverse)")
    ax.set_title(f"Individual contributions  ·  base value {explanation['base_value']:.3f}")
    for name, value in values.items():
        ax.text(value, name, f" {value:+.2f}", va="center",
                ha="left" if value > 0 else "right", fontsize=9)
    fig.tight_layout()
    return fig


def tab_xai(api: CreditScoringApi, config: dict[str, Any]) -> None:
    st.subheader("XAI 분석")
    st.markdown("#### 전역 해석 · SHAP Summary Plot")
    show_asset(config, "shap_summary.png", "전체 test 표본 기준 상위 15개 변수")

    explanation = st.session_state.get("explain")
    if explanation is None:
        st.info("**고객 심사** 탭에서 심사를 먼저 실행하면 개별 고객 설명이 표시됩니다.")
        return

    st.markdown("#### 지역 해석 · 개별 고객 Waterfall")
    st.pyplot(waterfall_figure(explanation, config))

    st.markdown("#### 거절 사유 Top 5")
    if explanation["approval_status"] == "APPROVED":
        st.success("승인 건입니다. 아래는 승인에 기여한 요인입니다.")
    for reason in explanation["rejection_reasons"]:
        color = config["dashboard"]["risk_colors"][
            "high" if reason["shap_contribution"] < 0 else "low"
        ]
        st.markdown(
            f"""
<div style="border-left:6px solid {color};padding:8px 14px;margin-bottom:8px;background:#00000008">
  <b>{reason['rank']}. {reason['feature_name_kr']}</b>
  <span style="color:{color};font-weight:600;margin-left:8px">
    기여도 {reason['shap_contribution']:+.2f}</span><br/>
  {reason['explanation']}
</div>
""",
            unsafe_allow_html=True,
        )
    st.caption("기여도는 신용도 관점입니다(음수 = 불리). 원본 SHAP 값은 부도 확률 기준입니다.")


# --- 탭 3: 공정성 -----------------------------------------------------------

def approval_rate_figure(fairness: dict[str, Any], config: dict[str, Any]):
    attributes = list(fairness["by_attribute"])
    fig, axes = plt.subplots(1, len(attributes), figsize=(6.5 * len(attributes), 4.5))
    axes = [axes] if len(attributes) == 1 else list(axes)
    colors = config["dashboard"]["risk_colors"]
    for ax, name in zip(axes, attributes):
        di = fairness["by_attribute"][name]["disparate_impact"]
        rates = pd.Series(di["approval_rates"]).sort_index()
        bar_colors = [colors["high"] if g == di["disadvantaged_group"] else colors["low"]
                      for g in rates.index]
        ax.bar(rates.index, rates.to_numpy(), color=bar_colors)
        for i, value in enumerate(rates.to_numpy()):
            ax.text(i, value + 0.01, f"{value:.3f}", ha="center", fontsize=10)
        ax.set_ylim(0, min(1.05, rates.max() * 1.25))
        ax.set_title(f"{name}  (DI {di['di_ratio']:.3f} / min-max {di['di_ratio_min_max']:.3f})")
        ax.set_ylabel("Approval rate")
    fig.tight_layout()
    return fig


def mitigation_figure(mitigation: dict[str, Any]):
    rows = []
    for name, block in mitigation["by_attribute"].items():
        for phase in ("before", "after"):
            rows.append({
                "attribute": name, "phase": phase,
                "DI ratio": block[phase]["di_ratio"],
                "EO TPR diff": block[phase]["equalized_odds_tpr_diff"],
                "EO FPR diff": block[phase]["equalized_odds_fpr_diff"],
                "Approval rate": block[phase]["approval_rate"],
            })
    data = pd.DataFrame(rows)
    metrics = ["DI ratio", "EO TPR diff", "EO FPR diff", "Approval rate"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.6 * len(metrics), 4.5))
    width = 0.35
    for ax, metric in zip(axes, metrics):
        for offset, phase, color in ((-width / 2, "before", "#4C72B0"), (width / 2, "after", "#55A868")):
            block = data[data["phase"] == phase]
            positions = range(len(block))
            ax.bar([p + offset for p in positions], block[metric].to_numpy(),
                   width=width, label=phase, color=color)
        ax.set_xticks(range(len(data["attribute"].unique())))
        ax.set_xticklabels(data["attribute"].unique(), fontsize=9)
        ax.set_title(metric, fontsize=11)
        ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def tab_fairness(api: CreditScoringApi, config: dict[str, Any]) -> None:
    st.subheader("공정성")
    try:
        fairness = api.fairness()
        mitigation = api.fairness_mitigation()
    except ApiError as error:
        show_api_error(error, config)
        return

    criteria = fairness["criteria"]
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("전체 승인율", f"{fairness['overall_approval_rate']:.4f}")
    col2.metric(f"DI Ratio ({fairness['worst_attribute']})", f"{fairness['di_ratio']:.4f}",
                help="SPEC 정의: 보호 집단 승인율 / 나머지 집단 승인율")
    col3.metric("DI Ratio (min/max)", f"{fairness['di_ratio_min_max']:.4f}",
                delta=f"{fairness['di_ratio_min_max'] - criteria['di_ratio_min']:+.4f} vs 0.8",
                help="관행 정의: 최저 집단 / 최고 집단 승인율")
    col4.metric("EO FPR 차이", f"{fairness['equalized_odds_fpr_diff']:.4f}",
                delta=f"{criteria['eo_diff_max'] - fairness['equalized_odds_fpr_diff']:+.4f} vs 0.10")

    st.markdown("#### 집단별 승인율")
    st.pyplot(approval_rate_figure(fairness, config))

    st.markdown("#### 보호 속성별 지표")
    rows = []
    for name, block in fairness["by_attribute"].items():
        di, eo = block["disparate_impact"], block["equalized_odds"]
        rows.append({
            "보호 속성": name, "불이익 집단": di["disadvantaged_group"],
            "DI (SPEC 정의)": round(di["di_ratio"], 4),
            "DI (min/max)": round(di["di_ratio_min_max"], 4),
            "DI 충족": "O" if di["passes"] else "X",
            "EO TPR차": round(eo["equalized_odds_tpr_diff"], 4),
            "EO FPR차": round(eo["equalized_odds_fpr_diff"], 4),
            "EO 충족": "O" if eo["passes"] else "X",
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

    st.markdown("#### Mitigation 전 / 후")
    st.caption(f"{mitigation['method']} · constraints={mitigation['constraints']} "
               f"· objective={mitigation['objective']}")
    st.pyplot(mitigation_figure(mitigation))
    st.info(mitigation["note"])


# --- 탭 4: 모델 성능 --------------------------------------------------------

def tab_performance(api: CreditScoringApi, config: dict[str, Any]) -> None:
    st.subheader("모델 성능")
    try:
        metrics = api.metrics()
    except ApiError as error:
        show_api_error(error, config)
        return

    cols = st.columns(6)
    targets = {"auc": 0.78, "ks": 0.28}
    for col, (name, value) in zip(cols, metrics.items()):
        if name in ("model_version", "dataset"):
            continue
        delta = None
        if name in targets:
            delta = f"{value - targets[name]:+.4f} vs 목표 {targets[name]}"
        col.metric(name.upper(), f"{value:.4f}", delta=delta)
    st.caption(f"모델 {metrics['model_version']} · {metrics['dataset']} set 기준")

    st.markdown("#### ROC Curve / Precision-Recall Curve")
    show_asset(config, "roc_pr_curves.png", "피처 그룹별 비교")
    st.markdown("#### 혼동행렬")
    show_asset(config, "confusion_matrices.png",
               f"승인 임계값 {config['thresholds']['approval_threshold']}")


# --- 탭 5: 통계 검증 --------------------------------------------------------

def tab_statistics(api: CreditScoringApi, config: dict[str, Any]) -> None:
    st.subheader("통계 검증 (ANOVA)")
    try:
        anova = api.anova()
    except ApiError as error:
        show_api_error(error, config)
        return

    summary = []
    for key, test in anova.items():
        summary.append({
            "검증": key, "비교 대상": test["factor"],
            "F 통계량": round(test["f_statistic"], 4),
            "p-value": f"{test['p_value']:.3e}",
            "η²": round(test["effect_size_eta_squared"], 4),
            "효과크기": test["effect_size_interpretation"],
            "유의": "O" if test["is_significant"] else "X",
        })
    st.dataframe(pd.DataFrame(summary), hide_index=True, width="stretch")

    for key, test in anova.items():
        st.markdown(f"#### {test['test_name']}")
        control = ", ".join(f"{k}={v}" for k, v in test["control"].items()) or "없음"
        st.caption(f"통제(ADR-011): {control} · 그룹당 n={list(test['group_n'].values())[0]}")
        means = pd.DataFrame({
            "집단": list(test["group_means"]),
            "n": [test["group_n"][g] for g in test["group_means"]],
            "평균 AUC": [round(v, 4) for v in test["group_means"].values()],
        })
        st.dataframe(means, hide_index=True, width="stretch")

        if test["post_hoc_tukey"]:
            st.markdown("**사후검정 (Tukey HSD)**")
            tukey = pd.DataFrame(test["post_hoc_tukey"])
            tukey["significant"] = tukey["significant"].map({True: "O", False: "X"})
            tukey.columns = ["비교", "평균 차이", "p_adj", "유의"]
            st.dataframe(tukey, hide_index=True, width="stretch")
        if test.get("effect_size_note"):
            st.warning(test["effect_size_note"])


# --- 진입점 ----------------------------------------------------------------

def render_sidebar(api: CreditScoringApi, config: dict[str, Any]) -> None:
    with st.sidebar:
        st.markdown("### 시스템 상태")
        st.caption(f"API: `{api.url}`")
        try:
            health = api.health()
        except ApiError as error:
            st.error("API 연결 실패")
            st.caption(str(error))
            return
        icon = "🟢" if health["status"] == "healthy" else "🟡"
        st.markdown(f"{icon} **{health['status']}**")
        st.caption(f"모델 {health['model_version']}")
        if health.get("champion"):
            st.caption(f"챔피언: `{health['champion']}`")
        missing = [k for k, v in health["artifacts_loaded"].items() if not v]
        if missing:
            st.warning(f"미로드 자원: {', '.join(missing)}")


def main() -> None:
    config = get_config()
    st.set_page_config(page_title="대안신용평가 대시보드", page_icon="📊", layout="wide")
    st.title("대안신용평가 심사 대시보드")
    st.caption(f"⚠️ {DISCLAIMER}")

    api = get_api(config)
    render_sidebar(api, config)

    tabs = st.tabs(TAB_NAMES)
    renderers = (tab_screening, tab_xai, tab_fairness, tab_performance, tab_statistics)
    for tab, renderer in zip(tabs, renderers):
        with tab:
            renderer(api, config)


# Streamlit 은 스크립트를 __main__ 으로 실행한다. 가드가 없으면 테스트에서 이 모듈을
# import 하는 것만으로 앱이 렌더링되어 Streamlit 전역 상태가 오염된다.
if __name__ == "__main__":
    main()
