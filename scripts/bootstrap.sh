#!/usr/bin/env bash
# 클론 직후 학습 산출물을 만든다. 컨테이너 안/밖 모두에서 동작한다.
#
#   docker compose --profile setup run --rm bootstrap     # 권장(환경 동일)
#   bash scripts/bootstrap.sh                             # 로컬 .venv 사용
#
# 이미 만들어진 산출물은 건너뛴다. 강제로 다시 만들려면 FORCE=1 을 준다.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# 컨테이너 안에서는 시스템 python, 밖에서는 프로젝트 .venv 를 쓴다.
if [ -x ".venv/bin/python" ] && [ -z "${IN_CONTAINER:-}" ] && [ ! -f "/.dockerenv" ]; then
    PY=".venv/bin/python"
else
    PY="python"
fi
export PYTHONPATH="$ROOT"
export MLFLOW_DISABLE_AGENT_HINT=1

RAW="data/raw/cs-training.csv"
GMSC_URL="https://raw.githubusercontent.com/JLZml/Credit-Scoring-Data-Sets/master/3.%20Kaggle/Give%20Me%20Some%20Credit/cs-training.csv"

step() { printf '\n\033[1m[bootstrap] %s\033[0m\n' "$1"; }
have() { [ -e "$1" ] && [ -z "${FORCE:-}" ]; }

step "1/6 데이터 확보"
if have "$RAW"; then
    echo "  이미 있음: $RAW"
else
    mkdir -p data/raw
    # Kaggle 원본은 인증이 필요해 공개 미러에서 받는다(README 참조).
    curl -fsSL -o "$RAW" "$GMSC_URL"
    echo "  내려받음: $RAW ($(wc -l < "$RAW") 줄)"
fi

step "2/6 시뮬레이션 + 전처리"
if have "data/processed/train.parquet"; then
    echo "  이미 있음: data/processed/train.parquet"
else
    $PY -m src.data.run_preprocess
fi

step "3/6 모델 학습 (18개 조합 x 20 fold, 약 11분)"
if have "models/champion_xgboost_v1.0.pkl"; then
    echo "  이미 있음: models/champion_xgboost_v1.0.pkl"
else
    $PY -m src.models.train
fi

step "4/6 씬파일러 uplift + ANOVA"
have "data/processed/thin_filer_uplift.csv" || $PY -m src.models.train --uplift-only
have "data/processed/anova_results.json"    || $PY -m src.statistics.anova > /dev/null

step "5/6 SHAP 설명 + 공정성"
have "data/processed/explain/shap_summary.png" || $PY -m src.explainer.run_explain
have "data/processed/fairness/fairness_results.json" || $PY -m src.fairness.run_fairness

step "6/6 리포트 자료"
have "docs/report_assets/feature_group_auc.png" || $PY -m src.models.report_assets

step "완료"
echo "  이제 실행하세요:  docker compose up"
