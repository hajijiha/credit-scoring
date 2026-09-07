# CLAUDE.md

대안 데이터(통신·공과금 납부 이력 등)를 결합해 씬파일러의 신용위험을 평가하고, 예측의 설명가능성과
집단 공정성까지 함께 검증하는 교육용 ML 시스템.

## 세션 시작 시 반드시 할 일

1. **`PROGRESS.md` 를 먼저 읽는다.** 현재 상태·다음 할 일·미해결 이슈가 거기에 있다.
2. 관련된 설계 결정은 `docs/decisions/ADR-*.md` 에서 확인한다.
3. 작업이 하나 끝날 때마다 **즉시** `PROGRESS.md` 를 갱신한다. 세션 마지막에 몰아 쓰지 않는다.

## 디렉터리 구조

```
credit-scoring/
├── CLAUDE.md              # 이 파일 (작업 규칙)
├── PROGRESS.md            # 작업 로그 — 세션 시작 시 필독
├── README.md
├── docker-compose.yml     # api / dashboard / mlflow (현재 스켈레톤)
├── Dockerfile             # python:3.12-slim 기반
├── requirements.txt       # 버전 고정. 확정 스냅샷은 requirements.lock
├── .env.example
├── config/config.yaml     # 모든 설정값의 단일 출처
├── data/raw/              # cs-training.csv (GMSC, git 제외)
├── src/
│   ├── data/              # 로더, 대안 데이터 시뮬레이터, 전처리
│   ├── models/            # 학습 파이프라인
│   ├── explainer/         # SHAP
│   ├── fairness/          # Fairlearn 지표 · 완화
│   ├── statistics/        # 반복 CV, ANOVA, 사후검정
│   ├── api/               # FastAPI
│   └── dashboard/         # Streamlit
├── models/                # 학습 산출물 (git 제외)
├── notebooks/
├── tests/
└── docs/decisions/        # ADR
```

## 작업 규칙

- **설정값은 `config/config.yaml` 에서만 읽는다.** 임계값·비율·포트·random_state 를 코드에 하드코딩하지 않는다.
- **절대경로 금지.** 모든 경로는 프로젝트 루트 기준 상대경로다.
- **공정성은 Fairlearn 만 사용한다. AIF360 은 도입 금지** (Python 3.12 비호환).
- **대시보드에서 `.pkl` / `.joblib` 를 직접 로드하지 않는다.** 모든 추론은 FastAPI HTTP 엔드포인트를 경유한다.
- **테스트 세트의 라벨을 학습·튜닝·임계값 선택에 사용하지 않는다.** 임계값 조정은 valid 세트로 한다.
- 커밋 메시지는 `type: 설명` 형식 (`feat:`, `fix:`, `chore:`, `docs:`, `test:`, `refactor:`).
- **새로운 설계 결정이 생기면 먼저 사용자에게 묻는다.** 확정되면 `docs/decisions/ADR-XXX-*.md` 로 기록한다.

## 설계 결정 (상세는 각 ADR 참조)

| ADR | 주제 |
|---|---|
| [ADR-001](docs/decisions/ADR-001-simulator.md) | 대안 데이터 시뮬레이터 · 씬파일러 정보량 축소 |
| [ADR-002](docs/decisions/ADR-002-bias-injection.md) | bias_ratio 를 통한 피처 측정 편향 주입 |
| [ADR-003](docs/decisions/ADR-003-score-mapping.md) | 점수 매핑(1000×(1−p)) · 등급 컷 · 승인 임계값 |
| [ADR-004](docs/decisions/ADR-004-mitigation.md) | Fairlearn ThresholdOptimizer (equalized_odds) |
| [ADR-005](docs/decisions/ADR-005-anova-unit.md) | ANOVA 관측 단위 = RepeatedStratifiedKFold 5×4 |
| [ADR-006](docs/decisions/ADR-006-main-dataset.md) | 메인 데이터셋 = GMSC |

## 환경

- Python 3.12.3 / venv 는 프로젝트 루트의 `.venv` (다른 프로젝트 venv 재사용 금지)
- 활성화: `source .venv/bin/activate`
