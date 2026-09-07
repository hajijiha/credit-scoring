# PROGRESS

> 세션 시작 시 이 파일을 먼저 읽는다. 작업이 하나 끝날 때마다 즉시 갱신한다(마지막에 몰아쓰지 않는다).

## 1. 현재 상태

기능 요구사항 8개 기준 진행도:

- [ ] **F1. 데이터 파이프라인 + 대안 데이터 시뮬레이터** — GMSC 로더, 대안 5변수 합성, 씬파일러 처리, 편향 주입 (ADR-001/002/006)
- [ ] **F2. 모델 학습 · 비교** — 전통 피처 vs 전통+대안 비교, LR/RF/XGBoost/LightGBM, 불균형 처리, MLflow 기록
- [ ] **F3. 신용점수 · 등급 · 승인 판정** — score = 1000×(1−p), A/B/C/D/E, 임계 0.30 (ADR-003)
- [ ] **F4. 설명가능성(XAI)** — SHAP 전역/개별 설명
- [ ] **F5. 공정성 측정** — Fairlearn, DI ratio 0.8~1.25, EO diff ≤ 0.10
- [ ] **F6. 공정성 완화** — ThresholdOptimizer(equalized_odds), 전/후 비교 (ADR-004)
- [ ] **F7. 통계 검정** — RepeatedStratifiedKFold 5×4, ANOVA + 사후검정 (ADR-005)
- [ ] **F8. 서빙 · 대시보드** — FastAPI(8000) + Streamlit(8501), 대시보드는 HTTP 경유만

기반 작업:

- [x] 프로젝트 스켈레톤, git 저장소, `.venv` (Python 3.12.3)
- [x] `config/config.yaml` — 모든 설정값 단일 출처
- [x] ADR-001 ~ ADR-006 작성
- [x] GMSC 원본 데이터 확보 및 검증
- [x] Dockerfile / docker-compose.yml 스켈레톤
- [ ] Docker 실제 빌드 검증 (현재 환경에 docker CLI 없음 — 아래 미해결 이슈 참조)

## 2. 마지막 세션 요약

**2026-09-08 — 뼈대 세팅 세션**

- 한 일: git identity 설정, `~/credit-scoring` 생성 및 `git init`, `.venv` 생성, 의존성 설치 및
  `requirements.lock` 고정, 디렉터리 구조 생성, `config.yaml` 작성, ADR 6건 작성,
  CLAUDE.md / PROGRESS.md / README.md 작성, GMSC 데이터 다운로드 및 검증, 첫 커밋.
- 확인된 수치 (GMSC `data/raw/cs-training.csv`):
  - 행 수 150,000 / 열 12 (인덱스 컬럼 포함)
  - `SeriousDlqin2yrs` 양성 비율 6.684% (10,026건) — 스펙의 "약 6.7%" 와 일치
  - 결측: `MonthlyIncome` 29,731건(19.82%), `NumberOfDependents` 3,924건(2.62%)
- 모델 코드는 작성하지 않음(이번 세션 범위 밖).

## 3. 다음 할 일 (우선순위 순)

1. `src/data/loader.py` — config 기반 GMSC 로더 + German Credit 로더 함수(미사용, ADR-006)
2. `src/data/simulator.py` — 대안 5변수 생성(target 상관 0.3~0.5 검증 포함), 씬파일러 마스킹/노이즈 (ADR-001)
3. `src/data/bias.py` — gender/age_group 생성 및 측정 편향 주입 (ADR-002)
4. `src/data/preprocess.py` — 결측 처리, 스케일링, train/valid/test 0.7/0.15/0.15 층화 분할
5. `tests/` — 시뮬레이터 상관계수 범위, 분할 비율, 라벨 누수 방지 테스트
6. F2(모델 학습) 이후는 위 8개 항목 순서를 따른다.

## 4. 미해결 이슈

- **[결정 대기] 기능 요구사항 8개 목록 확정** — 위 F1~F8 은 프로젝트 목표와 ADR 로부터 정리한 초안이다.
  사용자의 원 스펙 문서와 대조해 항목명을 확정할 것.
- **[환경] docker CLI 없음** — WSL2 배포판에 docker 가 설치돼 있지 않다
  (`The command 'docker' could not be found in this WSL 2 distro`).
  Docker Desktop 의 WSL 통합을 켜거나 배포판 내부에 엔진을 설치해야 컨테이너 검증이 가능하다.
  현재 `Dockerfile` / `docker-compose.yml` 은 **빌드 미검증 스켈레톤**이다.
- **[환경] GPU 없음, 메모리 7.4 GiB** — 150,000행 규모이므로 문제되지 않으나 SHAP 전체 데이터 계산은
  샘플링이 필요할 수 있다.
- **[데이터 품질] `age` 컬럼에 0 값 존재** (min=0, max=109). ADR-002 의 `age_group`(20-34/35-54/55+)
  구간화 전에 이상치 처리 방침을 정해야 한다. (제외 / 결측 처리 / 하한 클리핑)
- **[예정] 승인 임계값 0.30 재검토** — PR 곡선 확인 후 조정 여부 판단 (ADR-003).

## 5. 세션 히스토리

### 2026-09-08 — 뼈대 세팅
- 프로젝트 구조·설정·ADR 6건·문서 3종 생성, `.venv` 및 의존성 고정 완료.
- GMSC 150,000행 검증 통과 (양성률 6.684%, MonthlyIncome/NumberOfDependents 결측 확인).
- 첫 커밋 `chore: project skeleton, config, ADRs`. 모델 코드는 착수하지 않음.
