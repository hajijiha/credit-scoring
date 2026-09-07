# PROGRESS

> 세션 시작 시 이 파일을 먼저 읽는다. 작업이 하나 끝날 때마다 즉시 갱신한다(마지막에 몰아쓰지 않는다).

## 1. 현재 상태

과제 스펙의 기능 요구사항 8개 기준 진행도 (원문: `docs/SPEC.md`):

- [ ] **F1. 데이터 파이프라인** — 로더 / 대안 데이터 시뮬레이터 / 편향 주입 / 전처리 / 분할
  - [x] `src/data/loader.py` — `load_gmsc()`, `load_german_credit()` (ADR-006, ADR-007)
  - [x] `tests/test_loader.py` — 6건 통과
  - [ ] `src/data/simulator.py` — 대안 5변수, 씬파일러 처리 (ADR-001)
  - [ ] `src/data/bias.py` — gender/age_group 생성, 측정 편향 주입 (ADR-002)
  - [ ] `src/data/preprocess.py` — 결측 처리, 스케일링, 0.7/0.15/0.15 층화 분할
- [ ] **F2. 신용평가 모델** — 전통 vs 전통+대안 비교, 불균형 처리, 점수·등급·승인 판정 (ADR-003), MLflow 기록
- [ ] **F3. XAI** — SHAP 전역/개별 설명
- [ ] **F4. 공정성 · Mitigation** — Fairlearn 지표(DI 0.8~1.25, EO diff ≤ 0.10) + ThresholdOptimizer 전/후 비교 (ADR-004)
- [ ] **F5. ANOVA** — RepeatedStratifiedKFold 5×4 (그룹당 n=20), ANOVA + 사후검정 (ADR-005)
- [ ] **F6. FastAPI 7 엔드포인트** — 포트 8000
- [ ] **F7. Streamlit 5탭 대시보드** — 포트 8501, 추론은 FastAPI HTTP 경유만
- [ ] **F8. 구조 · 제출 요건** — 디렉터리 구조, 문서, Docker 구성, 재현성

기반 작업:

- [x] 프로젝트 스켈레톤, git 저장소, `.venv` (Python 3.12.3)
- [x] `config/config.yaml` — 모든 설정값 단일 출처
- [x] ADR-001 ~ ADR-007
- [x] GMSC 원본 데이터 확보 및 검증
- [x] Dockerfile / docker-compose.yml 스켈레톤 (내용은 아직 스켈레톤, 빌드 검증 미실시)
- [x] SHAP 0.52 동작 확인 (numpy 2.5 / pandas 3.0 / sklearn 1.9 조합)
- [ ] `docs/SPEC.md` — 과제 원문 저장 (아래 미해결 이슈 참조)

## 2. 마지막 세션 요약

**2026-09-08 (2세션) — 로더 구현**

- 한 일: Docker 사용 가능 확인, SHAP 스모크 테스트, ADR-007 작성,
  `src/data/loader.py` 및 `tests/test_loader.py` 구현, PROGRESS 체크리스트를 F1~F8 로 교체.
- 확인된 수치:
  - **ADR-007 age 필터: 1건 제거** (age=0 인 행 1건). 150,000 → **149,999행**
  - 타겟 양성 10,026건 / 149,999행 = **6.684%** (제거된 1건은 음성)
  - 결측: `MonthlyIncome` 29,731건(19.82%), `NumberOfDependents` 3,924건(2.62%)
  - `age` 범위: 필터 후 min=21, max=109
  - `pytest` 6건 전부 통과
- 환경: Docker 29.2.1 / Compose v5.1.0 / hello-world 정상
- SHAP 스모크: LightGBM 4.7.0 + shap 0.52.0 → `TreeExplainer` → `summary_plot`, `waterfall` 저장까지 정상.
  이진 분류에서 `Explanation.values` 가 `(n, features)` 2D 로 반환됨(클래스 축 없음) — 구현 시 참고.

**2026-09-08 (1세션) — 뼈대 세팅**

- git identity, 프로젝트 구조, `.venv`, 의존성 고정(`requirements.lock`), config.yaml,
  ADR-001~006, CLAUDE.md / PROGRESS.md / README.md, GMSC 다운로드·검증, 첫 커밋.

## 3. 다음 할 일 (우선순위 순)

1. **`docs/SPEC.md` 저장** — 과제 원문이 아직 전달되지 않았다. 받는 즉시 저장하고 위 F1~F8 항목의
   세부 요건(엔드포인트 7개 목록, 대시보드 5탭 구성, 제출 요건)을 이 체크리스트에 반영한다.
2. `src/data/simulator.py` — 대안 5변수 생성, target 상관 0.3~0.5 검증, 씬파일러 마스킹/노이즈 (ADR-001)
3. `src/data/bias.py` — gender / age_group 생성 및 측정 편향 주입 (ADR-002)
4. `src/data/preprocess.py` — 결측 처리, 스케일링, 층화 분할 0.7/0.15/0.15
5. 각 단계별 테스트 (`tests/test_simulator.py`, `tests/test_preprocess.py`)
6. F2(모델 학습) 착수

## 4. 미해결 이슈

- **[결정 대기] `docs/SPEC.md` 원문 미수신** — "아래에 원문 붙여넣기"로 요청받았으나 본문이 비어 있었다.
  F1~F8 상위 항목명만 반영한 상태이며, 세부 요건(FastAPI 7개 엔드포인트의 각 명세, Streamlit 5탭 구성,
  제출 요건)은 아직 알 수 없다. 원문 확보 전에는 F6/F7 착수 불가.
- **[미검증] Docker 이미지 빌드** — Docker 29.2.1 / Compose v5.1.0 사용 가능해졌으나,
  `Dockerfile` / `docker-compose.yml` 은 여전히 스켈레톤이고 실제 빌드는 아직 돌려보지 않았다.
  F8 단계에서 검증한다.
- **[구조] `load_config()` 위치** — 현재 `src/data/loader.py` 안에 있다. F2 이후 다른 모듈이 config 를
  읽기 시작하면 공용 모듈(`src/config.py` 등)로 분리해야 한다. 분리 시점에 사용자 확인 필요.
- **[예정] 승인 임계값 0.30 재검토** — PR 곡선 확인 후 조정 여부 판단 (ADR-003).
- **[환경] GPU 없음, 메모리 7.4 GiB** — 150,000행 규모는 문제없으나 SHAP 전체 데이터 계산 시 샘플링 필요.

## 5. 세션 히스토리

### 2026-09-08 (2세션) — 로더 구현
- SHAP 0.52 + numpy 2.5 / pandas 3.0 / sklearn 1.9 조합 정상 동작 확인, Docker 29.2.1 사용 가능.
- ADR-007(age<18 제거) 작성, `loader.py` 구현 → 1건 제거 / 149,999행 / 양성률 6.684%.
- `tests/test_loader.py` 6건 통과. 과제 원문(SPEC.md) 미수신으로 F6/F7 세부 요건은 보류.

### 2026-09-08 (1세션) — 뼈대 세팅
- 프로젝트 구조·설정·ADR 6건·문서 3종 생성, `.venv` 및 의존성 고정 완료.
- GMSC 150,000행 검증 통과 (양성률 6.684%, MonthlyIncome/NumberOfDependents 결측 확인).
- 첫 커밋 `chore: project skeleton, config, ADRs`. 모델 코드는 착수하지 않음.
