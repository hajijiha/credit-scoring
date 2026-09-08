# GitHub 제출 준비 가이드

SPEC §8 은 **commit 10개 이상**, **Issue 또는 PR 3개 이상**을 요구한다.
현재 로컬 커밋은 충분하고(13개), 원격 저장소가 없다.

## 1. 원격 저장소 생성 (사용자가 직접)

웹에서 새 저장소를 만든 뒤:

```bash
cd ~/credit-scoring
git remote add origin https://github.com/<사용자명>/credit-scoring.git
git branch -M main
git push -u origin main
```

또는 `gh` CLI 를 쓴다면:

```bash
gh auth login
gh repo create credit-scoring --private --source=. --remote=origin --push
```

> 현재 환경에는 `gh` 가 설치돼 있지 않다. `sudo apt install gh` 로 설치하거나 웹에서 만들면 된다.

## 2. Issue 역기록 계획

개발 중 실제로 발견하고 해결한 문제들이다. 사후 기록이라도 개발 이력으로서 의미가 있다.
**각 Issue 본문에 해당 커밋 해시를 링크하고 "closed by <sha>" 로 닫는다.**

| # | 제목 | 라벨 | 내용 요약 | 관련 커밋 |
|---|---|---|---|---|
| 1 | 대안 데이터 5변수의 결합 누수로 combined AUC 0.9999 | `bug` `data` | 개별 상관은 SPEC 범위(0.30~0.49)를 지켰으나 타겟 조건부로 변수들이 거의 독립이라 정보가 합산됐다. 조건부 독립 가정 결합 Cohen's d = 3.96 → 이론 AUC 0.9975(관측 0.9959). 공유 잠재요인 도입으로 0.9470 까지 낮춤. | `5114a85` |
| 2 | 씬파일러 판정 두 조건이 동일 집합이라 OR 정의가 형해화 | `bug` `data` | 연체 결측과 이력 부족이 같은 행에만 적용돼 교집합=합집합=45,000. 유형을 3갈래로 분화(ADR-010). | `5114a85` |
| 3 | 재학습 후 ANOVA 재실행 누락으로 `/anova` 가 stale 결과 서빙 | `bug` `api` | `fold_auc.csv` 는 갱신됐으나 `anova_results.json` 이 이전 데이터 기준이었다. F=4514.60 → 4245.65. | `ffedcf1` |
| 4 | `/explain` 이 연체 '기록 없음' 입력에서 500 | `bug` `api` `xai` | `float(None)` 실패. 씬파일러가 그대로 걸리는 경로였다. 센티넬 표기 + "기록이 없습니다" 문장 분기. | `07ba643` |
| 5 | 거절 사유가 SHAP Waterfall 과 다른 피처를 가리킴 | `bug` `xai` | 역스케일한 값을 모델에 넣어 SHAP 을 계산했다. 모델 입력과 표시값을 분리(`display_row`). | `06488b9` |
| 6 | 전처리기 joblib 을 다른 프로세스에서 역직렬화 불가 | `bug` `infra` | `python -m src.data.preprocessor` 로 저장하면 클래스 `__module__` 이 `__main__` 이 된다. CLI 분리. | `ffedcf1` |
| 7 | 챔피언 알고리즘 선정 근거가 통계와 불일치 | `docs` `modeling` | Tukey HSD 에서 XGBoost vs LightGBM p_adj=0.6468(유의차 없음). "성능 우위" 근거를 SHAP 지원 성숙도로 교체(ADR-012). | `06488b9` |
| 8 | ThresholdOptimizer 가 승인율을 0.80 → 0.955 로 끌어올림 | `bug` `fairness` | 부도율 6.7% 에서 accuracy 최대화는 "거의 전부 승인"이 최적해. 목적함수를 `balanced_accuracy_score` 로 교체(ADR-004 v2). | `064ff62` |

## 3. PR 구성 계획

이미 `main` 에 직접 커밋했으므로, 앞으로의 작업을 브랜치로 나눠 PR 을 만든다.
남은 작업이 그대로 PR 3건이 된다.

| PR | 브랜치 | 내용 | 닫을 Issue |
|---|---|---|---|
| 1 | `docs/experiment-report` | 실험 리포트 PDF 20페이지 작성 | — |
| 2 | `feat/notebooks` | EDA·모델링 노트북 추가 (SPEC 권장 항목) | — |
| 3 | `chore/ci` | GitHub Actions 로 pytest 자동 실행 | — |

이미 끝난 작업을 PR 로 만들고 싶다면, 각 기능 커밋을 브랜치로 되짚어 PR 을 열고
"이미 main 에 병합됨" 을 본문에 밝히는 방법도 있다. 다만 **작업 이력을 사후 조작하는 것으로
보일 수 있으므로 권하지 않는다.** 남은 작업으로 PR 요건을 채우는 편이 정직하다.

## 4. 체크리스트

- [ ] 원격 저장소 생성 및 `git push`
- [ ] 위 Issue 8건 등록 (3건 이상이면 요건 충족)
- [ ] 남은 작업을 브랜치 → PR 로 진행 (3건 이상)
- [ ] `docs/experiment_report.pdf` 20페이지 이상
