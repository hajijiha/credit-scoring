# 실험 보고서 구성

보고서 원문은 `docs/report/`의 장별 Markdown이며,
배포 파일은 [PDF](experiment_report.pdf)와 [HTML](experiment_report.html)이다.

| 문서 | 내용 |
|---|---|
| [00_cover.md](report/00_cover.md) | 표지·면책·목차 |
| [00a_reading_guide.md](report/00a_reading_guide.md) | 읽기 안내 |
| [01_summary.md](report/01_summary.md) | 요약 |
| [02_problem.md](report/02_problem.md) | 문제 정의 |
| [03_system.md](report/03_system.md) | 시스템 개요 |
| [04_data_eda.md](report/04_data_eda.md) | 데이터와 EDA |
| [05_simulator.md](report/05_simulator.md) | 대안 데이터 시뮬레이터 |
| [06_models.md](report/06_models.md) | 모델 비교 |
| [07_xai.md](report/07_xai.md) | 설명가능성 |
| [08_fairness.md](report/08_fairness.md) | 공정성과 완화 |
| [09_anova.md](report/09_anova.md) | 통계 검증 |
| [10_errors.md](report/10_errors.md) | 발견·수정한 오류 |
| [11_limitations.md](report/11_limitations.md) | 한계 |
| [12_conclusion.md](report/12_conclusion.md) | 결론 |
| [99_appendix.md](report/99_appendix.md) | 부록 |

## 재생성

학습·분석 산출물을 생성한 뒤 다음 명령으로 HTML을 빌드한다.

```bash
python scripts/build_report.py
python scripts/verify_report_numbers.py
```

생성된 `docs/experiment_report.html`을 브라우저에서 열고 PDF로 저장한다.
인쇄 설정은 A4, 배경 그래픽 활성화, 브라우저 머리글·바닥글 비활성화다.
표와 그림은 실험 산출물을 사용하며, 수치 대조 스크립트로 인용 값을 확인한다.
