"""리포트 인용 수치 대조 스크립트.

`docs/report/*.md` 에 적힌 수치를 산출물(`data/processed/`, `models/`, 원본 CSV)에서 **다시 계산**해
문서 안에 그 값이 실제로 있는지 확인한다. 문서 속 수치가 재생성 이전 값으로 남는 stale 오류
(리포트 9.2절, 5건 발생)를 잡기 위한 도구다.

실행 (프로젝트 루트 어디서든):
    python scripts/verify_report_numbers.py                 # 전체
    python scripts/verify_report_numbers.py --chapters 3 6  # 일부 장

전제: `scripts/bootstrap.sh` 로 산출물이 준비돼 있어야 한다. 일부 검사는 모델을 재학습(설명 소거 대조)하고
API 를 in-process 로 호출하므로 수 분이 걸린다.

대조 방식
- 값은 문서와 같은 형식(자릿수·부호·천 단위 구분)으로 만들어 **같은 줄**에 함께 있는지 본다.
- 문서의 수학 기호 마이너스(−)와 ASCII 하이픈(-)은 같은 것으로 취급한다.
- 대조 범위: 3·6·9·10·11장, 1장·요약·부록의 핵심 수치. 4·5·7·8장은 작성 세션의 대조 기록(PROGRESS.md)을 따른다.
종료 코드: 전부 일치하면 0, 하나라도 불일치하면 1.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import os
import re
import subprocess
import sys
from functools import cached_property
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from src.config import load_config  # noqa: E402

CFG = load_config()
REPORT = Path(CFG["report"]["source_dir"])
PROCESSED = Path(CFG["preprocess"]["output_dir"])
EXPLAIN = Path(CFG["explainer"]["output_dir"])
ASSETS = Path(CFG["explainer"]["report_assets_dir"])
FILES = {
    "guide": "00a_reading_guide.md", "summary": "01_summary.md", "1": "02_problem.md", "4": "05_simulator.md", "2": "03_system.md", "3": "04_data_eda.md",
    "6": "07_xai.md", "9": "10_errors.md", "10": "11_limitations.md", "11": "12_conclusion.md",
    "appendix": "99_appendix.md",
}


# --- 대조 도구 ----------------------------------------------------------------

class Checker:
    def __init__(self) -> None:
        self.results: list[tuple[str, str, bool, list[str]]] = []
        self.docs: dict[str, str] = {}

    def doc(self, chapter: str) -> str:
        if chapter not in self.docs:
            path = REPORT / FILES[chapter]
            self.docs[chapter] = path.read_text(encoding="utf-8") if path.exists() else ""
        return self.docs[chapter]

    def has(self, chapter: str, label: str, *subs: str, line: bool = True) -> None:
        """subs 가 문서에 (기본: 같은 줄에) 모두 있는지."""
        text = _norm(self.doc(chapter))
        needles = [_norm(s) for s in subs]
        if line and len(needles) > 1:
            ok = any(all(n in ln for n in needles) for ln in text.splitlines())
        else:
            ok = all(n in text for n in needles)
        self.results.append((chapter, label, ok, needles))

    def true(self, chapter: str, label: str, condition: bool, detail: object = "") -> None:
        self.results.append((chapter, label, bool(condition), [str(detail)] if detail != "" else []))


def _norm(text: str) -> str:
    return text.replace("−", "-")


def sg(value: float, digits: int) -> str:
    """부호를 붙인 고정 소수."""
    return f"{value:+.{digits}f}"


def pct(value: float, digits: int = 2) -> str:
    return f"{value * 100:.{digits}f}%"


def half_up(numerator: int, denominator: int, digits: int = 1) -> str:
    """정확한 분수를 반올림(half-up)한 백분율. 부동소수 오차로 65.15 → 65.1 이 되는 것을 막는다."""
    scale = 10 ** digits
    return f"{math.floor(numerator * 100 * scale / denominator + 0.5) / scale:.{digits}f}%"


def logit(p: float) -> float:
    return math.log(p / (1 - p))


# --- 공용 계산 (필요할 때 한 번만) -----------------------------------------------

class Context:
    def __init__(self) -> None:
        logging.disable(logging.CRITICAL)
        self.cfg = CFG
        self.y = self.cfg["data"]["target_column"]

    @cached_property
    def explain(self):
        from src.explainer.run_explain import build_explainer
        return build_explainer(self.cfg)

    @property
    def ex(self):
        return self.explain[0]

    @property
    def ctx(self):
        return self.explain[1]

    @cached_property
    def sample(self) -> pd.DataFrame:
        e = self.cfg["explainer"]
        return self.ctx["test"].sample(n=e["background_sample_size"],
                                       random_state=self.cfg["project"]["random_state"])

    @cached_property
    def sample_shap(self) -> np.ndarray:
        return self.ex.explain(self.sample)[0]

    @cached_property
    def base_value(self) -> float:
        return self.ex.explain(self.sample)[1]

    @cached_property
    def gi(self) -> pd.DataFrame:
        return pd.read_csv(EXPLAIN / "global_importance.csv")

    @cached_property
    def samples(self) -> dict:
        return json.loads((EXPLAIN / "rejection_reasons_sample.json").read_text())

    @cached_property
    def proba(self) -> pd.Series:
        test, cols = self.ctx["test"], self.ctx["columns"]
        return pd.Series(self.ctx["model"].predict_proba(test[cols])[:, 1], index=test.index)

    @cached_property
    def anova(self) -> dict:
        return json.loads((PROCESSED / self.cfg["anova"]["result_file"]).read_text())

    @cached_property
    def fairness(self) -> dict:
        return json.loads((PROCESSED / self.cfg["api"]["precomputed"]["fairness"]).read_text())

    @cached_property
    def models(self) -> pd.DataFrame:
        return pd.read_csv(PROCESSED / self.cfg["api"]["precomputed"]["model_results"])

    @cached_property
    def champion(self) -> pd.Series:
        m, c = self.models, self.cfg["models"]["champion"]
        return m[(m.algorithm == c["algorithm"]) & (m.feature_group == c["feature_group"])
                 & (m.imbalance == c["imbalance"])].iloc[0]

    @cached_property
    def uplift(self) -> pd.DataFrame:
        up = pd.read_csv(PROCESSED / self.cfg["api"]["precomputed"]["uplift"])
        return up[up.algorithm == self.cfg["models"]["champion"]["algorithm"]].set_index("segment")

    def seg(self, key: str) -> pd.Series:
        return self.uplift.loc[[i for i in self.uplift.index if i.endswith(key)][0]]

    @cached_property
    def sensitivity(self) -> pd.DataFrame:
        return pd.read_csv(PROCESSED / "fairness" / "bias_ratio_sensitivity.csv").set_index("bias_ratio")

    @cached_property
    def gmsc(self) -> pd.DataFrame:
        from src.data.loader import load_gmsc
        return load_gmsc(self.cfg)

    @cached_property
    def simulated(self) -> pd.DataFrame:
        from src.data.simulator import simulate
        return simulate(self.gmsc, self.cfg)

    @cached_property
    def full_raw(self) -> pd.DataFrame:
        pre = self.ctx["preprocessor"]
        return pd.concat([pre.inverse_transform_numeric(pd.read_parquet(PROCESSED / f"{k}.parquet"))
                          for k in ("train", "valid", "test")])

    @cached_property
    def alt_top5(self) -> np.ndarray:
        cols = self.ctx["columns"]
        idx = [cols.index(f) for f in self.cfg["features"]["alternative"]]
        top5 = np.argsort(-np.abs(self.sample_shap), axis=1)[:, :5]
        return np.isin(top5, idx)

    @cached_property
    def randomized_band(self) -> dict:
        test, out = self.ctx["test"], {}
        for attr in ("gender", "age_group"):
            rows = {}
            for group, rule in self.fairness["mitigation"][attr]["adjusted_thresholds"].items():
                mask = (test[attr] == group).to_numpy()
                lo, hi = sorted([rule["operation0"]["threshold"], rule["operation1"]["threshold"]])
                band = int((mask & (self.proba.to_numpy() > lo) & (self.proba.to_numpy() <= hi)).sum())
                rows[group] = dict(lo=lo, hi=hi, band=band, n=int(mask.sum()), p_ignore=rule["p_ignore"])
            out[attr] = rows
        return out


# --- 장별 대조 -----------------------------------------------------------------

def check_chapter_3(c: Checker, x: Context) -> None:
    ch, y, g, sim = "3", x.y, x.gmsc, x.simulated
    raw = pd.read_csv(Path(x.cfg["paths"]["data_raw"]) / x.cfg["data"]["gmsc_file"])
    c.true(ch, "원본 150,000행 · 필터 1행", len(raw) == 150_000 and int((raw.age < 18).sum()) == 1 and raw.age.min() == 0)
    c.has(ch, "원본 행수", "150,000행에 타겟 1개와 피처 10개")
    c.true(ch, "필터 후 11컬럼", g.shape[1] == 11)
    c.has(ch, "필터 후", f"{len(g):,}행, 최소 연령은 {int(g.age.min())}세")
    counts = g[y].value_counts()
    c.has(ch, "비부도", f"{counts[0]:,}", pct(counts[0] / len(g)))
    c.has(ch, "부도", f"**{counts[1]:,}**", f"**{pct(counts[1] / len(g))}**")
    c.has(ch, "불균형 배율", f"약 {round(counts[0] / counts[1])}건")
    miss = g.isna().mean()
    c.has(ch, "소득 결측", "`MonthlyIncome` |", pct(miss["MonthlyIncome"]))
    c.has(ch, "부양가족 결측", "`NumberOfDependents` |", pct(miss["NumberOfDependents"]))
    masked = x.cfg["simulator"]["thin_filer_masked_columns"]
    induced = sim[masked].isna().mean()
    c.true(ch, "유도 결측 3컬럼 동일", induced.nunique() == 1)
    c.has(ch, "유도 결측", "연체 3컬럼 (각각)", f"**{pct(induced.iloc[0])}**")
    mi = g.MonthlyIncome.isna()
    c.has(ch, "소득 있음 행", "있음", f"{g.DebtRatio[~mi].median():.3f}", pct(g[y][~mi].mean()))
    c.has(ch, "소득 결측 행", "결측 |", f"**{g.DebtRatio[mi].median():,.0f}**", pct(g[y][mi].mean()))
    big = g.DebtRatio > g.DebtRatio.quantile(0.99)
    c.true(ch, "DebtRatio 극단값 대부분이 소득 결측", g.MonthlyIncome[big].isna().mean() > 0.5,
           f"{g.MonthlyIncome[big].isna().mean():.3f}")
    fmt = {"RevolvingUtilizationOfUnsecuredLines": (".3f", ".2f"), "DebtRatio": (".3f", ",.0f"),
           "MonthlyIncome": (",.0f", ",.0f"), "age": (".0f", ".0f"),
           "NumberOfOpenCreditLinesAndLoans": (".0f", ".0f"), "NumberOfTimes90DaysLate": (".0f", ".0f")}
    for col, (fm, fp) in fmt.items():
        s = g[col].dropna()
        z = (s.max() - s.mean()) / s.std()
        c.has(ch, f"분포 {col}", f"`{col}`", f"| {s.median():{fm}} |", f"| {s.quantile(.99):{fp}} |",
              f"{s.max():,.0f}", f"{z:.1f}")
    rv = g.RevolvingUtilizationOfUnsecuredLines
    c.has(ch, "소진율>1 비율", "소진율 > 1 인 행", pct((rv > 1).mean()))
    c.has(ch, "소진율>1 부도", "부도율 (소진율 > 1)", pct(g[y][rv > 1].mean()))
    c.has(ch, "소진율≤1 부도", "부도율 (소진율 ≤ 1)", pct(g[y][rv <= 1].mean()))
    c.has(ch, "피어슨", "피어슨 상관", f"{np.corrcoef(rv, g[y])[0, 1]:+.3f}".replace("+", ""))
    c.has(ch, "스피어만", "스피어만 상관", sg(spearmanr(rv, g[y]).statistic, 3))
    c.has(ch, "6배", f"{g[y][rv > 1].mean() / g[y][rv <= 1].mean():.0f}배", line=False)
    code = g[masked].isin([96, 98])
    c.true(ch, "96/98 세 컬럼 같은 행", code.any(axis=1).sum() == code.all(axis=1).sum())
    c.has(ch, "96/98 행수", f"| {int(code.any(axis=1).sum())} (세 컬럼 모두 같은 행) |")
    c.has(ch, "96/98 부도", f"**{pct(g[y][code.any(axis=1)].mean())}**")
    a, b = masked[0], masked[2]
    keep = ~code.any(axis=1)
    c.has(ch, "연체 상관 전체", "(전체)", f"{g[a].corr(g[b]):.3f}")
    c.has(ch, "연체 상관 제외", "269행 제외", f"**{g.loc[keep, a].corr(g.loc[keep, b]):.3f}**")
    pairs = [(masked[i], masked[j]) for i in range(3) for j in range(i + 1, 3)]
    allc = [g[p].corr(g[q]) for p, q in pairs]
    exc = [g.loc[keep, p].corr(g.loc[keep, q]) for p, q in pairs]
    c.has(ch, "연체 상관 범위", f"0.98~0.99 로 끌어올린다. 제외하면 {min(exc):.2f}~{max(exc):.2f}")
    c.true(ch, "전체 연체 상관 0.98~0.99", 0.975 <= min(allc) and max(allc) < 0.995, allc)
    alt = x.cfg["features"]["alternative"]
    means = sim.groupby(y)[alt].mean()
    for f in alt:
        s = sim[f]
        is_rate = s.max() <= 1.0
        rng = f"{s.min():.2f} ~ {s.max():.2f}" if is_rate else f"{s.min():.0f} ~ {s.max():.0f}"
        m = ".3f" if is_rate else ".2f"
        c.has(ch, f"대안 {f}", f"`{f}`", rng, f"{s.mean():{m}}", f"{means.loc[0, f]:{m}}",
              f"{means.loc[1, f]:{m}}", f"{abs(np.corrcoef(s, sim[y])[0, 1]):.4f}")
    trad = x.cfg["features"]["traditional"]
    tc = sim[trad].corrwith(sim[y]).abs()
    c.has(ch, "전통 최대 상관", f"최대 {tc.max():.3f}(30-59일 연체)")
    c.true(ch, "전통 최대는 30-59일", tc.idxmax() == masked[0])
    cm = sim[trad + alt].corr()
    aa = cm.loc[alt, alt].to_numpy()[np.triu_indices(len(alt), 1)]
    c.has(ch, "대안-대안", f"**{aa.min():.3f} ~ {aa.max():.3f}** (평균 {aa.mean():.3f})")
    age_c = cm.loc["age", alt]
    biased = x.cfg["simulator"]["biased_features"]
    rest = [f for f in alt if f not in biased]
    c.true(ch, "대안-연령 두 층", age_c[biased].round(2).nunique() == 1 and age_c[rest].round(2).nunique() == 1)
    c.has(ch, "대안-연령", f"{age_c[biased].iloc[0]:.2f} (통신·공과금) / {age_c[rest].iloc[0]:.2f} (나머지 셋)")
    inc = cm.loc["MonthlyIncome", alt]
    ok = sim.MonthlyIncome.notna()
    logc = [np.corrcoef(sim.loc[ok, f], np.log1p(sim.loc[ok, "MonthlyIncome"]))[0, 1] for f in alt]
    c.has(ch, "대안-소득", f"{inc.min():.2f} ~ {inc.max():.2f} (`log1p` 소득과는 {np.mean(logc):.2f})")
    c.true(ch, "log1p 상관 범위 좁음", max(logc) - min(logc) < 0.01, logc)
    tt = cm.loc[trad, trad].abs()
    others = [t for t in trad if t not in masked]
    off = tt.loc[others, others].to_numpy()[np.triu_indices(len(others), 1)]
    c.true(ch, "그 밖 전통 대부분 <0.15", (off < 0.15).mean() > 0.8, f"{(off < 0.15).mean():.2f}")
    c.has(ch, "계좌-부동산", f"계좌 수 ↔ 부동산 대출 {tt.loc['NumberOfOpenCreditLinesAndLoans', 'NumberRealEstateLoansOrLines']:.2f} 이 최대")
    c.true(ch, "계좌-부동산이 그 밖 최대", abs(off.max() - tt.loc["NumberOfOpenCreditLinesAndLoans", "NumberRealEstateLoansOrLines"]) < 1e-9)
    cfg0 = copy.deepcopy(x.cfg)
    cfg0["simulator"]["bias_ratio"] = 0.0
    from src.data.simulator import simulate
    s0 = simulate(g, cfg0)
    c.true(ch, "bias 0 에서 다섯 피처 0.22", all(round(float(np.corrcoef(s0[f], s0.age)[0, 1]), 2) == 0.22 for f in alt))
    sc = x.cfg["simulator"]
    miss2 = sim[masked].isna().sum(1) >= sc["thin_filer_min_missing"]
    hist = sim[sc["credit_history"]["column"]] < sc["credit_history"]["min_months_for_normal"]
    c.has(ch, "판정 결측", "연체 컬럼 2개 이상 결측", f"{int(miss2.sum()):,}")
    c.has(ch, "판정 이력", "거래 이력 12개월 미만", f"{int(hist.sum()):,}")
    c.has(ch, "교집합", "교집합 (둘 다)", f"{int((miss2 & hist).sum()):,}")
    flag = sim[sc["thin_filer_flag_column"]]
    c.true(ch, "합집합=씬파일러", int((miss2 | hist).sum()) == int(flag.sum()))
    c.has(ch, "씬파일러", f"**{int(flag.sum()):,} ({flag.mean() * 100:.1f}%)**")
    types = sim[sc["thin_filer_type_column"]]
    months = sim.groupby(types)[sc["credit_history"]["column"]].mean()
    thin_n = int(flag.sum())
    c.has(ch, "유형 일반", "| 일반 |", f"{int((types == 'none').sum()):,}", f"({months['none']:.1f})")
    for t in ("missing_only", "history_only", "both"):
        n = int((types == t).sum())
        c.has(ch, f"유형 {t}", f"`{t}`", f"{n:,}", half_up(n, thin_n), f"({months[t]:.1f})")
    feats = ["RevolvingUtilizationOfUnsecuredLines", "NumberOfOpenCreditLinesAndLoans", "MonthlyIncome", "age",
             sc["credit_history"]["column"]] + alt
    z = sim[feats].apply(lambda col: (col - col.mean()) / col.std(ddof=0)).groupby(flag).mean()
    c.has(ch, "프로파일 이력", "`credit_history_months`", sg(z.loc[0, "credit_history_months"], 3), sg(z.loc[1, "credit_history_months"], 3))
    c.has(ch, "프로파일 소진율", "`RevolvingUtilization` |", sg(z.loc[0, feats[0]], 3), sg(z.loc[1, feats[0]], 3))
    c.has(ch, "프로파일 계좌", "`NumberOfOpenCreditLinesAndLoans` |", sg(z.loc[0, feats[1]], 3), sg(z.loc[1, feats[1]], 3))
    rest_z = z[[f for f in feats if f not in (feats[0], feats[1], "credit_history_months")]].abs().to_numpy().max()
    c.true(ch, "그 밖 |z| ≤ 0.005", rest_z <= 0.005, f"{rest_z:.4f}")
    dr = sim.groupby(flag)[y].mean()
    c.has(ch, "씬/일반 부도율", "| 부도율 |", pct(dr[0]), pct(dr[1]))
    for col, groups in (("gender", ("female", "male")), ("age_group", ("20-34", "35-54", "55+"))):
        vc, rate = sim[col].value_counts(), sim.groupby(col)[y].mean()
        for grp in groups:
            c.has(ch, f"보호속성 {grp}", f"| {grp} |", f"{vc[grp]:,}", pct(rate[grp]))
    ar = sim.groupby("age_group")[y].mean()
    c.has(ch, "연령 3배", f"55+ 의 {ar['20-34'] / ar['55+']:.0f}배")
    sens = x.sensitivity.loc[x.cfg["simulator"]["bias_ratio"]]
    c.has(ch, "7장 승인율", f"20-34 {sens['approval_age_group_20-34']:.4f} vs 55+ {sens['approval_age_group_55+']:.4f}")
    c.has(ch, "7장 FPR", f"{x.fairness['before']['age_group']['equalized_odds']['equalized_odds_fpr_diff']:.4f}")
    med = x.ctx["preprocessor"].medians
    c.has(ch, "중앙값", f"`MonthlyIncome` → {med['MonthlyIncome']:,.0f}", f"`NumberOfDependents` → {med['NumberOfDependents']:.0f}")
    rates = {}
    for k in ("train", "valid", "test"):
        d = pd.read_parquet(PROCESSED / f"{k}.parquet")
        rates[k] = d[y].mean()
        c.has(ch, f"분할 {k}", f"| {k} |", f"{len(d):,}", f"{d[y].mean() * 100:.4f}%")
        c.true(ch, f"{k} 24열", d.shape[1] == 24)
    c.true(ch, "부도율 차 0.001%p 이내", (max(rates.values()) - min(rates.values())) * 100 < 0.001)
    from src.data.preprocessor import feature_columns
    c.true(ch, "학습 피처 18", len(feature_columns(x.cfg, "combined")) == 18)


def check_chapter_6(c: Checker, x: Context) -> None:
    ch, cfg, cols, test = "6", x.cfg, x.ctx["columns"], x.ctx["test"]
    s, vals, gi = x.sample, x.sample_shap, x.gi
    c.has(ch, "기준값", f"**{x.base_value:.4f}**")
    c.has(ch, "기준값 확률", f"{1 / (1 + math.exp(-x.base_value)):.4f}")
    c.has(ch, "임계 로그오즈", f"{logit(cfg['thresholds']['approval_threshold']):.4f}")
    for i, r in gi.head(15).iterrows():
        c.has(ch, f"중요도 {i + 1}위", f"`{r.feature}`", f"{r.mean_abs_shap:.4f}")
    c.true(ch, "플래그 3종 0", (gi[gi.feature.str.endswith("_is_missing")].mean_abs_shap == 0).all())
    m = gi.mean_abs_shap.to_numpy()
    c.has(ch, "1위/2위", f"{m[0] / m[1]:.2f}배")
    c.has(ch, "1위/3위", f"{m[0] / m[2]:.2f}배")
    alt = gi[gi.feature_type == "alternative"].mean_abs_shap
    tr = gi[gi.feature_type == "traditional"].mean_abs_shap
    trnf = gi[(gi.feature_type == "traditional") & ~gi.feature.str.endswith("_is_missing")].mean_abs_shap
    c.has(ch, "대안 평균", f"**{alt.mean():.4f}**", f"{tr.mean():.4f}", f"**{alt.mean() / tr.mean():.1f}배**")
    c.has(ch, "플래그 제외", f"{trnf.mean():.4f} ({len(trnf)}개)", f"{alt.mean() / trnf.mean():.1f}배")
    share = alt.sum() / gi.mean_abs_shap.sum()
    c.has(ch, "비중", f"**{share * 100:.1f}%**", f"{(1 - share) * 100:.1f}%")
    c.true(ch, "대안 순위 1·2·4·6·7", sorted((gi.index[gi.feature_type == "alternative"] + 1).tolist()) == [1, 2, 4, 6, 7])
    top5, top1 = x.alt_top5.sum(1), x.alt_top5[:, 0]
    thin = s.is_thin_filer.to_numpy()
    for name, mask in (("| 전체 |", np.ones(len(s), bool)), ("| 일반 |", thin == 0), ("| 씬파일러 |", thin == 1)):
        c.has(ch, f"Top5 {name}", name, f"{mask.sum():,}", f"{top5[mask].mean():.2f}",
              half_up(int(top1[mask].sum()), int(mask.sum())))
    for t in ("history_only", "missing_only", "both"):
        mask = s.thin_filer_type.to_numpy() == t
        c.has(ch, f"Top5 {t}", f"└ `{t}`", f"| {mask.sum()} |", f"{top5[mask].mean():.2f}")
    pre = x.ctx["preprocessor"]
    i = cols.index("telecom_payment_rate")
    xv, v = s["telecom_payment_rate"].to_numpy(), vals[:, i]
    slope = np.polyfit(xv, v, 1)[0]
    sd = pre.scaler.scale_[list(pre.numeric_columns).index("telecom_payment_rate")]
    c.has(ch, "telecom spearman", f"**{spearmanr(xv, v).statistic:.3f}**")
    c.has(ch, "telecom 기울기 SD", f"{slope:.3f} 로그 오즈")
    c.has(ch, "telecom 기울기 %p", f"{slope * 0.01 / sd:.3f} 로그 오즈")
    c.has(ch, "telecom 범위", f"{v.min():.2f} ~ {sg(v.max(), 2)}")
    c.has(ch, "telecom sd", f"{sd:.4f}")
    rev = "RevolvingUtilizationOfUnsecuredLines"
    j = cols.index(rev)
    rz, w = s[rev].to_numpy(), vals[:, j]
    raw = x.ctx["test_raw"].loc[s.index, rev]
    p1, p99z = np.percentile(rz, [1, 99])
    c.has(ch, "revolving z 백분위", f"{p1:.3f} ~ {p99z:.3f}")
    c.has(ch, "revolving z 폭", f"폭 {p99z - p1:.3f}")
    c.has(ch, "revolving |z|>1", f"**{(np.abs(rz) > 1).sum()}개**", f"z = {rz.max():.1f}")
    c.has(ch, "revolving 원단위", f"{raw.median():.3f} / {raw.quantile(.99):.2f} / **{raw.max():,.0f}**")
    from src.explainer.explainer import axis_clip
    clip = cfg["explainer"]["dependence_axis_clip"][rev]
    upper, excluded = axis_clip(raw, clip)
    c.has(ch, "절단 상한", f"{clip} 백분위({upper:.2f})")
    c.has(ch, "제외 점 수", f"이상치 {excluded}개")
    outside = (raw > upper).to_numpy()
    c.has(ch, "제외 점 SHAP", f"{w[outside].min():.2f} ~ {w[outside].max():.2f}")
    for lo, hi, label in ((0, .1, "0 ~ 0.1"), (.1, .5, "0.1 ~ 0.5"), (.5, 1.0, "0.5 ~ 1.0"), (1.0, upper, f"1.0 ~ {upper:.2f}")):
        mask = ((raw >= lo) if lo == 0 else (raw > lo)) & (raw <= hi)
        c.has(ch, f"구간 {label}", f"| {label} |", f"| {int(mask.sum())} |", sg(w[mask.to_numpy()].mean(), 3))
    c.has(ch, "revolving spearman", sg(spearmanr(rz, w).statistic, 3))
    c.has(ch, "revolving SHAP 범위", f"{w.min():.2f} ~ {sg(w.max(), 2)}")
    g = x.gmsc[rev]
    c.has(ch, "GMSC 최대", f"{g.max():,.0f}, z = {(g.max() - g.mean()) / g.std():.1f}")
    # 설명 소거: 전통 전용 / 통합 모델을 챔피언 하이퍼파라미터로 재학습해 부호를 대조한다
    import shap
    import xgboost as xgb
    from src.data.preprocessor import feature_columns
    train = pd.read_parquet(PROCESSED / "train.parquet")
    y = x.y
    spw = (train[y] == 0).sum() / (train[y] == 1).sum()
    sp = {}
    for grp in ("traditional_only", "combined"):
        cc = feature_columns(cfg, grp)
        model = xgb.XGBClassifier(**cfg["models"]["xgboost"], scale_pos_weight=spw,
                                  random_state=cfg["project"]["random_state"], n_jobs=cfg["models"]["n_jobs"])
        model.fit(train[cc], train[y])
        sv = shap.TreeExplainer(model)(s[cc]).values
        sp[grp] = {f: spearmanr(s[f], sv[:, cc.index(f)]).statistic
                   for f in ("MonthlyIncome", "age", "NumberOfOpenCreditLinesAndLoans")}
    train_raw = x.ctx["train_raw"]
    for f in sp["combined"]:
        champ = spearmanr(s[f], vals[:, cols.index(f)]).statistic
        c.has(ch, f"설명소거 {f}", f"`{f}`", sg(spearmanr(train_raw[f], train_raw[y]).statistic, 3),
              sg(sp["traditional_only"][f], 3), sg(champ, 3))
        c.true(ch, f"재학습=챔피언 {f}", abs(sp["combined"][f] - champ) < 5e-4)
    names = {"extreme_rejected_thin_filer": "극단 거절 · 씬파일러", "extreme_rejected_normal": "극단 거절 · 일반",
             "borderline_rejected": "**경계 거절**", "borderline_approved": "경계 승인"}
    for key, e in x.samples.items():
        p = float(x.proba.loc[e["index"]])
        c.has(ch, f"사례표 {key}", names[key], f"{e['default_probability']:.4f}", sg(logit(p), 3),
              f"{e['credit_score']} / {e['risk_grade']}", e["age_group"])
        for r in e.get("rejection_reasons", e.get("approval_factors", [])):
            body = r["explanation"].split(" (기여도")[0]
            subs = [body, sg(r["shap_contribution"], 2)]
            if not r["direction_consistent"]:
                subs.append("— 다른 항목과의 조합에 따른 영향입니다")
            c.has(ch, f"사유 {key} #{r['rank']}", *subs)
            if key == "borderline_rejected":
                c.has(ch, f"방향표지 #{r['rank']}", body, "✓" if r["direction_consistent"] else "✗")
    b = x.samples["borderline_rejected"]
    bi = b["index"]
    bshap = x.ex.explain(test.loc[[bi]])[0][0]
    braw = x.ctx["test_raw"].loc[bi]
    thr = logit(cfg["thresholds"]["approval_threshold"])
    c.has(ch, "경계 gap", f"**{logit(float(x.proba.loc[bi])) - thr:.3f}**")
    c.has(ch, "경계 소진율", f"{braw[rev] * 100:.2f}%", f"{bshap[j]:.2f}", line=False)
    c.has(ch, "경계 telecom SHAP", f"({bshap[cols.index('telecom_payment_rate')]:.2f}, 부도 관점)")
    c.has(ch, "경계 나이·이력", f"{braw['age']:.0f}세", f"거래 이력 {int(braw['credit_history_months'])}개월", line=False)
    c.has(ch, "경계 idx", f"test idx {bi}")
    # 반사실: API 를 in-process 로 호출
    from fastapi.testclient import TestClient
    from src.api.main import app
    base = dict(cfg["dashboard"]["default_input"])
    T, U = base["telecom_payment_rate"], base["utility_payment_rate"]
    ref = lambda f: x.ex.reference_value(f, "55+")  # noqa: E731
    with TestClient(app) as client:
        def P(**mod):
            q = dict(base)
            q.update(mod)
            return client.post("/predict", json=q).json()["default_probability"]
        cases = [
            ("없음 (현재)", {}, "68% / 67%"),
            ("**통신비·공과금 각 +2%p**", dict(telecom_payment_rate=T + .02, utility_payment_rate=U + .02), "**70% / 69%**"),
            ("통신비만 +3%p", dict(telecom_payment_rate=T + .03), "71% / 67%"),
            ("공과금만 +7%p", dict(utility_payment_rate=U + .07), "68% / 74%"),
            ("통신비·공과금 각 +5%p", dict(telecom_payment_rate=T + .05, utility_payment_rate=U + .05), "73% / 72%"),
            ("둘 다 55+ 연령대 평균으로", dict(telecom_payment_rate=ref("telecom_payment_rate"),
                                          utility_payment_rate=ref("utility_payment_rate")),
             f"{ref('telecom_payment_rate') * 100:.0f}% / {ref('utility_payment_rate') * 100:.0f}%"),
            ("(참고) 앱 접속 빈도 6 → 23회", dict(app_login_frequency=round(ref("app_login_frequency"))), "68% / 67%"),
            ("(참고) 소비 일관성 25% → 56%", dict(spending_consistency=ref("spending_consistency")), "68% / 67%"),
        ]
        for label, mod, shown in cases:
            p = P(**mod)
            c.has(ch, f"반사실 {label}", label, shown, f"{p:.4f}", "거절" if p >= 0.30 else "승인")
        c.true(ch, "최소성: 동시 +1%p 거절", P(telecom_payment_rate=T + .01, utility_payment_rate=U + .01) >= .30)
        c.true(ch, "최소성: 통신비 +2%p 거절", P(telecom_payment_rate=T + .02) >= .30)
        c.true(ch, "최소성: 공과금 +6%p 거절", P(utility_payment_rate=U + .06) >= .30)
        api = client.post("/explain", json=base).json()
        c.true(ch, "API = 배치 문장", [r["explanation"] for r in api["rejection_reasons"]]
               == [r["explanation"] for r in b["rejection_reasons"]])


def check_chapter_9(c: Checker, x: Context) -> None:
    ch = "9"
    c.has(ch, "F 현행", f"현행 {x.anova['test_1_feature_group']['f_statistic']:.2f}")
    c.true(ch, "F 과거 4514.60 기록", "F=4514.60" in Path("PROGRESS.md").read_text(encoding="utf-8"))
    tests_src = "".join(p.read_text(encoding="utf-8") for p in Path("tests").glob("test_*.py"))
    for name in re.findall(r"`(test_[a-z_]+)`", x_doc(c, ch)):
        c.true(ch, f"테스트 존재 {name}", f"def {name}(" in tests_src)
    pv = x.models.pivot_table(index=["feature_group", "algorithm"], columns="imbalance", values="test_auc")
    d = pv.class_weight - pv.smote
    c.has(ch, "우세", f"| class_weight 우세 | {(d >= 1e-4).sum()} |")
    c.has(ch, "동등", f"| 사실상 동등 | {(d.abs() < 1e-4).sum()} |", f"{d[d.abs() < 1e-4].iloc[0]:.5f}")
    c.has(ch, "SMOTE 우세", f"| SMOTE 우세 | {(d <= -1e-4).sum()} |", f"{d[d <= -1e-4].iloc[0]:.4f}")
    m = x.models
    lg = m[(m.algorithm == "lightgbm") & (m.feature_group == "combined") & (m.imbalance == "smote")].cv_auc_std.iloc[0]
    c.has(ch, "CV SD", f"±{lg:.4f}")
    gi = x.gi
    alt = gi[gi.feature_type == "alternative"].mean_abs_shap
    tr = gi[gi.feature_type == "traditional"].mean_abs_shap
    c.has(ch, "#11 SHAP 1위", f"**{gi.mean_abs_shap[0]:.4f}**", line=False)
    c.has(ch, "#11 2위", f"**`{gi.feature[1]}` {gi.mean_abs_shap[1]:.4f}**")
    c.has(ch, "#11 배율", f"**{gi.mean_abs_shap[0] / gi.mean_abs_shap[1]:.2f}배**")
    c.has(ch, "#11 평균", f"{alt.mean():.4f} / {tr.mean():.4f} ({alt.mean() / tr.mean():.1f}배)")
    tk = {r["comparison"]: r for r in x.anova["test_2_algorithm"]["post_hoc_tukey"]}
    c.has(ch, "#11 Tukey", f"{sg(tk['lightgbm vs xgboost']['mean_diff'], 4)}, **{tk['lightgbm vs xgboost']['p_adj']:.4f}**")
    fs = m[(m.imbalance == "class_weight") & (m.feature_group == "traditional_only")].set_index("algorithm").fit_seconds
    c.has(ch, "#11 학습시간", f"XGB **{fs['xgboost']:.1f}초** vs LGBM **{fs['lightgbm']:.1f}초**")

    def git_show(rev: str, path: str) -> str:
        out = subprocess.run(["git", "show", f"{rev}:{path}"], capture_output=True, text=True)
        return out.stdout if out.returncode == 0 else ""
    old12 = git_show("38259e0", "docs/decisions/ADR-012-champion-algorithm.md")
    old_progress = git_show("06488b9", "PROGRESS.md")
    if old12 and old_progress:
        for val in ("0.6137", "18.3초", "16.8초", "+0.0009"):
            c.true(ch, f"과거값 ADR-012 {val}", val in old12)
        for val in ("1.3460", "0.7769", "0.6563", "0.2011", "1.7배", "3.3배"):
            c.true(ch, f"과거값 PROGRESS {val}", val in old_progress)
    c.true(ch, "ADR-004 40줄", len(Path("docs/decisions/ADR-004-mitigation.md").read_text(encoding="utf-8").splitlines()) == 40)


def x_doc(c: Checker, chapter: str) -> str:
    return c.doc(chapter)


def check_chapter_10(c: Checker, x: Context) -> None:
    ch, m, y = "10", x.models, x.y
    champ = x.champion
    trad = m[(m.algorithm == "xgboost") & (m.feature_group == "traditional_only") & (m.imbalance == "class_weight")].test_auc.iloc[0]
    c.has(ch, "챔피언 AUC", f"**{champ.test_auc:.4f}**")
    c.has(ch, "전통 AUC", f"({trad:.4f})")
    c.has(ch, "재현율", "재현율", f"{champ.test_recall:.4f}")
    c.has(ch, "정밀도", f"**{champ.test_precision:.4f}**")
    c.has(ch, "F1", "| F1 |", f"{champ.test_f1:.4f}")
    c.has(ch, "승인율", "전체 승인율", f"{x.fairness['before']['overall_approval_rate']:.4f}")
    gc = x.ctx["test"].gender.value_counts()
    c.has(ch, "성별 수", f"female {gc['female']:,} / male {gc['male']:,}")
    c.has(ch, "기준값", f"{1 / (1 + math.exp(-x.base_value)):.4f}")
    t1, t2 = x.anova["test_1_feature_group"], x.anova["test_2_algorithm"]
    for key, t in (("검증 1", t1), ("검증 2", t2)):
        sds = t["group_stds"].values()
        c.has(ch, f"η² {key}", key, f"{t['effect_size_eta_squared']:.4f}", f"{min(sds):.4f} ~ {max(sds):.4f}")
    comb = [r for r in t1["post_hoc_tukey"] if "combined" in r["comparison"] and "traditional" in r["comparison"]][0]
    tk = {r["comparison"]: r for r in t2["post_hoc_tukey"]}
    c.has(ch, "mean_diff 1", f"**+{abs(comb['mean_diff']):.4f}**")
    c.has(ch, "mean_diff 2", f"**+{abs(tk['lightgbm vs xgboost']['mean_diff']):.4f}**")
    mant, exp = f"{t1['p_value']:.2e}".split("e-")
    c.has(ch, "p 검증1", f"{mant} × 10⁻{''.join('⁰¹²³⁴⁵⁶⁷⁸⁹'[int(d)] for d in exp.lstrip('0'))}")
    train = pd.read_parquet(PROCESSED / "train.parquet")
    c.has(ch, "조각 행수", f"약 {len(train) / 20:,.0f}행")
    c.true(ch, "조각 부도 ≈350", abs(len(train) / 20 * train[y].mean() - 350) < 10)
    for attr, rows in x.randomized_band.items():
        total = 0
        for group, r in rows.items():
            total += r["band"]
            pi = "0" if r["p_ignore"] == 0 else f"{r['p_ignore'] * 100:.2f}%"
            c.has(ch, f"확률구간 {group}", f"| {group} |", f"{r['lo']:.4f} < p ≤ {r['hi']:.4f}", f"| {r['band']} |",
                  f"{r['band'] / r['n'] * 100:.2f}%", f"| {pi} |")
        c.has(ch, f"확률구간 합계 {attr}", f"**{total:,}**", f"**{total / len(x.ctx['test']) * 100:.2f}%**")
    male = x.fairness["mitigation"]["gender"]["adjusted_thresholds"]["male"]
    female = x.fairness["mitigation"]["gender"]["adjusted_thresholds"]["female"]
    c.has(ch, "0.40 남성", f"{male['p1'] * 100:.1f}% 확률로 `p > {male['operation1']['threshold']:.4f}`", line=False)
    c.has(ch, "0.40 여성", f"`p > {female['operation0']['threshold']:.4f}`, `p > {female['operation1']['threshold']:.4f}`",
          f"{female['p_ignore'] * 100:.2f}%", line=False)
    c.true(ch, "0.40 남성 거절 규칙", male["operation1"]["threshold"] < 0.40 <= male["operation0"]["threshold"])
    c.true(ch, "0.40 여성 승인", 0.40 <= min(female["operation0"]["threshold"], female["operation1"]["threshold"]))
    age_share = sum(r["band"] for r in x.randomized_band["age_group"].values()) / len(x.ctx["test"])
    c.true(ch, "17명 중 1명", round(1 / age_share) == 17)
    full = x.full_raw
    train_raw = x.ctx["train_raw"]
    c.true(ch, "전체 149,999행", len(full) == 149_999)
    alt = x.cfg["features"]["alternative"]
    corr = lambda d, f: abs(np.corrcoef(d[f], d[y])[0, 1])  # noqa: E731
    for f in alt:
        ct, cf = corr(train_raw, f), corr(full, f)
        c.has(ch, f"10.2 {f}", f"| `{f}` |", f"{ct:.4f}", f"{ct - 0.30:+.4f}", f"{cf:.4f}", f"{cf - 0.30:+.4f}")
    lowest = {k: min(alt, key=lambda f: corr(d, f)) for k, d in (("train", train_raw), ("full", full))}
    c.true(ch, "최저: train=utility, 전체=app_login",
           lowest == {"train": "utility_payment_rate", "full": "app_login_frequency"}, lowest)
    c.has(ch, "결론 여유", f"train {corr(train_raw, 'utility_payment_rate') - 0.30:.4f}, 전체 {corr(full, 'app_login_frequency') - 0.30:.4f}", line=False)
    rev = "RevolvingUtilizationOfUnsecuredLines"
    from src.explainer.explainer import axis_clip
    _, excluded = axis_clip(x.ctx["test_raw"].loc[x.sample.index, rev], x.cfg["explainer"]["dependence_axis_clip"][rev])
    c.has(ch, "Dependence 절단 행", f"이상치 {excluded}개 제외")


def check_chapter_11(c: Checker, x: Context) -> None:
    ch = "11"
    narrative = (ASSETS / "fairness_narrative.md").read_text(encoding="utf-8")
    sec5 = narrative.split("## 5. 결론\n\n")[1].split("\n\n## 6.")[0]
    c.true(ch, "§5 원문 포함", sec5 in c.doc(ch))
    seg = x.seg
    c.has(ch, "uplift 씬파일러", f"**{sg(seg('thin_filer').uplift, 3)}**")
    c.has(ch, "uplift 전체", f"**{sg(seg('all').uplift, 3)}**")
    for key in ("both", "missing_only", "history_only"):
        c.has(ch, f"uplift {key}", f"`thin:{key}` {sg(seg(key).uplift, 3)}")
    c.has(ch, "uplift 일반", f"일반 {sg(seg('normal').uplift, 3)}")
    t1 = x.anova["test_1_feature_group"]
    c.has(ch, "F/η²", f"F={t1['f_statistic']:.2f}", f"η²={t1['effect_size_eta_squared']:.4f}")
    sens = x.sensitivity
    c.has(ch, "gender DI", f"DI {sens.loc[0.0].gender_di_ratio:.4f} → 0.3 에서 {sens.loc[0.3].gender_di_ratio:.4f}")
    c.has(ch, "age DI", f"min/max DI {sens.loc[0.0].age_group_di_min_max:.4f}, 0.3 에서 {sens.loc[0.3].age_group_di_min_max:.4f}")
    fr = x.fairness
    after = fr["mitigation"]["age_group"]
    c.has(ch, "EO FPR", f"{fr['before']['age_group']['equalized_odds']['equalized_odds_fpr_diff']:.4f} → {after['after']['equalized_odds']['equalized_odds_fpr_diff']:.4f}")
    c.has(ch, "승인율", f"{after['before']['approval_rate']:.4f} → {after['after']['approval_rate']:.4f}")
    best = x.models.groupby("feature_group").test_auc.max()
    c.has(ch, "그룹 최고", f"{best['traditional_only']:.4f} / {best['alternative_only']:.4f} / {best['combined']:.4f}")
    segs = [i for i in x.uplift.index if i not in ("all", "thin_filer")]
    rt, rc = x.uplift.loc[segs].auc_traditional, x.uplift.loc[segs].auc_combined
    c.has(ch, "세그먼트 편차", f"{rt.max() - rt.min():.4f} → {rc.max() - rc.min():.4f}")
    tk = {r["comparison"]: r for r in x.anova["test_2_algorithm"]["post_hoc_tukey"]}
    c.has(ch, "Tukey", f"mean_diff {sg(tk['lightgbm vs xgboost']['mean_diff'], 4)}, p_adj {tk['lightgbm vs xgboost']['p_adj']:.4f}")
    top5 = x.alt_top5.sum(1)
    types = x.sample.thin_filer_type.to_numpy()
    c.has(ch, "Top5 대안 수", f"`history_only` {top5[types == 'history_only'].mean():.2f} < `both` {top5[types == 'both'].mean():.2f}", line=False)
    share = sum(r["band"] for r in x.randomized_band["age_group"].values()) / len(x.ctx["test"])
    c.has(ch, "확률적 결정", f"{share * 100:.2f}%", line=False)
    c.has(ch, "uplift 범위", f"{sg(seg('history_only').uplift, 3)} ~ {sg(seg('both').uplift, 3)}",
          f"{seg('both').uplift / seg('history_only').uplift:.1f}배", line=False)




def check_chapter_1(c: Checker, x: Context) -> None:
    ch = "1"
    spec = Path("docs/SPEC.md").read_text(encoding="utf-8")
    c.true(ch, "SPEC 인용 1,100만", "1,100만 명" in spec)
    c.true(ch, "SPEC 인용 카카오뱅크 87%", "AUC 87%" in spec)
    seg = x.seg
    c.has(ch, "전통 AUC 일반", f"일반 고객 **{seg('normal').auc_traditional:.4f}**")
    c.has(ch, "전통 AUC 씬파일러", f"씬파일러 **{seg('thin_filer').auc_traditional:.4f}**")
    c.has(ch, "전통 AUC both", f"**{seg('both').auc_traditional:.4f}**")
    champ = x.champion
    c.has(ch, "성공기준 AUC", "Test AUC-ROC", f"**{champ.test_auc:.4f}**")
    c.has(ch, "성공기준 KS", "Test KS", f"**{champ.test_ks:.4f}**")
    c.has(ch, "성공기준 PSI", "PSI", f"**{champ.psi_train_test:.4f}**")
    c.has(ch, "성공기준 uplift", "씬파일러 AUC 향상", f"**{sg(seg('thin_filer').uplift, 4)}**")
    t1, t2 = x.anova["test_1_feature_group"], x.anova["test_2_algorithm"]
    def sci(value: float) -> str:
        mant, exp = f"{value:.2e}".split("e-")
        return f"{mant} × 10⁻{''.join('⁰¹²³⁴⁵⁶⁷⁸⁹'[int(d)] for d in exp.lstrip('0'))}"
    c.has(ch, "성공기준 ANOVA", sci(t1["p_value"]), sci(t2["p_value"]))
    c.has(ch, "성공기준 거절사유", f"**Top {x.cfg['explainer']['top_rejection_reasons']}**")
    fr = x.fairness["before"]["age_group"]
    c.has(ch, "성공기준 공정성", f"min/max DI {fr['disparate_impact']['di_ratio_min_max']:.4f}",
          f"FPR 차이 {fr['equalized_odds']['equalized_odds_fpr_diff']:.4f}")
    c.has(ch, "성공기준 테스트", f"**{_collected_tests()['합계']}개, 100% 통과**")
    c.has(ch, "임계값 각주", f"승인 임계값 {x.cfg['thresholds']['approval_threshold']:.2f}")


def check_chapter_2(c: Checker, x: Context) -> None:
    ch = "2"
    ports = x.cfg["ports"]
    for name, port in (("api", ports["api"]), ("dashboard", ports["dashboard"]), ("mlflow", ports["mlflow"])):
        c.has(ch, f"포트 {name}", f"**{name}** |" if name != "mlflow" else "**mlflow** |", str(port))
    groups = {"src/data/": None, "src/models/": None, "src/explainer/": None, "src/fairness/": None,
              "src/statistics/": None, "src/api/": None, "src/dashboard/": None, "src/analysis/": None}
    total = 0
    for prefix in groups:
        lines = sum(len(f.read_text(encoding="utf-8").splitlines())
                    for f in Path(prefix).glob("*.py"))
        groups[prefix] = lines
        total += lines
        c.has(ch, f"줄수 {prefix}", f"`{prefix}`", f"| {lines:,} |")
    config_lines = len(Path("src/config.py").read_text(encoding="utf-8").splitlines())
    total += config_lines
    c.has(ch, "줄수 config.py", "`src/config.py`", f"| {config_lines} |")
    c.has(ch, "줄수 합계", f"**{total:,}**")
    requirements = Path("requirements.txt").read_text(encoding="utf-8")
    pins = dict(re.findall(r"^([A-Za-z0-9_.-]+)==([0-9][^\s#]*)", requirements, re.M))
    for package, label in (("pandas", "pandas"), ("numpy", "numpy"), ("scipy", "scipy"),
                           ("scikit-learn", "scikit-learn"), ("xgboost", "XGBoost"), ("lightgbm", "LightGBM"),
                           ("imbalanced-learn", "imbalanced-learn"), ("shap", "SHAP"), ("fairlearn", "Fairlearn"),
                           ("statsmodels", "statsmodels"), ("mlflow", "MLflow"), ("fastapi", "FastAPI"),
                           ("uvicorn", "uvicorn"), ("pydantic", "Pydantic"), ("streamlit", "Streamlit"),
                           ("pytest", "pytest")):
        c.has(ch, f"버전 {package}", f"{label} {pins[package]}", line=False)
    app = Path("src/dashboard/app.py").read_text(encoding="utf-8")
    tabs = re.search(r"TAB_NAMES = \[(.*?)\]", app, re.S).group(1)
    for tab in re.findall(r'"([^"]+)"', tabs):
        c.has(ch, f"탭 {tab}", tab, line=False)
    colors = x.cfg["dashboard"]["risk_colors"]
    c.has(ch, "위험 색상", colors["low"], colors["medium"], colors["high"])
    c.has(ch, "테스트 수", f"테스트 {_collected_tests()['합계']}개")
    c.has(ch, "응답시간", "20.3ms", line=False)


def check_summary(c: Checker, x: Context) -> None:
    ch = "summary"
    champ, fr, seg = x.champion, x.fairness, x.seg
    c.has(ch, "AUC", f"**{champ.test_auc:.4f}**")
    c.has(ch, "KS", f"**{champ.test_ks:.4f}**")
    c.has(ch, "PSI", f"**{champ.psi_train_test:.4f}**")
    c.has(ch, "uplift", f"**{sg(seg('thin_filer').uplift, 4)}**", f"{sg(seg('both').uplift, 4)}")
    before, after = fr["before"], fr["mitigation"]
    c.has(ch, "완화 전 DI", f"{before['gender']['disparate_impact']['di_ratio']:.4f} / {before['age_group']['disparate_impact']['di_ratio']:.4f}",
          f"**{before['age_group']['disparate_impact']['di_ratio_min_max']:.4f}**")
    c.has(ch, "완화 전 FPR", f"**{before['age_group']['equalized_odds']['equalized_odds_fpr_diff']:.4f}**")
    after_di = {a: after[a]["after"]["disparate_impact"] for a in ("gender", "age_group")}
    c.has(ch, "완화 후 DI", f"{after_di['gender']['di_ratio']:.4f} / {after_di['age_group']['di_ratio']:.4f}",
          f"{after_di['age_group']['di_ratio_min_max']:.4f}")
    c.has(ch, "완화 후 FPR", f"**{after['age_group']['after']['equalized_odds']['equalized_odds_fpr_diff']:.4f}**")
    for key, test in (("검증 1", x.anova["test_1_feature_group"]), ("검증 2", x.anova["test_2_algorithm"])):
        c.has(ch, f"ANOVA {key}", f"F = {test['f_statistic']:.2f}", f"η² = {test['effect_size_eta_squared']:.4f}")
    tk = {r["comparison"]: r for r in x.anova["test_2_algorithm"]["post_hoc_tukey"]}
    c.has(ch, "Tukey", f"p_adj = {tk['lightgbm vs xgboost']['p_adj']:.4f}", line=False)
    for key in ("history_only", "missing_only"):
        c.has(ch, f"유형별 uplift {key}", sg(seg(key).uplift, 4), line=False)
    segs = [i for i in x.uplift.index if i not in ("all", "thin_filer")]
    rt, rc = x.uplift.loc[segs].auc_traditional, x.uplift.loc[segs].auc_combined
    c.has(ch, "세그먼트 편차", f"{rt.max() - rt.min():.4f} → {rc.max() - rc.min():.4f}")
    sens = x.sensitivity
    c.has(ch, "민감도", f"{sens.loc[0.0].gender_di_ratio:.4f}", f"{sens.loc[0.0].age_group_di_min_max:.4f}")
    share = sum(r["band"] for r in x.randomized_band["age_group"].values()) / len(x.ctx["test"])
    c.has(ch, "확률적 결정", f"{share * 100:.2f}%", line=False)
    # 핵심 발견 ①: 4장·ADR-001 의 상보성 판정 수치를 그대로 쓰는지
    sources = _norm(c.doc("4") + Path("docs/decisions/ADR-001-simulator.md").read_text(encoding="utf-8")
                    + (REPORT / "06_models.md").read_text(encoding="utf-8"))
    for value in ("0.9999", "0.9470", "0.9396", "0.8985", "0.8355", "0.8902"):
        c.has(ch, f"발견① {value}", value, line=False)
        c.true(ch, f"발견① {value} 4장·ADR-001·5장 일치", value in sources)
    # 핵심 발견 ②: 씬파일러 유형별 uplift 표
    for label, key in (("| 일반 |", "normal"), ("`history_only`", "history_only"),
                       ("`missing_only`", "missing_only"), ("**`both`**", "both")):
        row = seg(key)
        c.has(ch, f"발견② {key}", label, f"{row.auc_traditional:.4f}", f"{row.auc_combined:.4f}",
              sg(row.uplift, 4))
    # 핵심 발견 ③: 연령대 승인율 격차
    row = x.sensitivity.loc[x.cfg["simulator"]["bias_ratio"]]
    gap = (row["approval_age_group_55+"] - row["approval_age_group_20-34"]) * 100
    c.has(ch, "발견③ 승인율 격차", f"{gap:.1f}%p", line=False)
    # 요약의 결론 블록은 §5 확정 문장의 축약본이다(원문 그대로는 11.1절). 핵심 주장과 수치만 대조한다.
    narrative = _norm((ASSETS / "fairness_narrative.md").read_text(encoding="utf-8"))
    claims = (f"씬파일러 {sg(seg('thin_filer').uplift, 3)}, 전체 {sg(seg('all').uplift, 3)}",
              "측정 편향(성별)", "피처 내재 상관(연령대)", "사후 임계값 조정이 현실적 대응")
    for claim in claims:
        c.has(ch, f"결론 문구 {claim[:14]}", claim, line=False)
    c.true(ch, "결론 문구가 §5 서술과 같은 주장", "사후 임계값 조정(ThresholdOptimizer)이" in narrative
           or "사후 임계값 조정" in narrative)


def check_guide(c: Checker, x: Context) -> None:
    ch = "guide"
    c.has(ch, "테스트 수", f"부록 D ({_collected_tests()['합계']}개)")
    errors = len(re.findall(r"^\| \d+ \| \d+세션", c.doc("9"), re.M))
    c.has(ch, "오류 건수", f"**9장**에 {errors}건으로 정리했다")
    spec = Path("docs/SPEC.md").read_text(encoding="utf-8")
    items = spec.split("### 9. 동료 평가 질문 예시")[1].count("- [ ]")
    checklist = c.doc(ch).split("## 평가 체크리스트 대응")[1].split("## 재현과 검증")[0]
    rows = len(re.findall(r"^\| *\d+ \|", checklist, re.M))
    c.true(ch, f"체크리스트 {items}항목 전부 매핑", rows == items, f"표 {rows}행 vs 명세 {items}항목")


def check_appendix(c: Checker, x: Context) -> None:
    ch = "appendix"
    counts = _collected_tests()
    for name, n in counts.items():
        if name == "합계":
            c.has(ch, "테스트 합계", f"**{n}**")
        else:
            c.has(ch, f"테스트 {name}", f"`{name}`", f"{n}")
    adrs = sorted(Path("docs/decisions").glob("ADR-*.md"))
    c.true(ch, "ADR 12건", len(adrs) == 12, len(adrs))
    for path in adrs:
        c.has(ch, f"ADR 색인 {path.name[:7]}", f"| {path.name[4:7]} |")
    from fastapi.testclient import TestClient
    from src.api.main import app
    with TestClient(app) as client:
        paths = client.get("/openapi.json").json()["paths"]
    c.true(ch, "엔드포인트 7개", len(paths) == 7, len(paths))
    for route in paths:
        c.has(ch, f"명세 {route}", f"{route}`", line=False)      # 문서는 `GET /health` 형식
    tk = {r["comparison"]: r for r in x.anova["test_2_algorithm"]["post_hoc_tukey"]}
    c.has(ch, "ADR-012 p_adj", f"p_adj = {tk['lightgbm vs xgboost']['p_adj']:.4f}", line=False)
    c.has(ch, "random_state", f"random_state = {x.cfg['project']['random_state']}", line=False)


def _collected_tests() -> dict[str, int]:
    """pytest 가 실제로 수집하는 테스트 수 (파라미터화 포함)."""
    out = subprocess.run([sys.executable, "-m", "pytest", "--co", "-q"], capture_output=True, text=True)
    counts: dict[str, int] = {}
    for line in out.stdout.splitlines():
        if "::" in line:
            name = line.split("::")[0].split("/")[-1]
            counts[name] = counts.get(name, 0) + 1
    counts["합계"] = sum(counts.values())
    return counts

CHECKS = {
    "guide": check_guide, "summary": check_summary, "1": check_chapter_1, "2": check_chapter_2, "3": check_chapter_3,
    "6": check_chapter_6, "9": check_chapter_9, "10": check_chapter_10, "11": check_chapter_11,
    "appendix": check_appendix,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--chapters", nargs="*", default=list(CHECKS), help=f"대조할 장 ({', '.join(CHECKS)})")
    args = parser.parse_args()
    unknown = [ch for ch in args.chapters if ch not in CHECKS]
    if unknown:
        parser.error(f"대조 함수가 없는 장: {unknown}")

    checker, context = Checker(), Context()
    for chapter in args.chapters:
        if not checker.doc(chapter):
            print(f"{chapter}장: 문서 없음 — 건너뜀")
            continue
        CHECKS[chapter](checker, context)

    failed = 0
    for chapter in args.chapters:
        rows = [r for r in checker.results if r[0] == chapter]
        if not rows:
            continue
        ok = sum(r[2] for r in rows)
        print(f"{FILES[chapter]:<20} {ok:>3}/{len(rows):<3} 일치")
        for _, label, passed, subs in rows:
            if not passed:
                failed += 1
                print(f"    ✗ {label}  {subs}")
    total = len(checker.results)
    print(f"합계 {total - failed}/{total} 일치")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
