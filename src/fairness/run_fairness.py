"""공정성 검증 · Mitigation 실행 및 리포트 자료 생성 (SPEC §4).

실행:
    python -m src.fairness.run_fairness
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.config import load_config, resolve_path
from src.data.preprocessor import feature_columns
from src.explainer.explainer import copy_to_report_assets
from src.explainer.run_explain import load_champion
from src.fairness.metrics import approval_mask, evaluate
from src.fairness.mitigation import run_mitigation

logger = logging.getLogger(__name__)

GROUP_LABEL = {"male": "Male", "female": "Female", "20-34": "20-34", "35-54": "35-54", "55+": "55+"}


def _style() -> None:
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams["figure.dpi"] = 120
    plt.rcParams["savefig.bbox"] = "tight"
    plt.rcParams["axes.unicode_minus"] = False


def approval_rate_chart(before: dict[str, Any], out: Path, config: dict[str, Any]) -> Path:
    """집단별 승인율 막대 차트 + DI Ratio 표시 (대시보드 공정성 탭 자료)."""
    attributes = config["fairness"]["sensitive_attributes"]
    fig, axes = plt.subplots(1, len(attributes), figsize=(7 * len(attributes), 6))
    axes = np.atleast_1d(axes)
    for ax, attribute in zip(axes, attributes):
        di = before[attribute]["disparate_impact"]
        rates = pd.Series(di["approval_rates"]).sort_index()
        colors = ["#C44E52" if g == di["disadvantaged_group"] else "#4C72B0" for g in rates.index]
        ax.bar([GROUP_LABEL.get(g, g) for g in rates.index], rates.to_numpy(), color=colors)
        for i, value in enumerate(rates.to_numpy()):
            ax.text(i, value + 0.01, f"{value:.3f}", ha="center", fontsize=11)
        ax.set_ylim(0, min(1.05, rates.max() * 1.25))
        ax.set_ylabel("Approval rate")
        ax.set_title(f"{attribute}  (DI ratio {di['di_ratio']:.3f}, "
                     f"{'PASS' if di['passes'] else 'FAIL'})")
    fig.suptitle("Approval rate by protected group (before mitigation)", y=1.02)
    path = out / "fairness_approval_rates.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def mitigation_chart(results: dict[str, Any], out: Path) -> Path:
    """Mitigation 전/후 비교 차트."""
    rows = []
    for attribute, res in results.items():
        for phase in ("before", "after"):
            rows.append({
                "attribute": attribute, "phase": phase,
                "DI ratio": res[phase]["disparate_impact"]["di_ratio"],
                "EO TPR diff": res[phase]["equalized_odds"]["equalized_odds_tpr_diff"],
                "EO FPR diff": res[phase]["equalized_odds"]["equalized_odds_fpr_diff"],
                "Approval rate": res[phase]["approval_rate"],
            })
    data = pd.DataFrame(rows)

    metrics = ["DI ratio", "EO TPR diff", "EO FPR diff", "Approval rate"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(5.5 * len(metrics), 6))
    guides = {"DI ratio": (0.8, 1.25), "EO TPR diff": (None, 0.10), "EO FPR diff": (None, 0.10)}
    for ax, metric in zip(axes, metrics):
        sns.barplot(data=data, x="attribute", y=metric, hue="phase", ax=ax,
                    palette={"before": "#4C72B0", "after": "#55A868"})
        for line in guides.get(metric, ()):
            if line is not None:
                ax.axhline(line, ls="--", c="crimson", lw=1.3)
        for container in ax.containers:
            ax.bar_label(container, fmt="%.3f", fontsize=9, padding=2)
        ax.set_title(metric)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.legend(fontsize=9)
    fig.suptitle("Fairness metrics before / after ThresholdOptimizer (equalized_odds)", y=1.03)
    path = out / "fairness_mitigation_comparison.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = load_config()
    _style()

    model, key = load_champion(cfg)
    cols = feature_columns(cfg, key[1])
    processed = resolve_path(cfg["preprocess"]["output_dir"])
    valid = pd.read_parquet(processed / "valid.parquet")
    test = pd.read_parquet(processed / "test.parquet")
    target = cfg["data"]["target_column"]

    proba = model.predict_proba(test[cols])[:, 1]
    y_test = test[target].to_numpy()

    logger.info("=== 완화 전 공정성 (챔피언 %s, test %s행) ===",
                "|".join(key), f"{len(test):,}")
    before = evaluate(y_test, proba, test, cfg)

    logger.info("=== Mitigation (ThresholdOptimizer, valid 로 적합 / test 로 평가) ===")
    mitigation = run_mitigation(model, cols, valid, test, cfg)

    out = resolve_path(cfg["explainer"]["output_dir"]).parent / "fairness"
    out.mkdir(parents=True, exist_ok=True)
    (out / "fairness_results.json").write_text(
        json.dumps({"before": before, "mitigation": mitigation}, ensure_ascii=False, indent=2,
                   default=str),
        encoding="utf-8",
    )

    paths = [approval_rate_chart(before, out, cfg), mitigation_chart(mitigation, out)]
    copy_to_report_assets(paths, cfg)
    logger.info("결과 저장: %s", out / "fairness_results.json")
    for path in paths:
        logger.info("  %s", path.relative_to(resolve_path(".")))

    # 핵심 논점: 보호 속성을 학습에서 뺐는데도 편향이 나타나는가?
    logger.info("=== 핵심 검증: proxy discrimination ===")
    logger.info("전체 승인율 %.4f", before["overall_approval_rate"])
    for attribute in cfg["fairness"]["sensitive_attributes"]:
        di = before[attribute]["disparate_impact"]
        eo = before[attribute]["equalized_odds"]
        logger.info(
            "  %-10s DI %.4f (%s) | min/max %.4f | EO TPR차 %.4f FPR차 %.4f (%s)",
            attribute, di["di_ratio"], "충족" if di["passes"] else "위반",
            di["di_ratio_min_max"], eo["equalized_odds_tpr_diff"],
            eo["equalized_odds_fpr_diff"], "충족" if eo["passes"] else "위반",
        )
        logger.info("    집단별 승인율: %s",
                    {k: round(v, 4) for k, v in di["approval_rates"].items()})


if __name__ == "__main__":
    main()
