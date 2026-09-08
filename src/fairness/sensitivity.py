"""bias_ratio 민감도 분석 (SPEC §4 실증).

SPEC §4 는 "시뮬레이터 함수에 `bias_ratio` 파라미터가 존재하고, 이 값을 조정하면
성별/연령대별 승인율 편향이 변경된다"를 요구한다. 이를 수치로 보인다.

각 bias_ratio 수준마다 데이터를 재생성하고 챔피언 조합을 다시 학습해
집단별 승인율과 DI Ratio 를 산출한다(전체 18조합 재학습이 아니라 챔피언 1개만).

실행:
    python -m src.fairness.sensitivity
"""

from __future__ import annotations

import copy
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
from src.data.loader import load_gmsc
from src.data.preprocessor import feature_columns, prepare
from src.data.simulator import simulate
from src.explainer.explainer import copy_to_report_assets
from src.fairness.metrics import approval_mask, disparate_impact
from src.models.evaluate import auc_score
from src.models.train import build_estimator, champion_combination

logger = logging.getLogger(__name__)

BIAS_LEVELS = (0.0, 0.1, 0.2, 0.3)


def run(config: dict[str, Any] | None = None,
        levels: tuple[float, ...] = BIAS_LEVELS) -> pd.DataFrame:
    cfg = config if config is not None else load_config()
    target = cfg["data"]["target_column"]
    key = champion_combination(cfg)
    cols = feature_columns(cfg, key[1])
    base = load_gmsc(cfg)

    rows: list[dict[str, Any]] = []
    for ratio in levels:
        level_cfg = copy.deepcopy(cfg)
        level_cfg["simulator"]["bias_ratio"] = ratio
        data = simulate(base, level_cfg)
        processed, _ = prepare(data, level_cfg, save=False)
        (x_train, y_train), (x_test, y_test) = processed["train"], processed["test"]

        pos_weight = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
        model = build_estimator(key[0], key[2], level_cfg, pos_weight)
        model.fit(x_train[cols], y_train)
        proba = model.predict_proba(x_test[cols])[:, 1]

        threshold = level_cfg["thresholds"]["approval_threshold"]
        entry: dict[str, Any] = {
            "bias_ratio": ratio,
            "shift_per_group": ratio * level_cfg["simulator"]["bias_shift_per_ratio"],
            "test_auc": auc_score(y_test, proba),
            "overall_approval_rate": float(approval_mask(proba, threshold).mean()),
        }
        for attribute in level_cfg["fairness"]["sensitive_attributes"]:
            di = disparate_impact(
                proba, x_test[attribute].rename(attribute), threshold,
                level_cfg["fairness"]["disadvantaged_groups"][attribute], level_cfg,
            )
            entry[f"{attribute}_di_ratio"] = di["di_ratio"]
            entry[f"{attribute}_di_min_max"] = di["di_ratio_min_max"]
            for group, rate in di["approval_rates"].items():
                entry[f"approval_{attribute}_{group}"] = float(rate)
        rows.append(entry)
        logger.info(
            "bias_ratio=%.1f (하향 %.3f) | AUC %.4f | 승인율 %.4f | "
            "gender DI %.4f | age_group DI %.4f (min/max %.4f)",
            ratio, entry["shift_per_group"], entry["test_auc"], entry["overall_approval_rate"],
            entry["gender_di_ratio"], entry["age_group_di_ratio"], entry["age_group_di_min_max"],
        )
    return pd.DataFrame(rows)


def chart(table: pd.DataFrame, out: Path) -> Path:
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams["savefig.bbox"] = "tight"
    fig, axes = plt.subplots(1, 3, figsize=(19, 6))

    gender_cols = [c for c in table.columns if c.startswith("approval_gender_")]
    for column in sorted(gender_cols):
        axes[0].plot(table["bias_ratio"], table[column], marker="o",
                     label=column.replace("approval_gender_", ""))
    axes[0].set_title("Approval rate by gender")

    age_cols = [c for c in table.columns if c.startswith("approval_age_group_")]
    for column in sorted(age_cols):
        axes[1].plot(table["bias_ratio"], table[column], marker="o",
                     label=column.replace("approval_age_group_", ""))
    axes[1].set_title("Approval rate by age group")

    for attribute, style in (("gender", "-o"), ("age_group", "-s")):
        axes[2].plot(table["bias_ratio"], table[f"{attribute}_di_ratio"], style,
                     label=f"{attribute} (protected/rest)")
        axes[2].plot(table["bias_ratio"], table[f"{attribute}_di_min_max"], style, ls="--",
                     alpha=0.6, label=f"{attribute} (min/max)")
    axes[2].axhline(0.8, ls="--", c="crimson", lw=1.4, label="4/5 rule 0.8")
    axes[2].set_title("Disparate Impact ratio")

    for ax in axes:
        ax.set_xlabel("bias_ratio")
        ax.set_xticks(list(table["bias_ratio"]))
        ax.legend(fontsize=10)
    axes[0].set_ylabel("Approval rate")
    axes[2].set_ylabel("DI ratio")
    fig.suptitle("Sensitivity to bias_ratio (champion model retrained at each level)", y=1.02)
    path = out / "bias_ratio_sensitivity.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = load_config()
    table = run(cfg)
    out = resolve_path(cfg["explainer"]["output_dir"]).parent / "fairness"
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "bias_ratio_sensitivity.csv", index=False)
    (out / "bias_ratio_sensitivity.json").write_text(
        json.dumps(table.to_dict(orient="records"), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    path = chart(table, out)
    copy_to_report_assets([path], cfg)
    logger.info("저장: %s", out / "bias_ratio_sensitivity.csv")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
