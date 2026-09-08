"""리포트용 표·그림 생성 (docs/report_assets/).

실행:
    python -m src.models.report_assets
전제: `python -m src.models.train` 이 끝나 data/processed/*.csv 가 있어야 한다.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)

from src.config import load_config, resolve_path
from src.data.preprocessor import FEATURE_GROUPS, feature_columns
from src.models.evaluate import auc_score, ks_statistic
from src.models.train import algorithms, build_estimator, load_splits

logger = logging.getLogger(__name__)

ASSETS_DIR = "docs/report_assets"
PALETTE = {"traditional_only": "#4C72B0", "alternative_only": "#DD8452", "combined": "#55A868"}

# 그림 안의 문자열은 영문으로 쓴다. 이 환경의 한글 폰트는 Windows 마운트(/mnt/c)에만 있어
# Docker 컨테이너에서 글리프가 깨진다. 설명은 리포트 본문(한글)에서 한다.
ALGO_LABEL = {"logistic_regression": "LogReg", "xgboost": "XGBoost", "lightgbm": "LightGBM"}
GROUP_LABEL = {
    "traditional_only": "Traditional only",
    "alternative_only": "Alternative only",
    "combined": "Combined",
}
SEGMENT_LABEL = {
    "all": "All", "thin_filer": "Thin filer", "normal": "Normal",
    "thin:missing_only": "Thin: missing only",
    "thin:history_only": "Thin: history only",
    "thin:both": "Thin: both",
}


def _assets_dir(config: dict[str, Any]) -> Path:
    path = resolve_path(ASSETS_DIR)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _style() -> None:
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams["figure.dpi"] = 120
    plt.rcParams["savefig.bbox"] = "tight"
    plt.rcParams["axes.unicode_minus"] = False


def feature_group_chart(results: pd.DataFrame, out: Path) -> Path:
    """피처 그룹별 test AUC 비교 (알고리즘 × 불균형 처리)."""
    data = results.copy()
    data["combo"] = (
        data["algorithm"].map(ALGO_LABEL) + "\n" + data["imbalance"].str.replace("_", " ")
    )
    data["group"] = data["feature_group"].map(GROUP_LABEL)
    order = (
        data.groupby("combo")["test_auc"].max().sort_values(ascending=False).index.tolist()
    )
    fig, ax = plt.subplots(figsize=(13, 6.5))
    sns.barplot(
        data=data, x="combo", y="test_auc", hue="group", order=order,
        hue_order=[GROUP_LABEL[g] for g in FEATURE_GROUPS],
        palette={GROUP_LABEL[k]: v for k, v in PALETTE.items()}, ax=ax,
    )
    ax.axhline(0.78, ls="--", c="crimson", lw=1.5, label="SPEC target 0.78")
    ax.set_ylim(0.5, 1.05)
    ax.set_xlabel("")
    ax.set_ylabel("Test AUC-ROC")
    ax.set_title("Test AUC by feature group")
    ax.legend(loc="lower left", fontsize=10, ncol=2, framealpha=0.9)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", fontsize=7.5, padding=2)
    ax.tick_params(axis="x", labelsize=11)
    path = out / "feature_group_auc.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def cv_distribution_chart(folds: pd.DataFrame, out: Path) -> Path:
    """fold 별 AUC 분포 — ANOVA 관측치의 산포를 보여준다."""
    data = folds.copy()
    data["group"] = data["feature_group"].map(GROUP_LABEL)
    data["algo"] = data["algorithm"].map(ALGO_LABEL)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    sns.boxplot(data=data, x="group", y="auc", order=[GROUP_LABEL[g] for g in FEATURE_GROUPS],
                palette={GROUP_LABEL[k]: v for k, v in PALETTE.items()}, ax=axes[0])
    axes[0].set_title("ANOVA test 1: fold AUC by feature group (n=120)")
    axes[0].set_xlabel(""); axes[0].set_ylabel("fold AUC")
    sns.boxplot(data=data, x="algo", y="auc", ax=axes[1], color="#8C8C8C")
    axes[1].set_title("ANOVA test 2: fold AUC by algorithm (n=120)")
    axes[1].set_xlabel(""); axes[1].set_ylabel("")
    for ax in axes:
        ax.tick_params(axis="x", labelrotation=10, labelsize=11)
    path = out / "cv_fold_auc_distribution.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def uplift_chart(uplift: pd.DataFrame, out: Path) -> Path:
    """세그먼트별 traditional vs combined AUC 와 uplift."""
    data = uplift.copy()
    data["seg"] = data["segment"].map(SEGMENT_LABEL).fillna(data["segment"])
    data["algo"] = data["algorithm"].map(ALGO_LABEL)
    seg_order = [SEGMENT_LABEL[s] for s in SEGMENT_LABEL if SEGMENT_LABEL[s] in set(data["seg"])]

    fig, axes = plt.subplots(1, 2, figsize=(17, 6.5))
    long = data.melt(
        id_vars=["algo", "seg"], value_vars=["auc_traditional", "auc_combined"],
        var_name="model", value_name="auc",
    ).replace({"auc_traditional": "Traditional", "auc_combined": "Combined"})
    sns.barplot(data=long, x="seg", y="auc", hue="model", order=seg_order, ax=axes[0],
                palette={"Traditional": "#4C72B0", "Combined": "#55A868"})
    axes[0].set_title("AUC by segment (traditional vs combined)")
    axes[0].set_xlabel(""); axes[0].set_ylabel("Test AUC"); axes[0].set_ylim(0.5, 1.0)
    axes[0].legend(fontsize=10, loc="lower right")

    sns.barplot(data=data, x="seg", y="uplift", hue="algo", order=seg_order, ax=axes[1])
    axes[1].axhline(0.03, ls="--", c="crimson", lw=1.5, label="SPEC target +0.03")
    axes[1].set_title("Alternative-data AUC uplift")
    axes[1].set_xlabel(""); axes[1].set_ylabel("uplift")
    axes[1].legend(fontsize=10)
    for ax in axes:
        ax.tick_params(axis="x", labelrotation=20, labelsize=10)
        for label in ax.get_xticklabels():
            label.set_ha("right")
    path = out / "thin_filer_uplift.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def curves_and_confusion(config: dict[str, Any], out: Path) -> list[Path]:
    """최우수 조합 기준 ROC / PR 곡선과 혼동행렬."""
    frames = load_splits(config)
    target = config["data"]["target_column"]
    threshold = config["thresholds"]["approval_threshold"]
    train, test = frames["train"], frames["test"]
    pos_weight = float((train[target] == 0).sum() / max((train[target] == 1).sum(), 1))

    scores: dict[str, np.ndarray] = {}
    best_algo = "lightgbm" if "lightgbm" in algorithms(config) else algorithms(config)[0]
    for group in FEATURE_GROUPS:
        cols = feature_columns(config, group)
        model = build_estimator(best_algo, "class_weight", config, pos_weight)
        model.fit(train[cols], train[target])
        scores[group] = model.predict_proba(test[cols])[:, 1]

    y = test[target].to_numpy()
    paths = []

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    for group, score in scores.items():
        fpr, tpr, _ = roc_curve(y, score)
        axes[0].plot(
            fpr, tpr, color=PALETTE[group], lw=2,
            label=f"{GROUP_LABEL[group]} (AUC {auc_score(y, score):.3f}, "
                  f"KS {ks_statistic(y, score):.3f})",
        )
        precision, recall, _ = precision_recall_curve(y, score)
        axes[1].plot(recall, precision, color=PALETTE[group], lw=2, label=GROUP_LABEL[group])
    axes[0].plot([0, 1], [0, 1], ls=":", c="gray")
    axes[0].set_xlabel("False Positive Rate"); axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title(f"ROC Curve ({ALGO_LABEL[best_algo]}, class_weight)")
    axes[0].legend(fontsize=10, loc="lower right")
    axes[1].axhline(y.mean(), ls=":", c="gray", label=f"base rate {y.mean():.3f}")
    axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall Curve")
    axes[1].legend(fontsize=10)
    path = out / "roc_pr_curves.png"
    fig.savefig(path); plt.close(fig); paths.append(path)

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    for ax, (group, score) in zip(axes, scores.items()):
        cm = confusion_matrix(y, (score >= threshold).astype(int))
        ConfusionMatrixDisplay(cm, display_labels=["Good", "Default"]).plot(
            ax=ax, cmap="Blues", colorbar=False, values_format=","
        )
        ax.set_title(f"{GROUP_LABEL[group]}\n(threshold {threshold})")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual" if group == FEATURE_GROUPS[0] else "")
    fig.suptitle(
        f"Confusion matrix ({ALGO_LABEL[best_algo]}, class_weight, threshold {threshold})", y=1.03
    )
    path = out / "confusion_matrices.png"
    fig.savefig(path); plt.close(fig); paths.append(path)
    return paths


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = load_config()
    _style()
    out = _assets_dir(cfg)
    processed = resolve_path(cfg["preprocess"]["output_dir"])

    results = pd.read_csv(processed / "model_results.csv")
    folds = pd.read_csv(processed / "fold_auc.csv")
    uplift = pd.read_csv(processed / "thin_filer_uplift.csv")

    made = [
        feature_group_chart(results, out),
        cv_distribution_chart(folds, out),
        uplift_chart(uplift, out),
        *curves_and_confusion(cfg, out),
    ]
    for path in made:
        logger.info("생성: %s", path.relative_to(resolve_path(".")))


if __name__ == "__main__":
    main()
