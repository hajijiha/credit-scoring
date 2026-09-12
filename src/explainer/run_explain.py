"""설명가능성 산출물 생성 (SPEC §3).

실행:
    python -m src.explainer.run_explain
전제: `python -m src.models.train` 으로 챔피언 모델이 저장돼 있어야 한다.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src.config import load_config, resolve_path
from src.data.preprocessor import Preprocessor, feature_columns
from src.explainer.explainer import CreditExplainer, copy_to_report_assets
from src.models.train import champion_combination

logger = logging.getLogger(__name__)


def _risk_grade(config: dict[str, Any], p_default: float) -> str:
    """ADR-003: score = 1000 x (1 - p_default) 에 등급 컷 적용."""
    score = config["scoring"]["score_max"] * (1 - p_default)
    for grade, cutoff in sorted(
        config["scoring"]["grade_cutoffs"].items(), key=lambda kv: -kv[1]
    ):
        if score >= cutoff:
            return grade
    return "E"


def load_champion(config: dict[str, Any]) -> tuple[Any, tuple[str, str, str]]:
    key = champion_combination(config)
    path = resolve_path(config["paths"]["models"]) / (
        f"champion_{key[0]}_{config['models']['model_version']}.pkl"
    )
    if not path.exists():
        raise FileNotFoundError(f"{path} 가 없습니다. 먼저 학습을 실행하세요.")
    return joblib.load(path), key


def build_explainer(config: dict[str, Any] | None = None) -> tuple[CreditExplainer, dict[str, Any]]:
    """챔피언 모델 + 원 단위 참조 데이터로 설명기를 만든다."""
    cfg = config if config is not None else load_config()
    model, key = load_champion(cfg)
    cols = feature_columns(cfg, key[1])

    processed = resolve_path(cfg["preprocess"]["output_dir"])
    pre = Preprocessor.load(processed / cfg["preprocess"]["scaler_file"])
    train = pd.read_parquet(processed / "train.parquet")
    test = pd.read_parquet(processed / "test.parquet")

    # 참조 평균은 원 단위로 계산해야 자연어 설명이 읽힌다.
    train_raw = pre.inverse_transform_numeric(train)
    test_raw = pre.inverse_transform_numeric(test)

    explainer = CreditExplainer(model, cols, train_raw, cfg)
    context = {
        "model": model, "key": key, "columns": cols, "preprocessor": pre,
        "train": train, "test": test, "train_raw": train_raw, "test_raw": test_raw,
    }
    return explainer, context


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = load_config()
    explainer, ctx = build_explainer(cfg)
    ecfg = cfg["explainer"]
    target = cfg["data"]["target_column"]
    threshold = cfg["thresholds"]["approval_threshold"]

    out = resolve_path(ecfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)

    test, test_raw, cols = ctx["test"], ctx["test_raw"], ctx["columns"]
    sample = test.sample(
        n=min(ecfg["background_sample_size"], len(test)),
        random_state=cfg["project"]["random_state"],
    )
    logger.info("챔피언 %s | SHAP 표본 %s행", "|".join(ctx["key"]), f"{len(sample):,}")

    paths: list[Path] = []

    # --- 전역 해석 ---
    importance = explainer.global_importance(sample)
    importance.to_csv(out / "global_importance.csv", index=False)
    paths.append(explainer.summary_plot(sample, out / "shap_summary.png"))
    clips = ecfg.get("dependence_axis_clip") or {}
    for feature in ecfg["dependence_features"]:
        clip = clips.get(feature)
        paths.append(explainer.dependence_plot(
            feature, sample, out / f"shap_dependence_{feature}.png",
            display=test_raw.loc[sample.index] if clip is not None else None,
            clip_percentile=clip,
        ))

    logger.info("전역 중요도 상위 %d개:", ecfg["summary_max_display"])
    for _, row in importance.head(ecfg["summary_max_display"]).iterrows():
        logger.info("  %-38s %-12s %.5f", row["feature"], row["feature_type"],
                    row["mean_abs_shap"])
    share = importance.head(ecfg["summary_max_display"])["feature_type"].value_counts().to_dict()
    logger.info("상위 %d개 구성: %s", ecfg["summary_max_display"], share)

    # --- 지역 해석 ---
    proba = ctx["model"].predict_proba(test[cols])[:, 1]
    test = test.assign(_proba=proba)
    test_raw = test_raw.assign(_proba=proba)

    samples: dict[str, Any] = {}
    for spec in ecfg["samples"]:
        mask = pd.Series(True, index=test.index)
        if spec["segment"] == "thin_filer":
            mask &= test["is_thin_filer"] == 1
        elif spec["segment"] == "normal":
            mask &= test["is_thin_filer"] == 0
        mask &= test["_proba"].between(spec["proba_min"], spec["proba_max"], inclusive="left")

        pool = test[mask]
        if pool.empty:
            logger.warning("%s: 조건에 맞는 사례가 없습니다 (p %.2f~%.2f, %s)",
                           spec["name"], spec["proba_min"], spec["proba_max"], spec["segment"])
            continue

        rejected = spec["decision"] == "rejected"
        # 구간 중앙에 가장 가까운 건을 고른다(경계 사례의 대표성).
        centre = (spec["proba_min"] + min(spec["proba_max"], 1.0)) / 2
        idx = (pool["_proba"] - centre).abs().idxmin()
        row_scaled, row_raw = test.loc[idx], test_raw.loc[idx]
        age_group = row_raw["age_group"]
        p_default = float(row_scaled["_proba"])

        explainer.waterfall_plot(
            row_scaled, out / f"shap_waterfall_{spec['name']}.png",
            title=f"{spec['name']} (p_default={p_default:.4f}, age_group={age_group})",
        )
        paths.append(out / f"shap_waterfall_{spec['name']}.png")

        factors = explainer.contributing_factors(
            row_scaled, age_group, adverse=rejected, display_row=row_raw
        )
        entry: dict[str, Any] = {
            "sample_name": spec["name"],
            "index": int(idx),
            "segment": "thin_filer" if row_scaled["is_thin_filer"] == 1 else "normal",
            "thin_filer_type": str(row_raw.get("thin_filer_type", "none")),
            "age_group": str(age_group),
            "gender": str(row_raw["gender"]),
            "actual_default": int(row_scaled[target]),
            "default_probability": round(p_default, 4),
            "credit_score": int(round(cfg["scoring"]["score_max"] * (1 - p_default))),
            "risk_grade": _risk_grade(cfg, p_default),
            "approval_status": "REJECTED" if rejected else "APPROVED",
        }
        if rejected:
            entry["rejection_reasons"] = factors
            entry["total_rejection_factors"] = len(factors)
        else:
            entry["approval_factors"] = factors
        samples[spec["name"]] = entry

        logger.info("%s: idx=%s p_default=%.4f (%s, %s)",
                    spec["name"], idx, p_default, entry["segment"], entry["approval_status"])
        for factor in factors:
            logger.info("  %d. %s", factor["rank"], factor["explanation"])

    (out / "rejection_reasons_sample.json").write_text(
        json.dumps(samples, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    copied = copy_to_report_assets(paths, cfg)
    logger.info("생성 %d개 / 리포트 자료 복사 %d개", len(paths), len(copied))
    for path in paths:
        logger.info("  %s", path.relative_to(resolve_path(".")))


if __name__ == "__main__":
    main()
