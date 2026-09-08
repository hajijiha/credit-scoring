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
    for feature in ecfg["dependence_features"]:
        paths.append(explainer.dependence_plot(
            feature, sample, out / f"shap_dependence_{feature}.png"
        ))

    logger.info("전역 중요도 상위 %d개:", ecfg["summary_max_display"])
    for _, row in importance.head(ecfg["summary_max_display"]).iterrows():
        logger.info("  %-38s %-12s %.5f", row["feature"], row["feature_type"],
                    row["mean_abs_shap"])
    share = importance.head(ecfg["summary_max_display"])["feature_type"].value_counts().to_dict()
    logger.info("상위 %d개 구성: %s", ecfg["summary_max_display"], share)

    # --- 지역 해석: 씬파일러 / 일반 각각 거절된 고객 1명 ---
    proba = ctx["model"].predict_proba(test[cols])[:, 1]
    test = test.assign(_proba=proba)
    test_raw = test_raw.assign(_proba=proba)

    samples: dict[str, Any] = {}
    for label, mask in (
        ("thin_filer", test["is_thin_filer"] == 1),
        ("normal", test["is_thin_filer"] == 0),
    ):
        rejected = test[mask & (test["_proba"] >= threshold)]
        if rejected.empty:
            logger.warning("%s 집단에 거절 사례가 없습니다.", label)
            continue
        idx = rejected["_proba"].idxmax()
        row_scaled = test.loc[idx]
        row_raw = test_raw.loc[idx]
        age_group = row_raw["age_group"]

        paths.append(explainer.waterfall_plot(
            row_scaled, out / f"shap_waterfall_{label}.png",
            title=f"{label} (p_default={row_scaled['_proba']:.3f}, age_group={age_group})",
        ))
        # SHAP 은 스케일된 입력으로, 설명 문장은 원 단위 값으로 만든다.
        reasons = explainer.rejection_reasons(row_scaled, age_group, display_row=row_raw)
        samples[label] = {
            "index": int(idx),
            "segment": label,
            "thin_filer_type": str(row_raw.get("thin_filer_type", "none")),
            "age_group": str(age_group),
            "gender": str(row_raw["gender"]),
            "actual_default": int(row_scaled[target]),
            "default_probability": round(float(row_scaled["_proba"]), 4),
            "credit_score": int(round(cfg["scoring"]["score_max"] * (1 - row_scaled["_proba"]))),
            "approval_status": "REJECTED",
            "rejection_reasons": reasons,
            "total_rejection_factors": len(reasons),
        }
        logger.info("%s 거절 사례 idx=%s p_default=%.4f", label, idx, row_scaled["_proba"])
        for reason in reasons:
            logger.info("  %d. %s", reason["rank"], reason["explanation"])

    (out / "rejection_reasons_sample.json").write_text(
        json.dumps(samples, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    copied = copy_to_report_assets(paths, cfg)
    logger.info("생성 %d개 / 리포트 자료 복사 %d개", len(paths), len(copied))
    for path in paths:
        logger.info("  %s", path.relative_to(resolve_path(".")))


if __name__ == "__main__":
    main()
