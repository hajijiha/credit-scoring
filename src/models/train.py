"""모델 학습 · 비교 (SPEC §2, ADR-005).

조합: 알고리즘 3 × 피처 그룹 3 × 불균형 처리 2 = 18개.
각 조합에 RepeatedStratifiedKFold(n_splits, n_repeats) 를 돌려 fold 별 AUC 를 모두 남긴다.
이 fold AUC 들이 ANOVA 검증 1(피처 그룹) / 검증 2(알고리즘)의 관측치가 된다.

실행:
    python -m src.models.train --repeats 1     # 스모크
    python -m src.models.train                 # 전체 (config 의 cv.n_repeats 사용)
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold
from xgboost import XGBClassifier

from src.config import load_config, resolve_path
from src.data.preprocessor import FEATURE_GROUPS, feature_columns
from src.models.evaluate import auc_score, classification_metrics, psi

logger = logging.getLogger(__name__)




def _model_cfg(config: dict[str, Any]) -> dict[str, Any]:
    return config["models"]


def algorithms(config: dict[str, Any]) -> tuple[str, ...]:
    return tuple(_model_cfg(config)["algorithms"])


def imbalance_methods(config: dict[str, Any]) -> tuple[str, ...]:
    return tuple(_model_cfg(config)["imbalance_methods"])


def build_estimator(
    algorithm: str, imbalance: str, config: dict[str, Any], scale_pos_weight: float
):
    """알고리즘 + 불균형 처리 조합의 추정기. 하이퍼파라미터는 config.models 에서 읽는다.

    class_weight: 알고리즘 자체의 가중치 옵션 사용
    smote: 학습 fold 에만 SMOTE 를 적용하는 imblearn 파이프라인 (누수 방지)
    """
    mcfg = _model_cfg(config)
    seed = config["project"]["random_state"]
    n_jobs = mcfg["n_jobs"]
    balanced = imbalance == "class_weight"

    if algorithm == "logistic_regression":
        params = dict(mcfg["logistic_regression"])
        model = LogisticRegression(
            class_weight="balanced" if balanced else None,
            random_state=seed,
            **params,
        )
    elif algorithm == "xgboost":
        params = dict(mcfg["xgboost"])
        model = XGBClassifier(
            scale_pos_weight=scale_pos_weight if balanced else 1.0,
            eval_metric="auc", random_state=seed, n_jobs=n_jobs, verbosity=0,
            **params,
        )
    elif algorithm == "lightgbm":
        params = dict(mcfg["lightgbm"])
        model = LGBMClassifier(
            class_weight="balanced" if balanced else None,
            random_state=seed, n_jobs=n_jobs, verbose=-1,
            **params,
        )
    else:
        raise ValueError(f"알 수 없는 알고리즘: {algorithm}")

    if imbalance == "smote":
        return ImbPipeline([("smote", SMOTE(random_state=seed)), ("model", model)])
    return model


def cross_validate_auc(
    estimator_factory, x: pd.DataFrame, y: pd.Series, n_splits: int, n_repeats: int, seed: int
) -> list[float]:
    """RepeatedStratifiedKFold fold 별 AUC 목록 (ADR-005: 그룹당 n = splits × repeats)."""
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    scores: list[float] = []
    for train_idx, valid_idx in cv.split(x, y):
        model = estimator_factory()
        model.fit(x.iloc[train_idx], y.iloc[train_idx])
        proba = model.predict_proba(x.iloc[valid_idx])[:, 1]
        scores.append(auc_score(y.iloc[valid_idx], proba))
    return scores


def _setup_mlflow(config: dict[str, Any]) -> None:
    """트래킹 URI·실험을 준비한다. 아티팩트 위치는 config 의 상대경로를 쓴다."""
    mcfg = config["mlflow"]
    mlflow.set_tracking_uri(mcfg["tracking_uri"])
    name = mcfg["experiment_name"]
    if mlflow.get_experiment_by_name(name) is None:
        artifact_location = resolve_path(mcfg["artifact_location"]).as_uri()
        mlflow.create_experiment(name, artifact_location=artifact_location)
    mlflow.set_experiment(name)


def _log_sklearn_model(model: Any, name: str = "model") -> None:
    """MLflow 에 모델을 기록한다.

    MLflow 3.x 기본 직렬화(skops)는 imblearn 파이프라인 타입을 거부하므로 cloudpickle 로 고정한다.
    """
    kwargs = {"serialization_format": mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE}
    try:
        mlflow.sklearn.log_model(model, name=name, **kwargs)
    except TypeError:                              # 구버전 시그니처 호환
        mlflow.sklearn.log_model(model, artifact_path=name, **kwargs)


def load_splits(config: dict[str, Any]) -> dict[str, pd.DataFrame]:
    """전처리된 parquet 을 읽는다. 없으면 안내 후 예외."""
    out_dir = resolve_path(config["preprocess"]["output_dir"])
    frames = {}
    for name in ("train", "valid", "test"):
        path = out_dir / f"{name}.parquet"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} 가 없습니다. 먼저 `python -m src.data.preprocessor` 를 실행하세요."
            )
        frames[name] = pd.read_parquet(path)
    return frames


def run(config: dict[str, Any] | None = None, n_repeats: int | None = None,
        log_mlflow: bool = True) -> pd.DataFrame:
    """모든 조합을 학습하고 결과 표를 반환한다."""
    cfg = config if config is not None else load_config()
    target = cfg["data"]["target_column"]
    seed = cfg["project"]["random_state"]
    n_splits = cfg["cv"]["n_splits"]
    repeats = cfg["cv"]["n_repeats"] if n_repeats is None else n_repeats
    threshold = cfg["thresholds"]["approval_threshold"]
    version = cfg["models"]["model_version"]

    frames = load_splits(cfg)
    y = {name: frames[name][target] for name in frames}
    pos_weight = float((y["train"] == 0).sum() / max((y["train"] == 1).sum(), 1))

    if log_mlflow:
        _setup_mlflow(cfg)

    models_dir = resolve_path(cfg["paths"]["models"])
    models_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    fold_records: list[dict[str, Any]] = []
    fitted: dict[tuple[str, str, str], Any] = {}
    run_ids: dict[str, str] = {}
    total_start = time.perf_counter()

    for group in FEATURE_GROUPS:
        cols = feature_columns(cfg, group)
        x = {name: frames[name][cols] for name in frames}
        for algorithm in algorithms(cfg):
            for imbalance in imbalance_methods(cfg):
                tag = f"{algorithm}|{group}|{imbalance}"
                started = time.perf_counter()

                def factory(a=algorithm, i=imbalance):
                    return build_estimator(a, i, cfg, pos_weight)

                fold_aucs = cross_validate_auc(
                    factory, x["train"], y["train"], n_splits, repeats, seed
                )
                model = factory()
                model.fit(x["train"], y["train"])
                elapsed = time.perf_counter() - started

                scores = {n: model.predict_proba(x[n])[:, 1] for n in ("train", "valid", "test")}
                valid_metrics = classification_metrics(y["valid"], scores["valid"], threshold)
                test_metrics = classification_metrics(y["test"], scores["test"], threshold)
                score_psi = psi(scores["train"], scores["test"])

                row = {
                    "algorithm": algorithm, "feature_group": group, "imbalance": imbalance,
                    "cv_auc_mean": float(np.mean(fold_aucs)),
                    "cv_auc_std": float(np.std(fold_aucs, ddof=1)),
                    "n_folds": len(fold_aucs),
                    "valid_auc": valid_metrics["auc"],
                    "test_auc": test_metrics["auc"],
                    "test_ks": test_metrics["ks"],
                    "test_precision": test_metrics["precision"],
                    "test_recall": test_metrics["recall"],
                    "test_f1": test_metrics["f1"],
                    "psi_train_test": score_psi,
                    "fit_seconds": elapsed,
                }
                rows.append(row)
                fitted[(algorithm, group, imbalance)] = model
                fold_records += [
                    {"algorithm": algorithm, "feature_group": group, "imbalance": imbalance,
                     "fold": i, "auc": v}
                    for i, v in enumerate(fold_aucs)
                ]
                logger.info(
                    "%-52s CV AUC %.4f±%.4f | test AUC %.4f KS %.4f PSI %.4f | %.1fs",
                    tag, row["cv_auc_mean"], row["cv_auc_std"],
                    row["test_auc"], row["test_ks"], row["psi_train_test"], elapsed,
                )

                if log_mlflow:
                    with mlflow.start_run(run_name=tag.replace("|", "_")) as active:
                        run_ids[tag] = active.info.run_id
                        mlflow.log_params({
                            "algorithm": algorithm, "feature_group": group,
                            "imbalance": imbalance, "n_features": len(cols),
                            "n_splits": n_splits, "n_repeats": repeats,
                            "random_state": seed, "model_version": version,
                            "approval_threshold": threshold,
                        })
                        mlflow.log_metrics({
                            "cv_auc_mean": row["cv_auc_mean"], "cv_auc_std": row["cv_auc_std"],
                            "valid_auc": row["valid_auc"], "test_auc": row["test_auc"],
                            "test_ks": row["test_ks"], "test_f1": row["test_f1"],
                            "test_precision": row["test_precision"],
                            "test_recall": row["test_recall"],
                            "psi_train_test": row["psi_train_test"],
                        })
                        for i, v in enumerate(fold_aucs):
                            mlflow.log_metric("fold_auc", v, step=i)
                        mlflow.set_tags({
                            "feature_group": group, "algorithm": algorithm,
                            "imbalance_method": imbalance,
                        })
                        # imblearn 파이프라인은 skops 직렬화가 거부하므로 cloudpickle 을 쓴다.
                        _log_sklearn_model(model)

    results = pd.DataFrame(rows).sort_values("test_auc", ascending=False).reset_index(drop=True)
    folds = pd.DataFrame(fold_records)

    out_dir = resolve_path(cfg["preprocess"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_dir / "model_results.csv", index=False)
    folds.to_csv(out_dir / "fold_auc.csv", index=False)          # ANOVA 입력
    logger.info("결과 저장: %s, %s", out_dir / "model_results.csv", out_dir / "fold_auc.csv")
    if log_mlflow:
        _log_and_register(results, fitted, cfg, run_ids)

    logger.info("총 소요 %.1f초 (%d개 조합, 조합당 %d fold)",
                time.perf_counter() - total_start, len(rows), n_splits * repeats)
    return results


def _log_and_register(
    results: pd.DataFrame, models: dict[str, Any], config: dict[str, Any], run_ids: dict[str, str]
) -> pd.DataFrame:
    """알고리즘별 최고 조합을 models/ 에 저장하고 최우수 모델을 Registry 에 등록한다."""
    from mlflow.tracking import MlflowClient

    version = config["models"]["model_version"]
    models_dir = resolve_path(config["paths"]["models"])
    models_dir.mkdir(parents=True, exist_ok=True)

    saved = []
    for algorithm, group in results.groupby("algorithm", sort=False):
        best = group.sort_values("test_auc", ascending=False).iloc[0]
        key = (best["algorithm"], best["feature_group"], best["imbalance"])
        path = models_dir / f"{algorithm}_{version}.pkl"
        joblib.dump(models[key], path)
        saved.append({
            "algorithm": algorithm, "feature_group": best["feature_group"],
            "imbalance": best["imbalance"], "test_auc": best["test_auc"],
            "file": path.name,
        })
        logger.info("최종 모델 저장: %s (%s / %s, test AUC %.4f)",
                    path.name, best["feature_group"], best["imbalance"], best["test_auc"])

    champion = results.sort_values("test_auc", ascending=False).iloc[0]
    champion_key = (champion["algorithm"], champion["feature_group"], champion["imbalance"])
    champion_path = models_dir / f"champion_{champion['algorithm']}_{version}.pkl"
    joblib.dump(models[champion_key], champion_path)

    registered_name = config["mlflow"].get("registered_model_name", "credit-scoring")
    run_id = run_ids.get("|".join(champion_key))
    try:
        client = MlflowClient()
        model_version = mlflow.register_model(f"runs:/{run_id}/model", registered_name)
        client.set_registered_model_alias(registered_name, "champion", model_version.version)
        client.set_model_version_tag(registered_name, model_version.version, "stage", "Production")
        client.set_model_version_tag(
            registered_name, model_version.version, "combination", "|".join(champion_key)
        )
        logger.info("Registry 등록: %s v%s (alias=champion, stage=Production) — %s test AUC %.4f",
                    registered_name, model_version.version, "|".join(champion_key),
                    champion["test_auc"])
    except Exception as exc:                       # 등록 실패가 학습 결과를 버리게 두지 않는다
        logger.warning("Registry 등록 실패(%s): %s", type(exc).__name__, exc)

    out = pd.DataFrame(saved)
    out.to_csv(resolve_path(config["preprocess"]["output_dir"]) / "saved_models.csv", index=False)
    return out


def thin_filer_uplift(config: dict[str, Any] | None = None) -> pd.DataFrame:
    """씬파일러 / 일반 집단에서 traditional_only vs combined AUC 비교 (SPEC §2)."""
    cfg = config if config is not None else load_config()
    target = cfg["data"]["target_column"]
    seed = cfg["project"]["random_state"]
    frames = load_splits(cfg)
    train, test = frames["train"], frames["test"]
    pos_weight = float((train[target] == 0).sum() / max((train[target] == 1).sum(), 1))

    rows = []
    for algorithm in algorithms(cfg):
        scores: dict[str, np.ndarray] = {}
        for group in ("traditional_only", "combined"):
            cols = feature_columns(cfg, group)
            model = build_estimator(algorithm, "class_weight", cfg, pos_weight)
            model.fit(train[cols], train[target])
            scores[group] = model.predict_proba(test[cols])[:, 1]

        segments: list[tuple[str, pd.Series]] = [
            ("all", pd.Series(True, index=test.index)),
            ("thin_filer", test["is_thin_filer"] == 1),
            ("normal", test["is_thin_filer"] == 0),
        ]
        # ADR-010: 씬파일러 유형별로 대안 데이터 효과가 다른지 본다.
        type_col = cfg["simulator"]["thin_filer_type_column"]
        for kind in cfg["simulator"]["thin_filer_type_mix"]:
            segments.append((f"thin:{kind}", test[type_col] == kind))

        for segment, mask in segments:
            if mask.sum() == 0:
                continue
            y_seg = test.loc[mask, target]
            if y_seg.nunique() < 2:
                continue
            base = auc_score(y_seg, scores["traditional_only"][mask.to_numpy()])
            comb = auc_score(y_seg, scores["combined"][mask.to_numpy()])
            rows.append({
                "algorithm": algorithm, "segment": segment, "n": int(mask.sum()),
                "positive_rate": float(y_seg.mean()),
                "auc_traditional": base, "auc_combined": comb, "uplift": comb - base,
                "meets_target": bool(not segment.startswith("thin") or comb - base >= 0.03),
            })

    out = pd.DataFrame(rows)
    path = resolve_path(cfg["preprocess"]["output_dir"]) / "thin_filer_uplift.csv"
    out.to_csv(path, index=False)
    logger.info("씬파일러 uplift 저장: %s", path)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="신용평가 모델 학습")
    parser.add_argument("--repeats", type=int, default=None,
                        help="RepeatedStratifiedKFold 반복 수 (기본: config 의 cv.n_repeats)")
    parser.add_argument("--no-mlflow", action="store_true", help="MLflow 로깅 생략")
    parser.add_argument("--uplift-only", action="store_true", help="씬파일러 uplift 만 계산")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = load_config()
    if args.uplift_only:
        print(thin_filer_uplift(cfg).to_string(index=False))
        return
    results = run(cfg, n_repeats=args.repeats, log_mlflow=not args.no_mlflow)
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
