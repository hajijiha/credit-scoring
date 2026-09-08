"""설정 로딩 · 경로 해석.

프로젝트의 모든 설정값은 config/config.yaml 에서만 읽는다.
모든 경로는 프로젝트 루트 기준 상대경로이며, 여기서 절대경로로 해석한다.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

# src/config.py -> src -> 프로젝트 루트
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_CONFIG_PATH = "config/config.yaml"


def load_config(config_path: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    """config.yaml 을 읽어 dict 로 반환한다.

    경로 우선순위: 인자 > 환경변수 CONFIG_PATH > 기본값 config/config.yaml
    상대경로는 프로젝트 루트 기준으로 해석한다.
    """
    raw = config_path or os.getenv("CONFIG_PATH") or DEFAULT_CONFIG_PATH
    path = resolve_path(raw)
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_yaml(relative: str | os.PathLike[str]) -> dict[str, Any]:
    """프로젝트 루트 기준 상대경로의 YAML 을 읽는다(보조 설정 파일용)."""
    with open(resolve_path(relative), encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(relative: str | os.PathLike[str]) -> Path:
    """상대경로를 프로젝트 루트 기준 절대경로로 변환한다. 절대경로는 그대로 둔다."""
    path = Path(relative)
    return path if path.is_absolute() else PROJECT_ROOT / path
