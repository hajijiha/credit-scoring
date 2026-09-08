"""전처리 파이프라인 실행 CLI.

    python -m src.data.run_preprocess

전처리기 클래스를 정의한 모듈(`src.data.preprocessor`)을 직접 `-m` 으로 실행하면
클래스의 `__module__` 이 `"__main__"` 이 되어 저장된 joblib 을 다른 프로세스에서
역직렬화할 수 없다. 그래서 실행 진입점을 분리한다.
"""

from __future__ import annotations

import logging

from src.config import load_config
from src.data.loader import load_gmsc
from src.data.preprocessor import prepare
from src.data.simulator import simulate


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = load_config()
    data = simulate(load_gmsc(cfg), cfg)
    prepare(data, cfg, save=True)


if __name__ == "__main__":
    main()
