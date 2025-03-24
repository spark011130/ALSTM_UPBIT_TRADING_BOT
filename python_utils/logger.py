import logging
import os

def get_logger(name: str, log_file: str = "outputs/trading.log", level=logging.INFO) -> logging.Logger:
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s', datefmt="%Y-%m-%d %H:%M:%S")

    # FileHandler에 utf-8 인코딩 추가
    handler = logging.FileHandler(log_file, encoding='utf-8')
    handler.setFormatter(formatter)

    # Console에 utf-8 인코딩 추가
    console = logging.StreamHandler()
    console.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    logger.addHandler(console)
    logger.propagate = False  # 중복 로그 방지

    return logger