# Frequently used functions

import logging
def default_logger() -> logging.RootLogger:
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter(
        fmt='%(asctime)s %(levelname)s: %(message)s', 
        datefmt='%Y/%m/%d %H:%M:%S'
    )
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger

import argparse
def read_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Occupy GPU memory almost full')
    parser.add_argument('-g', '--gpu', type=int, default=-1, nargs='*', help='Indices of gpus to occupy. default:-1 (all gpus)')
    args = parser.parse_args()
    return args
