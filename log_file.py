# loggings.py
import logging
import os
import sys


def phase_1(acc):
    log = logging.getLogger(acc)
    log.setLevel("DEBUG")

    handler = logging.FileHandler(f"C:\\Users\\geeth\\PycharmProjects\\Opencv_\\logging_files\\{acc}.log", mode='w')
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    log.addHandler(handler)

    # Add console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    log.addHandler(console_handler)

    return log