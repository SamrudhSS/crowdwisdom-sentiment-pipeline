"""
Shared logging configuration. Every pipeline run writes a timestamped log
file under logs/ (console output mirrored to disk) so you can check exactly
what each stage did -- how many posts were scraped, how many price bars
came back, which API calls were made, warnings/errors, timing per stage --
after the fact, without having to re-run anything.
"""
from __future__ import annotations

import logging
import os
import sys
from datetime import datetime

from config import CFG


def setup_logging(run_name: str = "pipeline") -> logging.Logger:
    os.makedirs(CFG.logs_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(CFG.logs_dir, f"{run_name}_{ts}.log")

    logger = logging.getLogger("crowdwisdom")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()  # avoid duplicate handlers on repeated calls (e.g. in tests)

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(fmt)
    logger.addHandler(console_handler)

    logger.info(f"Logging to {log_path}")
    return logger
