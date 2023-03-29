"""Some helpful constants."""

import logging
import multiprocessing
import os

THEIA_LOG = getattr(logging, os.environ.get("THEIA_LOG_LEVEL", "INFO"))

MAX_WORKERS = max(1, multiprocessing.cpu_count() // 2)
MIN_DATA_SIZE = 2**19  # Load at least this much data
MAX_DATA_SIZE = 2**31  # Limit to loading 500MB of pixels
EPSILON = 1e-8  # To avoid divide-by-zero errors
