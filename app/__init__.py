import os
import sys

# torch and xgboost each bundle their own OpenMP runtime; loading both in one
# process aborts on macOS. KMP_DUPLICATE_LIB_OK lets them coexist, and pinning
# OpenMP to a single thread avoids the thread-pool init segfault when both run.
# numpy uses Apple's Accelerate BLAS (not OpenMP), so eval matmuls stay fast.
# On Linux/VPS a single shared libgomp is used, so this guard is skipped and
# full multithreading is preserved.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
if sys.platform == "darwin":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
