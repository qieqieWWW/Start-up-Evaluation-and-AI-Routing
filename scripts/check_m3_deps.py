#!/usr/bin/env python
# 检查 M3 依赖
import sys
print("Python version:", sys.version)

try:
    import gymnasium
    print("gymnasium OK:", gymnasium.__version__)
except ImportError as e:
    print("gymnasium NOT FOUND:", e)

try:
    import numpy as np
    print("numpy OK:", np.__version__)
except ImportError as e:
    print("numpy NOT FOUND:", e)

try:
    import pandas as pd
    print("pandas OK:", pd.__version__)
except ImportError as e:
    print("pandas NOT FOUND:", e)
