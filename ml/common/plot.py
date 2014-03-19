import sys
import os

import matplotlib
headless = False
if sys.platform != 'nt' and 'DISPLAY' not in os.environ:
    matplotlib.use("Agg")
    headless = True
import matplotlib.pyplot as plt