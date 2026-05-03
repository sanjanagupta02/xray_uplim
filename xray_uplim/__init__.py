"""
xray_uplim
==========
Unified X-ray non-detection upper limit calculator.

Supports NuSTAR, XMM-Newton, Swift/XRT, and Chandra/ACIS (requires CIAO).

Quickstart — NuSTAR
-------------------
>>> from xray_uplim.nustar import run_uplim
>>> run_uplim(
...     base_path="/data/NuSTAR/",
...     obsid="80202052002",
...     ra="20:17:11.360",
...     dec="+58:12:08.10",
... )
"""

from .nustar.pipeline  import run_uplim, process_module, combine_modules
from .nustar.config    import Config
from .statistics       import kraft_upper_limit, gehrels_upper_limit, net_count_rate
from .eef              import compute_eef

__version__ = "2.0.0"
__author__  = "Sanjana Gupta"
__all__ = [
    "run_uplim",
    "process_module",
    "combine_modules",
    "Config",
    "kraft_upper_limit",
    "gehrels_upper_limit",
    "net_count_rate",
    "compute_eef",
]
