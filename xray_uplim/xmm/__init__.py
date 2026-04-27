"""
xray_uplim.xmm
--------------
XMM-Newton EPIC upper-limit pipeline.

Per-instrument results only — MOS1, MOS2, and PN are never combined
(different effective areas, response matrices, and PSF shapes).
"""

from .pipeline import run_uplim, process_instrument
from .config   import XMMConfig
