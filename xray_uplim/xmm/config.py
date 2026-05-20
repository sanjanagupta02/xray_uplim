"""
xray_uplim.xmm.config
---------------------
XMM-Newton EPIC configuration dataclass.

The user only needs to edit the top-level run script.  All instrument-
specific constants (PI conversion, PATTERN limits, PSF file names) are
defined here and used automatically by the pipeline.

Energy bands
------------
XMM EPIC PI channels: 1 PI = 1 eV, so E_keV = PI / 1000.

Named bands:
    'full'      0.2 – 12.0 keV   (PI  200 – 12000)
    'soft'      0.5 –  2.0 keV   (PI  500 –  2000)
    'hard'      2.0 – 10.0 keV   (PI 2000 – 10000)
    'medium'    1.0 –  2.0 keV   (PI 1000 –  2000)
    'ultrasoft' 0.2 –  0.5 keV   (PI  200 –   500)

Custom band: tuple e.g. (0.5, 7.0) or string '(0.5, 7.0)'.

PSF calibration
---------------
The XMM EPIC PSF is stored in the SAS Current Calibration Files (CCF):
    XRT1_XPSF_*.CCF  —  MOS1
    XRT2_XPSF_*.CCF  —  MOS2
    XRT3_XPSF_*.CCF  —  PN
By default the pipeline looks for these in xray_uplim/data/xmm/psf/.
Set psf_dir= to override with your own SAS_CCFPATH directory.

Download:
    https://www.cosmos.esa.int/web/xmm-newton/current-calibration-files
    (search for XRT1_XPSF, XRT2_XPSF, XRT3_XPSF)
"""

import os
import re
import math
import glob
import warnings
from dataclasses import dataclass, field
from typing import List, Union, Tuple


@dataclass
class XMMConfig:
    """
    All user-configurable parameters for an XMM-Newton upper-limit calculation.

    Parameters
    ----------
    data_dir : str
        Directory containing SAS-processed event files and exposure maps
        (typically the ODF working directory where you ran epproc/emproc
        and eexpmap).
    obsid : str or list of str
        XMM obs ID (e.g. '0881990901'), or list for co-adding.
        Single obs: data_dir is the working dir.
        Multiple obs: data_dir is parent, each obs in data_dir/<obsid>/.
    ra : str or float
        Source right ascension.  Accepts "HH:MM:SS.ss" or decimal degrees.
    dec : str or float
        Source declination.  Accepts "±DD:MM:SS.ss" or decimal degrees.
    src_radius_arcsec : float
        Source extraction radius in arcseconds.
        Typical on-axis values: 15–20 arcsec (MOS FWHM ~4.5", PN ~6").
        Increase for off-axis sources where the PSF broadens.
    bkg_radius_arcsec : float
        Outer radius of background annulus in arcseconds.
    bkg_inner_factor : float
        Inner radius of background annulus = src_radius_arcsec * this value.
    psf_fwhm_arcsec : float
        PSF FWHM for PSF-weighted exposure diagnostic.
        MOS on-axis ~4.5 arcsec; PN on-axis ~6 arcsec.
    energy_band : str or tuple
        Named band or custom (e_lo_kev, e_hi_kev) tuple.  See module docstring.
    instruments : list of str
        EPIC instruments to process.  Any subset of ['MOS1', 'MOS2', 'PN'].
        Results are always reported per-instrument — never combined across
        instruments (PN and MOS have different effective areas and responses).
    bkg_mode : str
        'annulus' — background annulus centred on source (default).
        'manual'  — user-supplied background circle; set bkg_ra / bkg_dec.
    bkg_ra, bkg_dec : str or float
        Background circle centre.  Only used when bkg_mode='manual'.
    exp_stat : str
        Exposure summary statistic: 'median' (recommended), 'mean',
        or 'psf_weighted' (diagnostic only).
    confidence_levels : list of float
        One-sided confidence levels, e.g. [0.9545, 0.9973] ≈ [2σ, 3σ].
    psf_dir : str
        Path to directory containing XRT[1-3]_XPSF_*.CCF files.
        Leave empty if you have manually placed the CCF files in
        xray_uplim/data/xmm/psf/ (not bundled with the package — too large).
        Download from:
        https://www.cosmos.esa.int/web/xmm-newton/current-calibration-files
    use_gui : bool
        Open interactive region selector before each instrument.
    gui_per_obs : bool
        When use_gui=True and multiple obsids are given, open a separate
        interactive GUI for each observation (True) or only for the first
        observation and carry the aperture settings to all others (False).
    save_plots : bool
        Save diagnostic plots to <data_dir>/ul_products/.
    """

    # -- Observation ----------------------------------------------------------
    data_dir : str = ""
    obsid    : Union[str, List[str]] = ""

    # -- Source position ------------------------------------------------------
    ra  : Union[str, float] = ""
    dec : Union[str, float] = ""

    # -- Aperture -------------------------------------------------------------
    src_radius_arcsec : float = 20.0
    bkg_radius_arcsec : float = 60.0
    bkg_inner_factor  : float = 1.5

    # -- PSF ------------------------------------------------------------------
    psf_fwhm_arcsec : float = 5.0   # conservative default; MOS ~4.5", PN ~6"

    # -- Energy band ----------------------------------------------------------
    energy_band : Union[str, Tuple[float, float]] = 'full'

    # -- Instruments ----------------------------------------------------------
    instruments : List[str] = field(default_factory=lambda: ['MOS1', 'MOS2', 'PN'])

    # -- Background -----------------------------------------------------------
    bkg_mode : str               = 'annulus'
    bkg_ra   : Union[str, float] = ""
    bkg_dec  : Union[str, float] = ""

    # -- Exposure statistic ---------------------------------------------------
    exp_stat : str = 'median'

    # -- Confidence levels ----------------------------------------------------
    confidence_levels : List[float] = field(
        default_factory=lambda: [0.9545, 0.9973])

    # -- PSF calibration ------------------------------------------------------
    psf_dir : str = ""

    # -- Output ---------------------------------------------------------------
    use_gui     : bool = False
    gui_per_obs : bool = False
    save_plots  : bool = True
    src_name    : str  = ''   # optional — used in plot titles

    # =========================================================================
    # Instrument constants — do not edit
    # =========================================================================

    # Named energy bands (e_lo, e_hi) in keV
    ENERGY_BANDS = {
        'full'      : (0.2, 12.0),
        'soft'      : (0.5,  2.0),
        'hard'      : (2.0, 10.0),
        'medium'    : (1.0,  2.0),
        'ultrasoft' : (0.2,  0.5),
    }

    # XMM EPIC PI calibration: 1 PI = 1 eV  →  E_keV = PI / 1000
    PI_PER_KEV = 1000

    # Maximum PATTERN per instrument
    PATTERN_LIMITS = {
        'MOS1': 12,   # singles(0) + doubles(1-4) + triples(5-8) + quads(9-12)
        'MOS2': 12,
        'PN'  :  4,   # singles(0) + doubles(1-4) only — PN reads out faster
    }

    # INSTRUME keyword value in FITS header
    INSTRUME_KEYS = {
        'MOS1': 'EMOS1',
        'MOS2': 'EMOS2',
        'PN'  : 'EPN',
    }

    # PSF CCF filename glob per instrument (XRT1=MOS1, XRT2=MOS2, XRT3=PN)
    PSF_GLOBS = {
        'MOS1': 'XRT1_XPSF_*.CCF',
        'MOS2': 'XRT2_XPSF_*.CCF',
        'PN'  : 'XRT3_XPSF_*.CCF',
    }

    # On-axis PSF FWHM per instrument (arcsec) — used for PSF-weighted exposure
    PSF_FWHM_DEFAULT = {
        'MOS1': 4.5,
        'MOS2': 4.5,
        'PN'  : 6.0,
    }

    # =========================================================================
    # Methods
    # =========================================================================

    @property
    def obsids(self) -> List[str]:
        if isinstance(self.obsid, list):
            return [str(o).strip() for o in self.obsid]
        return [str(self.obsid).strip()]

    def resolve_energy_band(self):
        """Return (e_lo_kev, e_hi_kev)."""
        if isinstance(self.energy_band, tuple):
            return float(self.energy_band[0]), float(self.energy_band[1])
        if isinstance(self.energy_band, str):
            m = re.fullmatch(r'\(\s*([0-9.]+)\s*,\s*([0-9.]+)\s*\)',
                             self.energy_band.strip())
            if m:
                return float(m.group(1)), float(m.group(2))
        key = self.energy_band.lower()
        if key not in self.ENERGY_BANDS:
            raise ValueError(
                f"Unknown energy_band '{self.energy_band}'. "
                f"Use one of {list(self.ENERGY_BANDS)} or a (e_lo, e_hi) tuple.")
        return self.ENERGY_BANDS[key]

    def energy_to_pi(self, e_lo_kev, e_hi_kev):
        """
        Convert energy bounds in keV to integer PI channel bounds.

        Uses floor/ceil to ensure the band is strictly inclusive at both edges.
        """
        pi_lo = int(math.floor(e_lo_kev * self.PI_PER_KEV))
        pi_hi = int(math.ceil (e_hi_kev * self.PI_PER_KEV))
        return pi_lo, pi_hi

    def resolve_psf_dir(self):
        """
        Return path to directory containing XMM PSF CCF files.

        Search order:
            1. cfg.psf_dir            (user-specified)
            2. $SAS_CCFPATH env var   (set automatically by SAS initialisation)
            3. xray_uplim/data/xmm/psf/  (manually placed files)
        """
        def _has_psf_files(d):
            """True if directory d contains at least one XRT?_XPSF_*.CCF file."""
            return bool(glob.glob(os.path.join(d, 'XRT?_XPSF_*.CCF')))

        # 1. Explicit user setting
        if self.psf_dir:
            if not os.path.isdir(self.psf_dir):
                raise FileNotFoundError(
                    f"psf_dir does not exist: {self.psf_dir}")
            if not _has_psf_files(self.psf_dir):
                raise FileNotFoundError(
                    f"psf_dir exists but contains no XRT?_XPSF_*.CCF files:\n"
                    f"  {self.psf_dir}\n"
                    "Download from: "
                    "https://www.cosmos.esa.int/web/xmm-newton/current-calibration-files")
            return self.psf_dir

        # 2. $SAS_CCFPATH — may be colon-separated list of directories
        sas_ccfpath = os.environ.get('SAS_CCFPATH', '')
        if sas_ccfpath:
            for d in sas_ccfpath.split(':'):
                d = d.strip()
                if d and os.path.isdir(d) and _has_psf_files(d):
                    return d
            # Env var set but CCF files not found in any listed directory
            warnings.warn(
                f"$SAS_CCFPATH is set ({sas_ccfpath!r}) but no XRT?_XPSF_*.CCF "
                "files were found in any of its directories. "
                "Falling back to xray_uplim/data/xmm/psf/.",
                UserWarning, stacklevel=3)

        # 3. Manually placed files in the package data directory
        bundled = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'data', 'xmm', 'psf')
        if os.path.isdir(bundled) and _has_psf_files(bundled):
            return bundled

        raise FileNotFoundError(
            "XMM PSF CCF files not found. Search order tried:\n"
            "  1. psf_dir= setting in run_uplim.py\n"
            "  2. $SAS_CCFPATH environment variable (set by SAS initialisation)\n"
            "  3. xray_uplim/data/xmm/psf/\n"
            "If SAS is installed, source your SAS setup script so $SAS_CCFPATH is\n"
            "exported, then re-run — no other config change needed.\n"
            "To download: "
            "https://www.cosmos.esa.int/web/xmm-newton/current-calibration-files\n"
            "(search XRT1_XPSF, XRT2_XPSF, XRT3_XPSF)")

    def validate(self):
        """Raise ValueError for obviously wrong settings."""
        if not self.data_dir:
            raise ValueError("data_dir is empty.")
        if not self.obsid or (isinstance(self.obsid, list) and len(self.obsid) == 0):
            raise ValueError("obsid is empty.")
        if not self.ra or not self.dec:
            raise ValueError("ra and dec must be set.")
        if not self.instruments:
            raise ValueError("instruments list is empty.")
        for inst in self.instruments:
            if inst not in ('MOS1', 'MOS2', 'PN'):
                raise ValueError(
                    f"Unknown instrument '{inst}'. "
                    "Use 'MOS1', 'MOS2', and/or 'PN'.")
        if self.bkg_mode not in ('annulus', 'manual'):
            raise ValueError(
                f"bkg_mode must be 'annulus' or 'manual', not '{self.bkg_mode}'.")
        if self.bkg_mode == 'manual' and (not self.bkg_ra or not self.bkg_dec):
            raise ValueError(
                "bkg_mode='manual' requires bkg_ra and bkg_dec to be set.")
        if self.exp_stat not in ('median', 'mean', 'psf_weighted'):
            raise ValueError(
                f"exp_stat must be 'median', 'mean', or 'psf_weighted', "
                f"not '{self.exp_stat}'.")
        for cl in self.confidence_levels:
            if not 0.0 < cl < 1.0:
                raise ValueError(
                    f"Confidence level {cl} is outside (0, 1).")
