"""
xray_uplim.swift.config
-----------------------
Swift XRT configuration dataclass.

Energy bands
------------
Swift XRT PI channels: 1 PI ≈ 10 eV  →  E_keV ≈ PI / 100.
Standard bands:
    'full'   0.3 – 10.0 keV  (PI   30 – 1000)
    'soft'   0.3 –  1.5 keV  (PI   30 –  150)
    'hard'   1.5 – 10.0 keV  (PI  150 – 1000)
    'ultrasoft' 0.3 – 1.0 keV (PI 30 – 100)

Custom band: tuple e.g. (0.5, 7.0) or string '(0.5, 7.0)'.

PSF calibration
---------------
The PSF is modelled as a King + Gaussian profile using coefficients from
the XIMAGE calibration file psfconst_xrt.fits, which is bundled with
HEASoft under image/ximage/cal/swift/xrt/psfconst_xrt.fits.

In practice the Gaussian fraction P0 = 0 for all current calibrations,
so the PSF reduces to a pure King profile:
    EEF(r) = 1 − [1 + (r/rc)²]^(1−η)
where rc and η depend on energy and off-axis angle.

Search order for the PSF file:
    1. <caldb_dir>/data/swift/xrt/cpf/psf/psfconst_xrt.fits
    2. $CALDB/data/swift/xrt/cpf/psf/psfconst_xrt.fits
    3. xray_uplim/data/swift/psf/psfconst_xrt.fits  (dev copy)

Readout modes
-------------
PC (Photon Counting) : standard mode for faint sources; grades 0–12.
WT (Window Timing)   : used for bright sources; grades 0–2.
Mode is auto-detected from the event file DATAMODE keyword.
"""

import os
import re
import math
import glob
import warnings
from dataclasses import dataclass, field
from typing import List, Tuple, Union


@dataclass
class SwiftConfig:
    """
    All user-configurable parameters for a Swift XRT upper-limit calculation.

    Parameters
    ----------
    data_dir : str
        Directory containing the Swift XRT processed event file(s) and
        exposure map (output of xrtpipeline / xrtexpomap).
    obsid : str
        Swift observation ID (11 digits, e.g. '00012345001').
        Used for file globbing and output naming.
    ra : str or float
        Source right ascension.  Accepts "HH:MM:SS.ss" or decimal degrees.
    dec : str or float
        Source declination.  Accepts "±DD:MM:SS.ss" or decimal degrees.
    src_radius_arcsec : float
        Source extraction radius in arcseconds.
        Swift XRT on-axis PSF FWHM ~5–6 arcsec; typical aperture 20–30 arcsec.
    bkg_radius_arcsec : float
        Outer radius of background annulus in arcseconds.
    bkg_inner_factor : float
        Inner radius of background annulus = src_radius_arcsec × this value.
    psf_fwhm_arcsec : float
        PSF FWHM used only for PSF-weighted exposure diagnostic.
        Swift XRT on-axis FWHM ≈ 5–6 arcsec.
    energy_band : str or tuple
        Named band or custom (e_lo_kev, e_hi_kev) tuple.  See module docstring.
    caldb_dir : str
        Path to CALDB root directory.  Leave empty to use $CALDB or the
        bundled dev copy of psfconst_xrt.fits.
    bkg_mode : str
        'annulus' — background annulus centred on source (default).
        'manual'  — user-supplied background circle; set bkg_ra / bkg_dec.
    bkg_ra, bkg_dec : str or float
        Background circle centre.  Only used when bkg_mode='manual'.
    exp_stat : str
        Exposure summary statistic: 'median' (recommended), 'mean',
        or 'psf_weighted'.
    confidence_levels : list of float
        One-sided confidence levels, e.g. [0.9545, 0.9973] ≈ [2σ, 3σ].
    use_gui : bool
        Open interactive region selector before processing.
    save_plots : bool
        Save diagnostic plots to <data_dir>/ul_products/.
    """

    # -- Observation ----------------------------------------------------------
    data_dir : str = ""
    obsid    : Union[str, List[str]] = ""   # str or list for co-added observations

    # -- Source position ------------------------------------------------------
    ra  : Union[str, float] = ""
    dec : Union[str, float] = ""

    # -- Aperture -------------------------------------------------------------
    src_radius_arcsec : float = 20.0
    bkg_radius_arcsec : float = 80.0
    bkg_inner_factor  : float = 1.5

    # -- PSF ------------------------------------------------------------------
    psf_fwhm_arcsec : float = 6.0   # on-axis XRT FWHM

    # -- Energy band ----------------------------------------------------------
    energy_band : Union[str, Tuple[float, float]] = 'full'

    # -- CALDB ----------------------------------------------------------------
    caldb_dir : str = ""

    # -- Background -----------------------------------------------------------
    bkg_mode : str               = 'annulus'
    bkg_ra   : Union[str, float] = ""
    bkg_dec  : Union[str, float] = ""

    # -- Exposure statistic ---------------------------------------------------
    exp_stat : str = 'median'

    # -- Confidence levels ----------------------------------------------------
    confidence_levels : List[float] = field(
        default_factory=lambda: [0.9545, 0.9973])

    # -- Output ---------------------------------------------------------------
    use_gui    : bool = False
    save_plots : bool = True

    # =========================================================================
    # Instrument constants — do not edit
    # =========================================================================

    # Named energy bands (e_lo, e_hi) in keV
    ENERGY_BANDS = {
        'full'      : (0.3, 10.0),
        'soft'      : (0.3,  1.5),
        'hard'      : (1.5, 10.0),
        'ultrasoft' : (0.3,  1.0),
    }

    # Swift XRT PI calibration: ~1 PI = 10 eV  →  E_keV ≈ PI / 100
    PI_PER_KEV = 100

    # Grade (PATTERN) limits per readout mode
    GRADE_LIMITS = {
        'PC': 12,
        'WT':  2,
    }

    # Sky pixel scale (arcsec/pixel) for Swift XRT
    XRT_PSCALE = 2.36   # arcsec/pix

    # PSF filename inside CALDB subtree and dev location
    PSF_FILENAME  = 'psfconst_xrt.fits'
    PSF_CALDB_SUB = os.path.join('data', 'swift', 'xrt', 'cpf', 'psf')

    # =========================================================================
    # Methods
    # =========================================================================

    @property
    def obsids(self) -> List[str]:
        """Normalise obsid to a list of strings (always at least one element)."""
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
        """Convert energy bounds in keV to integer PI channel bounds."""
        pi_lo = int(math.floor(e_lo_kev * self.PI_PER_KEV))
        pi_hi = int(math.ceil (e_hi_kev * self.PI_PER_KEV))
        return pi_lo, pi_hi

    def resolve_psf_file(self):
        """
        Return path to psfconst_xrt.fits.

        Search order:
            1. <caldb_dir>/data/swift/xrt/cpf/psf/psfconst_xrt.fits
            2. $CALDB/data/swift/xrt/cpf/psf/psfconst_xrt.fits
            3. xray_uplim/data/swift/psf/psfconst_xrt.fits  (dev copy)
        """
        def _check(d):
            p = os.path.join(d, self.PSF_FILENAME)
            return p if os.path.isfile(p) else None

        # 1. Explicit caldb_dir
        if self.caldb_dir:
            p = _check(os.path.join(self.caldb_dir, self.PSF_CALDB_SUB))
            if p:
                return p
            raise FileNotFoundError(
                f"psfconst_xrt.fits not found in caldb_dir:\n"
                f"  {os.path.join(self.caldb_dir, self.PSF_CALDB_SUB)}")

        # 2. $CALDB environment variable
        caldb_env = os.environ.get('CALDB', '')
        if caldb_env:
            p = _check(os.path.join(caldb_env, self.PSF_CALDB_SUB))
            if p:
                return p
            warnings.warn(
                f"$CALDB is set ({caldb_env!r}) but psfconst_xrt.fits not found "
                f"under {self.PSF_CALDB_SUB}. Falling back to bundled dev copy.",
                UserWarning, stacklevel=3)

        # 3. Dev copy bundled with the package
        bundled = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'data', 'swift', 'psf', self.PSF_FILENAME)
        if os.path.isfile(bundled):
            return bundled

        raise FileNotFoundError(
            "Swift XRT PSF file (psfconst_xrt.fits) not found. Search order:\n"
            "  1. <caldb_dir>/data/swift/xrt/cpf/psf/  (set caldb_dir= in config)\n"
            "  2. $CALDB/data/swift/xrt/cpf/psf/       (set by HEASoft initialisation)\n"
            "  3. xray_uplim/data/swift/psf/            (dev copy)\n"
            "The file ships with HEASoft under:\n"
            "  <heasoft>/image/ximage/cal/swift/xrt/psfconst_xrt.fits\n"
            "Copy it to any of the locations above.")

    def validate(self):
        """Raise ValueError for obviously wrong settings."""
        if not self.data_dir:
            raise ValueError("data_dir is empty.")
        if not self.obsid or (isinstance(self.obsid, list) and len(self.obsid) == 0):
            raise ValueError("obsid is empty.")
        for oid in self.obsids:
            if not oid:
                raise ValueError("One of the obsid entries is empty.")
        if not self.ra or not self.dec:
            raise ValueError("ra and dec must be set.")
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
                raise ValueError(f"Confidence level {cl} is outside (0, 1).")
