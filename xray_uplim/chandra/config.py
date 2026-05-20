"""
xray_uplim.chandra.config
--------------------------
Chandra ACIS configuration dataclass.

Energy bands
------------
ACIS PI channels: 1 PI ≈ 1 eV  →  energy filter in eV.
Standard bands:
    'broad'      0.5 –  7.0 keV  (ACIS standard; best S/N)
    'soft'       0.5 –  2.0 keV
    'medium'     2.0 –  4.0 keV
    'hard'       4.0 –  7.0 keV
    'full'       0.5 – 10.0 keV  (high background above 7 keV — use with care)
    'ultrasoft'  0.3 –  1.0 keV

Custom band: tuple e.g. (1.0, 6.0) or string '(1.0, 6.0)'.

Directory layout
----------------
Single obsid:
    base_path/obsid/           ← chandra_repro working dir
    base_path/obsid/repro/     ← chandra_repro output (evt2 + asol here)
    base_path/obsid/repro/fluximage/  ← fluximage output

Multiple obsids (co-added):
    base_path/obsid1/repro/
    base_path/obsid2/repro/
    …

CIAO requirement
----------------
CIAO must be initialised in the shell before running this pipeline
(conda activate ciao-XX && source $ASCDS_INSTALL/bin/ciao.sh, or
equivalent).  Required tools: chandra_repro, fluximage, dmlist,
dmstat, dmkeypar, aprates.
"""

import os
import re
from dataclasses import dataclass, field
from typing import List, Tuple, Union


@dataclass
class ChandraConfig:
    """
    All user-configurable parameters for a Chandra ACIS upper-limit run.

    Parameters
    ----------
    base_path : str
        Root directory containing one sub-directory per obsid.
    obsid : str or list of str
        Chandra observation ID(s).  Pass a list to co-add multiple
        observations.  Counts and exposures are summed; a single
        combined upper limit is returned (plus per-obs rows in the CSV).
    ra : str or float
        Source right ascension.  Accepts "HH:MM:SS.ss" or decimal degrees.
    dec : str or float
        Source declination.  Accepts "±DD:MM:SS.ss" or decimal degrees.
    src_radius_arcsec : float
        Source extraction radius in arcseconds.
        Chandra on-axis FWHM ≈ 0.5–1"; typical aperture 2–10".
    bkg_radius_arcsec : float
        Outer radius of the background annulus in arcseconds.
    bkg_inner_factor : float
        Inner radius of background annulus = src_radius_arcsec × this factor.
    psf_fwhm_arcsec : float
        Gaussian PSF FWHM used for the EEF correction.
        On-axis ACIS ≈ 0.5–1"; increases strongly off-axis.
        Set to the PSF FWHM at your source position if off-axis.
    energy_band : str or tuple
        Named band or custom (e_lo_kev, e_hi_kev) tuple.
    bkg_mode : str
        'annulus' — annulus around source (default).
        'manual'  — separate background circle; set bkg_ra / bkg_dec.
    bkg_ra, bkg_dec : str or float
        Background circle centre.  Only used when bkg_mode='manual'.
    confidence_levels : list of float
        One-sided confidence levels, e.g. [0.9545, 0.9973] ≈ [2σ, 3σ].
    run_repro : bool
        If True (default), run chandra_repro automatically if the repro/
        directory for an obsid does not yet exist.
    use_aprates : bool
        If True (default), use CIAO aprates for the Bayesian upper limit.
        If False, fall back to our Kraft/Gehrels implementation.
        The Kraft result is always printed as a cross-check.
    use_gui : bool
        Open the interactive region selector before processing.
    gui_per_obs : bool
        True  — independent GUI for each observation.
        False — GUI for the first observation; regions reused for all.
        Ignored when use_gui=False.
    save_plots : bool
        Save diagnostic plots to <base_path>/ul_products/.
    """

    # -- Observation ----------------------------------------------------------
    base_path : str = ""
    obsid     : Union[str, List[str]] = ""

    # -- Source position ------------------------------------------------------
    ra  : Union[str, float] = ""
    dec : Union[str, float] = ""

    # -- Aperture -------------------------------------------------------------
    src_radius_arcsec : float = 5.0
    bkg_radius_arcsec : float = 30.0
    bkg_inner_factor  : float = 2.0

    # -- PSF ------------------------------------------------------------------
    psf_fwhm_arcsec : float = 0.9   # on-axis ACIS FWHM; increase for off-axis

    # -- Energy band ----------------------------------------------------------
    energy_band : Union[str, Tuple[float, float]] = 'broad'

    # -- Background -----------------------------------------------------------
    bkg_mode : str               = 'annulus'
    bkg_ra   : Union[str, float] = ""
    bkg_dec  : Union[str, float] = ""

    # -- Statistics -----------------------------------------------------------
    confidence_levels : List[float] = field(
        default_factory=lambda: [0.9545, 0.9973])

    # -- CIAO location --------------------------------------------------------
    ciao_prefix : str = ""      # Leave empty for auto-detection (recommended).
                                # Set to the root of the CIAO conda env if
                                # auto-detection fails, e.g.
                                #   ciao_prefix = "~/opt/miniconda3/envs/ciao-4.16"

    # -- Reprocessing ---------------------------------------------------------
    run_repro  : bool = True    # auto-run chandra_repro if repro/ not found

    # -- Statistical backend --------------------------------------------------
    use_aprates : bool = True   # True: CIAO aprates (recommended)
                                # False: pure-Python Kraft (no CIAO for stats)

    # -- GUI & output ---------------------------------------------------------
    use_gui     : bool = True
    gui_per_obs : bool = False
    save_plots  : bool = True
    src_name    : str  = ''   # optional — used in plot titles

    # -- Flux / Luminosity conversion (optional) ----------------------------------
    # Set compute_flux=True to convert count-rate upper limits to flux/luminosity.
    # Requires internet access (WebPIMMS + HEASARC NH tool).
    compute_flux    : bool  = False
    nh_cm2          : float = None   # None = auto-fetch from HI4PI via HEASARC
    spectral_model  : str   = 'powerlaw'   # 'powerlaw','blackbody','bremsstrahlung','apec'
    photon_index    : float = 2.0          # power law photon index (Γ)
    temperature_kev : float = 1.0          # kT in keV (blackbody / bremss / apec)
    abundance       : float = 1.0          # solar abundance (apec only)
    redshift        : float = None         # None = no luminosity output
    cosmology       : str   = 'Planck18'   # 'Planck18', 'WMAP9', or 'custom'
    h0              : float = 67.4         # only used when cosmology='custom'
    omega_m         : float = 0.315        # only used when cosmology='custom'

    # =========================================================================
    # Instrument constants — do not edit
    # =========================================================================

    # Named energy bands: (e_lo_kev, e_hi_kev)
    ENERGY_BANDS : dict = field(default_factory=lambda: {
        'broad'     : (0.5,  7.0),
        'soft'      : (0.5,  2.0),
        'medium'    : (2.0,  4.0),
        'hard'      : (4.0,  7.0),
        'full'      : (0.5, 10.0),
        'ultrasoft' : (0.3,  1.0),
    }, repr=False)

    # Reference (spectral-weight) energy per named band (keV)
    # Used as the 'bands' third argument to fluximage.
    BAND_REFENERGY : dict = field(default_factory=lambda: {
        'broad'     : 1.497,   # log-midpoint 0.5–7 keV
        'soft'      : 0.900,
        'medium'    : 2.828,
        'hard'      : 5.292,
        'full'      : 2.236,
        'ultrasoft' : 0.548,
    }, repr=False)

    # ACIS sky pixel scale (arcsec/pixel)
    ACIS_PSCALE : float = field(default=0.492, repr=False)

    # =========================================================================

    @property
    def obsids(self) -> List[str]:
        """Normalise obsid to a list of strings."""
        if isinstance(self.obsid, list):
            return [str(o).strip() for o in self.obsid]
        return [str(self.obsid).strip()]

    def resolve_energy_band(self):
        """
        Return (e_lo_kev, e_hi_kev, e_lo_ev, e_hi_ev, ref_kev).

        e_lo_ev / e_hi_ev are integer eV values for CIAO energy filters.
        ref_kev is the spectral reference energy for fluximage bands= parameter.
        """
        if isinstance(self.energy_band, tuple):
            e_lo_kev = float(self.energy_band[0])
            e_hi_kev = float(self.energy_band[1])
            e_lo_ev  = int(round(e_lo_kev * 1000))
            e_hi_ev  = int(round(e_hi_kev * 1000))
            ref_kev  = (e_lo_kev + e_hi_kev) / 2.0
            return e_lo_kev, e_hi_kev, e_lo_ev, e_hi_ev, ref_kev

        if isinstance(self.energy_band, str):
            m = re.fullmatch(r'\(\s*([0-9.]+)\s*,\s*([0-9.]+)\s*\)',
                             self.energy_band.strip())
            if m:
                e_lo_kev = float(m.group(1))
                e_hi_kev = float(m.group(2))
                e_lo_ev  = int(round(e_lo_kev * 1000))
                e_hi_ev  = int(round(e_hi_kev * 1000))
                ref_kev  = (e_lo_kev + e_hi_kev) / 2.0
                return e_lo_kev, e_hi_kev, e_lo_ev, e_hi_ev, ref_kev

        key = str(self.energy_band).strip().lower()
        if key not in self.ENERGY_BANDS:
            raise ValueError(
                f"Unknown Chandra energy_band '{self.energy_band}'. "
                f"Choose from {list(self.ENERGY_BANDS)} "
                "or supply a custom (e_lo_kev, e_hi_kev) tuple.")
        e_lo_kev, e_hi_kev = self.ENERGY_BANDS[key]
        e_lo_ev  = int(round(e_lo_kev * 1000))
        e_hi_ev  = int(round(e_hi_kev * 1000))
        ref_kev  = self.BAND_REFENERGY.get(key, (e_lo_kev + e_hi_kev) / 2.0)
        return e_lo_kev, e_hi_kev, e_lo_ev, e_hi_ev, ref_kev

    def validate(self):
        """Raise ValueError for obviously wrong settings."""
        if not self.base_path:
            raise ValueError("base_path is empty.")
        if not self.obsid or (isinstance(self.obsid, list) and len(self.obsid) == 0):
            raise ValueError("obsid is empty.")
        for oid in self.obsids:
            if not oid:
                raise ValueError("One of the obsid entries is empty.")
        if not self.ra or not self.dec:
            raise ValueError("ra and dec must be set.")
        if self.src_radius_arcsec <= 0:
            raise ValueError("src_radius_arcsec must be > 0.")
        if self.bkg_mode not in ('annulus', 'manual'):
            raise ValueError(
                f"bkg_mode must be 'annulus' or 'manual', not '{self.bkg_mode}'.")
        if self.bkg_mode == 'manual' and (not self.bkg_ra or not self.bkg_dec):
            raise ValueError(
                "bkg_mode='manual' requires bkg_ra and bkg_dec to be set.")
        for cl in self.confidence_levels:
            if not 0.0 < cl < 1.0:
                raise ValueError(f"Confidence level {cl} is outside (0, 1).")
        self.resolve_energy_band()
