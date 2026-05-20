"""
xray_uplim.nustar.config
------------------------
Single dataclass holding every user-facing parameter.
Pass a Config instance to run_uplim() or process_module().
"""

from dataclasses import dataclass, field
from typing import List, Union, Tuple


@dataclass
class Config:
    """
    All user-configurable parameters for a NuSTAR upper-limit calculation.

    Parameters
    ----------
    base_path : str
        Root directory containing the observation folder.
    obsid : str
        NuSTAR observation ID (folder name inside base_path).
    ra : str or float
        Source right ascension.  Accepts "HH:MM:SS.ss" or decimal degrees.
    dec : str or float
        Source declination.  Accepts "±DD:MM:SS.ss" or decimal degrees.
    src_radius_arcsec : float
        Radius of the source extraction circle in arcseconds.
        NuSTAR EEF: ~50% at 20", ~60% at 30", ~80% at 60".
        Divide the final count-rate upper limit by the EEF to correct for
        flux outside the aperture.
    bkg_radius_arcsec : float
        Outer radius of the background annulus in arcseconds.
        Inner radius is set automatically to src_radius_arcsec * bkg_inner_factor.
    bkg_inner_factor : float
        Inner radius of background annulus = src_radius_arcsec * this value.
        Default 1.2 leaves a buffer to exclude PSF wings from the background.
    psf_fwhm_arcsec : float
        PSF FWHM in arcseconds.  Used only for the PSF-weighted exposure
        diagnostic and the radial profile plot.
        Harrison et al. 2013 (ApJ 770, 103): on-axis FWHM ~ 18".
        Increase to ~20-25" for sources more than ~2' off-axis.
    energy_band : str or tuple
        Named band (aligned with NuSTAR CALDB energy files):
            'full'       3.0–79.0 keV  (all 6 CALDB files, spectrally combined)
            'extra-soft' 3.0– 4.5 keV
            'soft'       4.5– 6.0 keV
            'iron'       6.0– 8.0 keV  (Fe K band)
            'medium'     8.0–12.0 keV
            'hard'       12.0–20.0 keV
            'ultra-hard' 20.0–79.0 keV
        For any named band other than 'full', exactly one CALDB PSF file is
        used with no spectral combination required.
        Custom band: tuple of (e_lo_kev, e_hi_kev), e.g. (8.0, 24.0).
    modules : list of str
        FPMs to process.  Any subset of ['A', 'B'].
    bkg_mode : str
        'annulus' — automatic annulus centred on the source (recommended).
        'manual'  — user-supplied background circle; set bkg_ra / bkg_dec.
    bkg_ra : str or float
        Background circle centre RA.  Only used when bkg_mode='manual'.
    bkg_dec : str or float
        Background circle centre Dec.  Only used when bkg_mode='manual'.
    exp_stat : str
        Statistic used to summarise exposure-map pixels inside the source
        aperture into a single effective exposure time.
        'median'       — recommended for non-detections; robust, no PSF assumption.
        'mean'         — fine when vignetting variation across aperture is small.
        'psf_weighted' — diagnostic only; unreliable for off-axis sources.
        All three values are always printed regardless of this choice.
    confidence_levels : list of float
        One-sided confidence levels for the upper limits.
        Common choices (Gaussian convention):
            0.9000 → 1.28σ   0.9500 → 1.64σ
            0.9545 ≈ 2σ      0.9973 ≈ 3σ
    caldb_dir : str
        Path to the CALDB root directory (the folder that contains
        data/nustar/…).  If empty, the $CALDB environment variable is used.
        Required for EEF computation from the NuSTAR PSF.  If neither
        caldb_dir nor $CALDB is set, EEF correction is skipped and only
        aperture count-rate upper limits are reported.
    save_plots : bool
        Whether to save diagnostic plots to <base_path>/<obsid>/ul_products/.
    """

    # -- Observation ----------------------------------------------------------
    base_path : str                    = ""
    obsid     : Union[str, List[str]]  = ""   # str or list for co-added obs

    # -- Source position ------------------------------------------------------
    ra  : Union[str, float] = ""
    dec : Union[str, float] = ""

    # -- Aperture sizes -------------------------------------------------------
    src_radius_arcsec : float = 60.0
    bkg_radius_arcsec : float = 200.0
    bkg_inner_factor  : float = 1.2

    # -- PSF ------------------------------------------------------------------
    psf_fwhm_arcsec : float = 18.0

    # -- Energy band ----------------------------------------------------------
    energy_band : Union[str, Tuple[float, float]] = 'full'

    # -- Modules --------------------------------------------------------------
    modules : List[str] = field(default_factory=lambda: ['A', 'B'])

    # -- Background mode ------------------------------------------------------
    bkg_mode : str              = 'annulus'
    bkg_ra   : Union[str, float] = ""
    bkg_dec  : Union[str, float] = ""

    # -- Exposure statistic ---------------------------------------------------
    exp_stat : str = 'median'

    # -- Confidence levels ----------------------------------------------------
    confidence_levels : List[float] = field(
        default_factory=lambda: [0.9545, 0.9973])

    # -- CALDB ----------------------------------------------------------------
    caldb_dir : str = ""

    # -- Spectral weighting ---------------------------------------------------
    psf_gamma : float = 2.0
    # Photon index used when combining PSF files across energy sub-bands for
    # the full band (or any custom band spanning multiple CALDB files).
    # The weight for each sub-band is  integral_{e_lo}^{e_hi} E^{-Gamma} dE.
    # Gamma=2 is a reasonable prior for X-ray binaries and AGN.
    # Use Gamma=1.7 for a harder spectrum; Gamma=0 for flat (equal weight per
    # band).  Has no effect for single-band observations (e.g. 'soft', 'hard').

    # -- GUI per observation --------------------------------------------------
    gui_per_obs : bool = False
    # When True and use_gui=True, opens the interactive region selector for
    # EACH observation independently.  Use when pointings differ or the source
    # falls at the edge / off-chip in some observations.
    # When False (default), GUI runs once per FPM on the FIRST observation;
    # the chosen aperture and background regions are carried across subsequent
    # observations (sky coordinates are re-projected into each obs's pixel frame).

    # -- Interactive GUI ------------------------------------------------------
    use_gui : bool = False
    # When True, opens an interactive matplotlib window before each FPM is
    # processed so the user can adjust the source and background regions
    # visually.  Requires a display (not suitable for batch/HPC runs).

    # -- Output ---------------------------------------------------------------
    save_plots : bool = True
    src_name   : str  = ''    # optional — used in plot titles

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

    # -------------------------------------------------------------------------

    ENERGY_BANDS = {
        'full':       (3.0,  79.0),   # all 6 CALDB energy files combined (Gamma=2 weights)
        'extra-soft': (3.0,   4.5),   # CALDB nuXX2dpsfen1
        'soft':       (4.5,   6.0),   # CALDB nuXX2dpsfen2
        'iron':       (6.0,   8.0),   # CALDB nuXX2dpsfen3  (Fe K band)
        'medium':     (8.0,  12.0),   # CALDB nuXX2dpsfen4
        'hard':       (12.0, 20.0),   # CALDB nuXX2dpsfen5
        'ultra-hard': (20.0, 79.0),   # CALDB nuXX2dpsfen6
    }

    @property
    def obsids(self) -> List[str]:
        """Return observation IDs as a list (always, even for a single obsid)."""
        if isinstance(self.obsid, list):
            return [str(o).strip() for o in self.obsid]
        return [str(self.obsid).strip()]

    def resolve_energy_band(self):
        """Return (e_lo, e_hi) in keV."""
        if isinstance(self.energy_band, tuple):
            return float(self.energy_band[0]), float(self.energy_band[1])

        # Accept string representations of tuples e.g. "(8.0, 24.0)"
        if isinstance(self.energy_band, str):
            import re
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

    def validate(self):
        """Raise ValueError for obviously wrong settings."""
        if not self.base_path:
            raise ValueError("base_path is empty.")
        if isinstance(self.obsid, list):
            if not self.obsid:
                raise ValueError("obsid list is empty.")
        else:
            if not self.obsid:
                raise ValueError("obsid is empty.")
        if not self.ra or not self.dec:
            raise ValueError("ra and dec must be set.")
        if self.bkg_mode == 'manual' and (not self.bkg_ra or not self.bkg_dec):
            raise ValueError(
                "bkg_mode='manual' requires bkg_ra and bkg_dec to be set.")
        if self.exp_stat not in ('median', 'mean', 'psf_weighted'):
            raise ValueError(
                f"exp_stat must be 'median', 'mean', or 'psf_weighted', "
                f"not '{self.exp_stat}'.")
        if self.psf_gamma < 0:
            raise ValueError(
                f"psf_gamma must be >= 0, not {self.psf_gamma}.")
        for cl in self.confidence_levels:
            if not 0.0 < cl < 1.0:
                raise ValueError(
                    f"Confidence level {cl} is outside (0, 1).")
