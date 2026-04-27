"""
xray_uplim.xmm.eef
-------------------
Encircled Energy Fraction (EEF) for XMM-Newton EPIC from PSF CCF files.

Physics
-------
XMM EPIC PSF is tabulated in SAS Current Calibration Files (CCF):
    XRT1_XPSF_*.CCF  —  MOS1
    XRT2_XPSF_*.CCF  —  MOS2
    XRT3_XPSF_*.CCF  —  PN

Each CCF holds a 2-D grid of 512×512 PSF images at:
    11 energies   : 100, 1500, 3000, 4500, 6000, 7500, 9000,
                    10500, 12000, 13500, 15000 eV
     6 off-axis   : 0, 3, 6, 9, 12, 15 arcmin
Pixel scale       : 1.1 arcsec/pix
                    (Y_PIXSZ = 0.04 in CCF header is in mm, not arcsec;
                     0.04 mm × 27.5 arcsec/mm at f=7500 mm = 1.10 arcsec/pix)
PSF centre        : pixel (255.5, 255.5) in 0-indexed 512×512 image

The EEF at aperture radius r for a source at off-axis angle θ and band
centre energy E_c is computed by:

  1. Loading all 66 PSF images into a (n_E, n_θ, 512, 512) grid.
  2. 2-D bilinear interpolation over (E_c, θ) between the four
     nearest grid cells — both dimensions simultaneously.
  3. Integrating the interpolated PSF within a circular aperture of
     radius r arcsec using the shared integrate_eef().

This is strictly better than NuSTAR's approach (which interpolates only
in θ) because the full energy × angle grid lives in one file.

Off-axis angle
--------------
θ is computed from the telescope pointing stored in the EVENTS header
(RA_PNT / DEC_PNT keywords written by SAS epproc/emproc).  The shared
off_axis_angle() from xray_uplim.eef is reused directly — it also checks
RA_NOM, DEC_NOM, RA_OBJ, DEC_OBJ as fallbacks.

Public API
----------
load_xmm_psf_grid(ccf_path)
    → (psf_grid, energies_ev, thetas_arcmin)

interpolate_xmm_psf(energy_ev, theta_arcmin, psf_grid, energies, thetas)
    → (512, 512) normalised PSF image

compute_xmm_eef(cfg, instrument, evt_hdr, r_src_arcsec, e_lo_kev, e_hi_kev)
    → dict  (same key structure as NuSTAR compute_eef())
"""

import os
import re
import glob
import warnings
import numpy as np
from astropy.io import fits

from ..eef    import off_axis_angle, integrate_eef
from ..coords import parse_coord
from .config  import XMMConfig


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

XMM_PSF_PSCALE   = 1.1     # arcsec/pix
# Derivation: Y_PIXSZ = 0.04 in the CCF header is in *millimetres*
# (= XMM MOS CCD physical pixel size, 40 µm = 0.04 mm).
# XMM focal length = 7500 mm  →  plate scale = 206265 / 7500 = 27.5 arcsec/mm
# Angular pixel scale = 0.04 mm × 27.5 arcsec/mm = 1.10 arcsec/pix
# (Using 0.04 arcsec/pix was wrong: 20" aperture → r=500 px → entire 512×512
#  image covered → EEF = 1.0 always, no correction applied.)
XMM_MAX_OFFAXIS  = 15.0    # arcmin — largest tabulated off-axis angle


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _renorm(arr):
    """Clip negatives and renormalise a 2-D array to unit pixel sum."""
    arr = np.clip(arr, 0.0, None)
    s   = arr.sum()
    return arr / s if s > 0 else arr


# ---------------------------------------------------------------------------
# PSF grid loading
# ---------------------------------------------------------------------------

def load_xmm_psf_grid(ccf_path: str):
    """
    Load all PSF images from an XMM EPIC CCF file into a structured grid.

    Reads every 2-D IMAGE extension, extracts the energy (eV) and off-axis
    angle (arcmin) from the ENERGY and THETA header keywords (or from the
    extension name as a fallback), and assembles a 4-D array.

    Parameters
    ----------
    ccf_path : str  — path to XRT[1-3]_XPSF_*.CCF

    Returns
    -------
    psf_grid       : numpy.ndarray, shape (n_energies, n_angles, 512, 512)
                     Each image is normalised to unit pixel sum.
    energies_ev    : numpy.ndarray, shape (n_energies,) — sorted, in eV
    thetas_arcmin  : numpy.ndarray, shape (n_angles,)   — sorted, in arcmin

    Raises
    ------
    ValueError  if fewer than 4 PSF images are found (likely wrong file).
    """
    raw = []   # list of (energy_eV, theta_arcmin, 512×512 array)

    with fits.open(ccf_path, memmap=True) as hdul:
        for hdu in hdul:
            if hdu.data is None or np.ndim(hdu.data) != 2:
                continue

            hdr    = hdu.header
            energy = hdr.get('ENERGY', None)
            theta  = hdr.get('THETA',  None)

            # Fallback: parse extension name e.g. '1500eV_03armin'
            if energy is None or theta is None:
                m = re.fullmatch(r'(\d+)eV_(\d+)armin',
                                 hdu.name.strip(), re.IGNORECASE)
                if m:
                    energy = float(m.group(1))
                    theta  = float(m.group(2))

            if energy is None or theta is None:
                continue   # header extension, not a PSF image

            img = hdu.data.astype(np.float64)
            s   = img.sum()
            if s <= 0:
                warnings.warn(
                    f"PSF image '{hdu.name}' in {os.path.basename(ccf_path)} "
                    "sums to zero — skipping.",
                    RuntimeWarning, stacklevel=2)
                continue
            img /= s
            raw.append((float(energy), float(theta), img))

    if len(raw) < 4:
        raise ValueError(
            f"Only {len(raw)} PSF images found in {ccf_path}. "
            "Expected 66 (11 energies × 6 off-axis angles). "
            "Check that the file is a valid XRT[1-3]_XPSF CCF.")

    # Build sorted unique grids
    energies_ev   = np.array(sorted({e for e, _, _ in raw}))
    thetas_arcmin = np.array(sorted({t for _, t, _ in raw}))
    n_e, n_t      = len(energies_ev), len(thetas_arcmin)

    ny, nx   = raw[0][2].shape
    psf_grid = np.zeros((n_e, n_t, ny, nx), dtype=np.float64)

    e_idx = {e: i for i, e in enumerate(energies_ev)}
    t_idx = {t: i for i, t in enumerate(thetas_arcmin)}

    for energy, theta, img in raw:
        psf_grid[e_idx[energy], t_idx[theta]] = img

    print(f"  PSF grid loaded: {n_e} energies × {n_t} angles "
          f"({energies_ev[0]:.0f}–{energies_ev[-1]:.0f} eV, "
          f"{thetas_arcmin[0]:.0f}–{thetas_arcmin[-1]:.0f} arcmin), "
          f"pscale={XMM_PSF_PSCALE}\" /pix")

    return psf_grid, energies_ev, thetas_arcmin


# ---------------------------------------------------------------------------
# 2-D bilinear interpolation
# ---------------------------------------------------------------------------

def interpolate_xmm_psf(energy_ev: float, theta_arcmin: float,
                         psf_grid: np.ndarray,
                         energies: np.ndarray,
                         thetas: np.ndarray) -> np.ndarray:
    """
    Interpolate the XMM PSF at an arbitrary (energy, off-axis angle) point
    using pixel-by-pixel 2-D bilinear interpolation.

    The interpolation uses the four nearest grid cells:

        PSF = (1-fe)*(1-ft)*PSF[ie_lo, it_lo]
            +    fe *(1-ft)*PSF[ie_hi, it_lo]
            + (1-fe)*   ft *PSF[ie_lo, it_hi]
            +    fe *   ft *PSF[ie_hi, it_hi]

    where fe = fractional position between ie_lo and ie_hi in energy,
    and   ft = fractional position between it_lo and it_hi in off-axis angle.

    Parameters
    ----------
    energy_ev    : float  — band-centre energy in eV
    theta_arcmin : float  — source off-axis angle in arcmin
    psf_grid     : (n_E, n_θ, ny, nx) array from load_xmm_psf_grid()
    energies     : (n_E,) sorted energy grid in eV
    thetas       : (n_θ,) sorted off-axis angle grid in arcmin

    Returns
    -------
    psf : (ny, nx) float array, normalised to unit pixel sum
    """
    def _bracket(val, grid):
        """Return (i_lo, i_hi, frac) for val in grid. Clamps at edges."""
        if val <= grid[0]:
            return 0, 0, 0.0
        if val >= grid[-1]:
            n = len(grid) - 1
            return n, n, 0.0
        i_hi = int(np.searchsorted(grid, val, side='right'))
        i_lo = i_hi - 1
        frac = (val - grid[i_lo]) / (grid[i_hi] - grid[i_lo])
        return i_lo, i_hi, float(frac)

    ie_lo, ie_hi, fe = _bracket(energy_ev,    energies)
    it_lo, it_hi, ft = _bracket(theta_arcmin, thetas)

    psf = (
        (1 - fe) * (1 - ft) * psf_grid[ie_lo, it_lo] +
        fe       * (1 - ft) * psf_grid[ie_hi, it_lo] +
        (1 - fe) * ft       * psf_grid[ie_lo, it_hi] +
        fe       * ft       * psf_grid[ie_hi, it_hi]
    )

    return _renorm(psf)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_xmm_eef(cfg: XMMConfig, instrument: str, evt_hdr,
                     r_src_arcsec: float,
                     e_lo_kev: float, e_hi_kev: float) -> dict:
    """
    Compute the EEF for an XMM-Newton source aperture from the CCF PSF.

    Steps
    -----
    1. Resolve the CCF PSF file for this instrument via cfg.resolve_psf_dir().
    2. Compute source off-axis angle θ from the EVENTS header pointing
       (RA_PNT / DEC_PNT — written by SAS epproc/emproc).
    3. Load the full (energy × angle) PSF grid from the CCF.
    4. Interpolate the PSF at band-centre energy and θ using 2-D bilinear
       interpolation across both axes simultaneously.
    5. Integrate the EEF within r_src_arcsec of the PSF image centre.

    Parameters
    ----------
    cfg          : XMMConfig
    instrument   : 'MOS1', 'MOS2', or 'PN'
    evt_hdr      : astropy.io.fits.Header  — EVENTS extension header
    r_src_arcsec : float  — source aperture radius in arcseconds
    e_lo_kev     : float  — lower energy bound (keV)
    e_hi_kev     : float  — upper energy bound (keV)

    Returns
    -------
    dict with keys:
        eef              float        EEF at r_src_arcsec  (primary value)
        theta_arcmin     float        source off-axis angle (arcmin)
        pointing_ra      float        telescope pointing RA (degrees)
        pointing_dec     float        telescope pointing Dec (degrees)
        psf_file         str          CCF path used
        energy_ev        float        band-centre energy used for interpolation
        extrapolated     bool         True if θ > 15' (beyond CCF limit)
        eef_capped       float|None   EEF at θ = 15' if extrapolated, else None
    """
    # -- Step 1: locate CCF file ----------------------------------------------
    psf_dir  = cfg.resolve_psf_dir()
    psf_glob = cfg.PSF_GLOBS[instrument]
    matches  = glob.glob(os.path.join(psf_dir, psf_glob))
    if not matches:
        raise FileNotFoundError(
            f"No XMM PSF CCF file found for {instrument} "
            f"(pattern: {psf_glob}) in:\n  {psf_dir}\n"
            "Download XRT[1-3]_XPSF_*.CCF from:\n"
            "  https://www.cosmos.esa.int/web/xmm-newton/current-calibration-files\n"
            "and place them in the psf_dir (or the bundled xray_uplim/data/xmm/psf/).")

    # Use the most recent CCF version (alphabetically last = highest version number)
    ccf_path = sorted(matches)[-1]

    # -- Step 2: off-axis angle -----------------------------------------------
    src_coord = parse_coord(cfg.ra, cfg.dec)
    theta, pt_ra, pt_dec = off_axis_angle(
        src_coord.ra.deg, src_coord.dec.deg, evt_hdr)

    extrapolated = False
    eef_capped   = None

    if theta > XMM_MAX_OFFAXIS:
        warnings.warn(
            f"{instrument}: source off-axis angle {theta:.2f}' exceeds the "
            f"XMM CCF PSF limit ({XMM_MAX_OFFAXIS:.0f}'). "
            f"EEF will be computed at {XMM_MAX_OFFAXIS:.0f}' (capped). "
            "For highly off-axis sources the PSF is significantly degraded "
            "and the upper limit should be treated with caution.",
            UserWarning, stacklevel=2)
        extrapolated = True

    theta_use = min(theta, XMM_MAX_OFFAXIS)

    # -- Step 3: load PSF grid ------------------------------------------------
    psf_grid, energies, thetas = load_xmm_psf_grid(ccf_path)

    # -- Step 4: interpolate --------------------------------------------------
    # Band-centre energy in eV
    energy_ev = (e_lo_kev + e_hi_kev) / 2.0 * 1000.0

    # Warn if band centre is outside the CCF energy range (unusual)
    if energy_ev < energies[0] or energy_ev > energies[-1]:
        warnings.warn(
            f"{instrument}: band-centre energy {energy_ev:.0f} eV is outside "
            f"the CCF PSF grid ({energies[0]:.0f}–{energies[-1]:.0f} eV). "
            "Clamping to the nearest grid edge.",
            UserWarning, stacklevel=2)

    psf_image = interpolate_xmm_psf(energy_ev, theta_use,
                                     psf_grid, energies, thetas)

    # If extrapolated, also compute EEF at the capped angle for reference
    if extrapolated:
        psf_capped = interpolate_xmm_psf(energy_ev, XMM_MAX_OFFAXIS,
                                          psf_grid, energies, thetas)
        eef_capped = integrate_eef(psf_capped, r_src_arcsec, XMM_PSF_PSCALE)

    # -- Step 5: integrate EEF ------------------------------------------------
    eef = integrate_eef(psf_image, r_src_arcsec, XMM_PSF_PSCALE)

    print(f"  {instrument}: EEF={eef:.4f}  "
          f"θ={theta:.2f}'  "
          f"E_centre={energy_ev:.0f} eV  "
          f"r={r_src_arcsec:.1f}\"")

    return {
        'eef':          eef,
        'theta_arcmin': theta,
        'pointing_ra':  pt_ra,
        'pointing_dec': pt_dec,
        'psf_file':     ccf_path,
        'energy_ev':    energy_ev,
        'extrapolated': extrapolated,
        'eef_capped':   eef_capped,
    }
