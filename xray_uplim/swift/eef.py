"""
xray_uplim.swift.eef
---------------------
Encircled Energy Fraction (EEF) for Swift XRT using the King + Gaussian
PSF model from the XIMAGE calibration file psfconst_xrt.fits.

Background
----------
XIMAGE/SOSTA computes PSF corrections using swift_psf.c, which reads
psfconst_xrt.fits and applies:

    PSF(r) = P0 * Gauss(sigma=P1) + (1 - P0) * King(rc=P2, eta=P3)

Each parameter P_i is a bilinear function of energy and off-axis angle:

    P_i = COEF0 + COEF1*(offaxis*10) + COEF2*(energy*10)
                + COEF3*(offaxis*10)*(energy*10)

    where  energy  is in keV  and  offaxis  is in arcmin.
    (The ×10 scaling is hard-coded in swift_psf.c: scl_ene = scl_off = 10.)

The EEF (fraction of total PSF flux within circular radius r arcsec) is:

    totflux = 2π·P0·P1² + π·P2²·(1−P0)/(P3−1)        [normalisation]
    fg      = 2π·P0·P1²·[1 − exp(−rad²/(2·P1²))]       [Gaussian term]
    fk      = π·P2²·(1−P0)/(1−P3)·[(1+(rad/P2)²)^(1−P3) − 1]  [King term]
    EEF(r)  = (fg + fk) / totflux

    where  rad = r / 2.36  (arcsec → XRT sky pixels, as in swift_psf.c).

In practice P0 = 0 for all current calibrations → pure King profile:

    EEF(r) = 1 − [1 + (rad/rc)²]^(1−eta)

Public API
----------
compute_swift_eef(cfg, evt_hdr, r_src_arcsec, e_lo_kev, e_hi_kev)
    → dict with keys: eef, theta_arcmin, energy_ev,
                      pointing_ra, pointing_dec, psf_file,
                      extrapolated, eef_capped
"""

import os
import math
import warnings
import numpy as np
from astropy.io   import fits
from astropy.table import Table

from ..coords import parse_coord


# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------

SWIFT_XRT_PSCALE    = 2.36    # arcsec/pixel (sky pixel scale in event files)
SWIFT_PSF_SCALE     = 10.0    # hard-coded scaling in swift_psf.c
SWIFT_MAX_OFFAXIS   = 12.0    # arcmin — XRT field of view radius; warn beyond


# ---------------------------------------------------------------------------
# Coefficient loading (cached)
# ---------------------------------------------------------------------------

_psf_cache = {}   # path → par array


def _load_psf_coeffs(psf_file):
    """
    Read the King+Gaussian PSF coefficients from psfconst_xrt.fits.

    Returns
    -------
    par : ndarray, shape (4, 4)
        par[col_idx][row_idx] where
            col_idx = 0..3  →  COEF0..COEF3
            row_idx = 0..3  →  P0..P3
        Matches the indexing in swift_psf.c:
            c[i] = par[0][i] + offaxis*par[1][i]
                             + energy*par[2][i]
                             + offaxis*energy*par[3][i]
        (with energy and offaxis already multiplied by SWIFT_PSF_SCALE=10)
    """
    if psf_file in _psf_cache:
        return _psf_cache[psf_file]

    with fits.open(psf_file) as hdul:
        tbl = Table(hdul[1].data)

    # Rows are ordered P0, P1, P2, P3 — verify
    par_names = [str(p).strip() for p in tbl['PAR']]
    expected  = ['P0', 'P1', 'P2', 'P3']
    if par_names != expected:
        raise ValueError(
            f"Unexpected PAR row order in {psf_file}: {par_names}. "
            f"Expected {expected}.")

    # par[col_idx] = COEF{col_idx} column values for all 4 rows (P0..P3)
    par = np.zeros((4, 4), dtype=float)
    for col_i, coef_col in enumerate(['COEF0', 'COEF1', 'COEF2', 'COEF3']):
        par[col_i] = np.asarray(tbl[coef_col], dtype=float)

    _psf_cache[psf_file] = par
    return par


# ---------------------------------------------------------------------------
# Off-axis angle
# ---------------------------------------------------------------------------

def _off_axis_angle(src_ra_deg, src_dec_deg, evt_hdr):
    """
    Compute off-axis angle (arcmin) from source to XRT pointing direction.

    Uses RA_NOM / DEC_NOM from the event file header (the commanded
    pointing, which equals the optical axis to sufficient accuracy for
    PSF purposes).  Falls back to RA_OBJ / DEC_OBJ if NOM keywords
    are absent.

    Returns
    -------
    theta_arcmin  : float
    pointing_ra   : float  (deg)
    pointing_dec  : float  (deg)
    """
    pt_ra  = evt_hdr.get('RA_NOM',  evt_hdr.get('RA_OBJ',  None))
    pt_dec = evt_hdr.get('DEC_NOM', evt_hdr.get('DEC_OBJ', None))

    if pt_ra is None or pt_dec is None:
        warnings.warn(
            "RA_NOM/DEC_NOM and RA_OBJ/DEC_OBJ not found in event header. "
            "Assuming on-axis (theta=0). EEF may be slightly overestimated.",
            UserWarning, stacklevel=3)
        return 0.0, float(src_ra_deg), float(src_dec_deg)

    pt_ra  = float(pt_ra)
    pt_dec = float(pt_dec)

    # Great-circle separation (small-angle haversine)
    cos_dec = math.cos(math.radians(pt_dec))
    d_ra    = (src_ra_deg - pt_ra) * cos_dec
    d_dec   = src_dec_deg - pt_dec
    theta_deg    = math.sqrt(d_ra**2 + d_dec**2)
    theta_arcmin = theta_deg * 60.0

    return theta_arcmin, pt_ra, pt_dec


# ---------------------------------------------------------------------------
# EEF at a single (energy, off-axis, radius)
# ---------------------------------------------------------------------------

def _compute_eef_single(par, energy_kev, theta_arcmin, r_arcsec):
    """
    Evaluate King+Gaussian EEF at one energy, off-axis angle, and radius.

    Mirrors swift_psf.c exactly, including the ×10 scaling and the
    arcsec→pixel conversion for the radius.

    Parameters
    ----------
    par           : ndarray (4,4)  — from _load_psf_coeffs()
    energy_kev    : float
    theta_arcmin  : float
    r_arcsec      : float

    Returns
    -------
    eef : float in [0, 1]
    """
    # Apply the ×10 scaling from swift_psf.c
    e   = energy_kev    * SWIFT_PSF_SCALE
    th  = theta_arcmin  * SWIFT_PSF_SCALE

    # Compute PSF parameters P0..P3
    c = np.array([
        par[0][i] + th * par[1][i] + e * par[2][i] + th * e * par[3][i]
        for i in range(4)
    ])
    P0, P1, P2, P3 = c   # Gauss fraction, Gauss sigma, King rc, King eta

    # Convert radius: arcsec → XRT sky pixels (as in swift_psf.c)
    rad = r_arcsec / SWIFT_XRT_PSCALE

    # Guard against degenerate parameters
    if (P3 - 1.0) == 0.0:
        warnings.warn("King eta = 1.0 (divergent PSF). Returning EEF=1.",
                      RuntimeWarning)
        return 1.0

    # Total flux normalisation
    gauss_flux = 2.0 * math.pi * P0 * P1**2  if P0 > 0 and P1 > 0 else 0.0
    king_flux  = math.pi * P2**2 * (1.0 - P0) / (P3 - 1.0)
    totflux    = gauss_flux + king_flux

    if totflux <= 0.0:
        warnings.warn("PSF total flux ≤ 0. Returning EEF=1.", RuntimeWarning)
        return 1.0

    # Gaussian enclosed flux
    if P0 > 0 and P1 > 0:
        fg = gauss_flux * (1.0 - math.exp(-rad**2 / (2.0 * P1**2)))
    else:
        fg = 0.0

    # King enclosed flux
    king_norm = math.pi * P2**2 * (1.0 - P0) / (1.0 - P3)
    fk = king_norm * ((1.0 + (rad / P2)**2)**(1.0 - P3) - 1.0)

    eef = (fg + fk) / totflux
    return float(np.clip(eef, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def _spectral_weighted_eef(par, theta_arcmin, r_src_arcsec,
                            e_lo_kev, e_hi_kev, gamma=2.0, n_samples=20):
    """
    Compute the spectral-weighted EEF by numerically integrating over the band.

    EEF_eff = integral_{e_lo}^{e_hi} EEF(E, theta) * E^{-gamma} dE
              / integral_{e_lo}^{e_hi} E^{-gamma} dE

    Uses uniform sampling with the trapezoidal rule.

    Returns
    -------
    eef_eff     : float   weighted EEF
    energy_eff  : float   spectral-weighted mean energy (keV) — for display only
    """
    energies = np.linspace(e_lo_kev, e_hi_kev, n_samples)

    # Guard: if gamma is very close to zero treat all weights as equal
    if abs(gamma) < 1e-9:
        weights = np.ones(n_samples)
    else:
        # Avoid division by zero at E=0 (shouldn't happen for X-ray bands)
        weights = energies ** (-gamma)

    eef_values = np.array([
        _compute_eef_single(par, e, theta_arcmin, r_src_arcsec)
        for e in energies
    ])

    total_weight = np.trapz(weights, energies)
    if total_weight <= 0.0:
        # Fallback: plain midpoint
        mid = 0.5 * (e_lo_kev + e_hi_kev)
        return _compute_eef_single(par, mid, theta_arcmin, r_src_arcsec), mid

    eef_eff    = float(np.trapz(weights * eef_values, energies) / total_weight)
    energy_eff = float(np.trapz(weights * energies,   energies) / total_weight)
    return float(np.clip(eef_eff, 0.0, 1.0)), energy_eff


def compute_swift_eef(cfg, evt_hdr, r_src_arcsec, e_lo_kev, e_hi_kev,
                      psf_gamma=2.0):
    """
    Compute the Swift XRT EEF for a given source aperture radius.

    The EEF is computed as a spectral-weighted average over the energy band,
    using the King+Gaussian PSF model from psfconst_xrt.fits.  This matches
    the approach used by NuSTAR (spectral combination of per-band PSF files).

        EEF_eff = integral EEF(E, theta) * E^{-gamma} dE
                  / integral E^{-gamma} dE

    evaluated numerically at 20 energy points across the band.

    Parameters
    ----------
    cfg          : SwiftConfig
    evt_hdr      : fits.Header   merged event-file header
    r_src_arcsec : float         source aperture radius in arcsec
    e_lo_kev     : float         energy band lower bound (keV)
    e_hi_kev     : float         energy band upper bound (keV)
    psf_gamma    : float         photon index for spectral weighting (default 2.0)

    Returns
    -------
    dict with keys:
        eef           : float   spectral-weighted EEF at r_src_arcsec (0–1)
        theta_arcmin  : float   source off-axis angle (arcmin)
        energy_kev    : float   spectral-weighted mean energy (keV) — display only
        psf_gamma     : float   photon index used
        pointing_ra   : float   XRT pointing RA (deg)
        pointing_dec  : float   XRT pointing Dec (deg)
        psf_file      : str     path to psfconst_xrt.fits used
        extrapolated  : bool    True if theta > SWIFT_MAX_OFFAXIS
        eef_capped    : float or None   EEF at max off-axis if extrapolated
    """
    # Locate and load PSF coefficient file
    psf_file = cfg.resolve_psf_file()
    par      = _load_psf_coeffs(psf_file)

    # Source coordinates
    src_coord = parse_coord(cfg.ra, cfg.dec)

    # Off-axis angle
    theta_arcmin, pt_ra, pt_dec = _off_axis_angle(
        src_coord.ra.deg, src_coord.dec.deg, evt_hdr)

    # Extrapolation check
    extrapolated = theta_arcmin > SWIFT_MAX_OFFAXIS
    eef_capped   = None
    if extrapolated:
        warnings.warn(
            f"Source off-axis angle {theta_arcmin:.1f}' exceeds XRT "
            f"field of view ({SWIFT_MAX_OFFAXIS}'), likely a background "
            "observation. EEF capped at maximum off-axis value.",
            UserWarning, stacklevel=2)
        eef_capped, _ = _spectral_weighted_eef(
            par, SWIFT_MAX_OFFAXIS, r_src_arcsec, e_lo_kev, e_hi_kev, psf_gamma)
        theta_for_eef = SWIFT_MAX_OFFAXIS
    else:
        theta_for_eef = theta_arcmin

    eef, energy_eff_kev = _spectral_weighted_eef(
        par, theta_for_eef, r_src_arcsec, e_lo_kev, e_hi_kev, psf_gamma)

    print(f"  XRT PSF: theta={theta_arcmin:.2f}'  "
          f"E_eff={energy_eff_kev:.2f} keV (Gamma={psf_gamma:.1f})  "
          f"r={r_src_arcsec:.1f}\"  "
          f"EEF={eef:.4f}"
          + (" [EXTRAPOLATED]" if extrapolated else ""))

    return {
        'eef'          : eef,
        'theta_arcmin' : theta_arcmin,
        'energy_kev'   : energy_eff_kev,
        'psf_gamma'    : psf_gamma,
        'pointing_ra'  : pt_ra,
        'pointing_dec' : pt_dec,
        'psf_file'     : psf_file,
        'extrapolated' : extrapolated,
        'eef_capped'   : eef_capped,
    }
