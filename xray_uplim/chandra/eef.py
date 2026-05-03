"""
xray_uplim.chandra.eef
-----------------------
Encircled Energy Fraction for Chandra ACIS.

Chandra's on-axis PSF is nearly diffraction-limited (FWHM ≈ 0.5 arcsec)
but degrades substantially off-axis.  We model it as a 2-D Gaussian for
simplicity:

    EEF(r) = 1 − exp(−r² / (2σ²))
    σ = FWHM / (2√(2 ln 2))

This is accurate on-axis and a reasonable approximation at moderate
off-axis angles where the PSF is still roughly Gaussian.  For sources
with strong PSF non-Gaussianity (far off-axis), a more detailed model
(e.g. MARX simulation or Sherpa PSF library) would be needed — but in
those cases the EEF is usually close enough to 1 that the correction is
minor compared to other uncertainties.

Usage
-----
>>> from xray_uplim.chandra.eef import compute_chandra_eef
>>> info = compute_chandra_eef(src_radius_arcsec=5.0, psf_fwhm_arcsec=0.9)
>>> info['eef']   # 0.9999...
"""

import math


def compute_chandra_eef(src_radius_arcsec: float,
                        psf_fwhm_arcsec: float) -> dict:
    """
    Gaussian EEF for Chandra ACIS at the given aperture radius.

    Parameters
    ----------
    src_radius_arcsec : float
        Extraction aperture radius in arcseconds.
    psf_fwhm_arcsec : float
        Gaussian FWHM of the PSF in arcseconds.
        On-axis ACIS: ~0.5–1.0 arcsec.
        Increase for off-axis sources (PSF degrades quickly beyond ~5').

    Returns
    -------
    dict with keys:
        eef                 : float  — encircled energy fraction (0 < EEF ≤ 1)
        psf_fwhm_arcsec     : float  — FWHM used
        src_radius_arcsec   : float  — aperture radius used
        sigma_arcsec        : float  — Gaussian sigma used
    """
    sigma = psf_fwhm_arcsec / (2.0 * math.sqrt(2.0 * math.log(2.0)))
    eef   = 1.0 - math.exp(-0.5 * (src_radius_arcsec / sigma) ** 2)
    return {
        'eef'               : min(eef, 1.0),
        'psf_fwhm_arcsec'   : psf_fwhm_arcsec,
        'src_radius_arcsec' : src_radius_arcsec,
        'sigma_arcsec'      : sigma,
    }
