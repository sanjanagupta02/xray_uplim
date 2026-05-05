"""
xray_uplim.exposure
-------------------
Effective exposure time from an observatory exposure map.

The exposure map encodes vignetting, dead-time, and chip gaps in a single
image (units: seconds).  For a non-detection we summarise the pixel values
inside the source aperture into one number, t_eff, which is then used to
convert count upper limits into count-rate upper limits.

Why not use the header LIVETIME?
    The LIVETIME keyword gives the total good-time live time for the whole
    detector, ignoring vignetting.  At the source position — especially if
    it is off-axis — the effective exposure can be significantly lower.
    The exposure map gives the correct per-pixel value.

Statistic choices
-----------------
median       — recommended for non-detections.  No PSF shape assumption,
               robust against partially-clipped chip-gap edge pixels.
mean         — fine when vignetting variation across the aperture is small
               (typically < 5% within a 60" circle near the aimpoint).
psf_weighted — sum(PSF_i * exp_i) / sum(PSF_i) with a circular Gaussian PSF.
               Physically motivated for a point source but only reliable
               on-axis; the NuSTAR PSF broadens and becomes asymmetric
               off-axis, so this is printed as a diagnostic only.
"""

import numpy as np


def circle_mask(shape, cx, cy, r):
    """
    Boolean mask of pixels whose centre lies within radius r of (cx, cy).

    Parameters
    ----------
    shape : (ny, nx)
    cx, cy : float  — centre (0-indexed pixel coordinates)
    r      : float  — radius in pixels

    Returns
    -------
    numpy bool array of shape `shape`
    """
    y, x = np.ogrid[:shape[0], :shape[1]]
    return (x - cx)**2 + (y - cy)**2 <= r**2


def gaussian_psf_weights(shape, cx, cy, fwhm_pix, mask):
    """
    Circular Gaussian PSF weights evaluated at each pixel, zeroed outside mask.

    Parameters
    ----------
    shape    : (ny, nx)
    cx, cy   : float  — source pixel position
    fwhm_pix : float  — PSF FWHM in pixels
    mask     : bool array  — pixels to include

    Returns
    -------
    weight array of shape `shape`, normalised to peak = 1 at (cx, cy)
    """
    sigma = fwhm_pix / 2.355
    y, x  = np.ogrid[:shape[0], :shape[1]]
    w     = np.exp(-((x - cx)**2 + (y - cy)**2) / (2 * sigma**2))
    w[~mask] = 0.0
    return w


def compute_exposure_area_ratio(exp_data, cx_src, cy_src, r_src_pix,
                                bkg_mode,
                                r_bkg_inner_pix=None, r_bkg_outer_pix=None,
                                cx_bkg=None, cy_bkg=None, r_bkg_pix=None):
    """
    Exposure-map-weighted source-to-background area ratio (Tier 1).

    Replaces the purely geometric π*r_src² / π*r_bkg² ratio with one
    that weights by the actual exposure-map pixel values in each region.
    This accounts for vignetting gradients between the source and
    background apertures: when the background annulus extends to larger
    off-axis angles, its effective exposure per pixel is lower than at
    the source position, so the geometric ratio under-estimates how many
    background counts to subtract.

    Parameters
    ----------
    exp_data   : 2-D float array  — exposure map in seconds
    cx_src, cy_src : float        — source centre in exposure-map pixels
    r_src_pix  : float            — source aperture radius in pixels
    bkg_mode   : str              — 'annulus' or 'manual'

    For annulus mode (bkg_mode == 'annulus'):
        r_bkg_inner_pix : float  — inner radius of the background annulus
        r_bkg_outer_pix : float  — outer radius of the background annulus

    For manual mode (bkg_mode == 'manual'):
        cx_bkg, cy_bkg : float   — background circle centre in exp-map pixels
        r_bkg_pix      : float   — background circle radius in pixels

    Returns
    -------
    area_ratio : float
        sum(exp_map pixels in source aperture) /
        sum(exp_map pixels in background region)

    Raises
    ------
    RuntimeError
        If the summed background exposure is zero (background region
        outside the exposure map or fully vignetted).
    ValueError
        If bkg_mode is not 'annulus' or 'manual'.
    """
    src_mask    = circle_mask(exp_data.shape, cx_src, cy_src, r_src_pix)
    exp_sum_src = np.sum(exp_data[src_mask])

    if bkg_mode == 'annulus':
        outer_mask = circle_mask(exp_data.shape, cx_src, cy_src, r_bkg_outer_pix)
        inner_mask = circle_mask(exp_data.shape, cx_src, cy_src, r_bkg_inner_pix)
        bkg_mask   = outer_mask & ~inner_mask
    elif bkg_mode == 'manual':
        bkg_mask = circle_mask(exp_data.shape, cx_bkg, cy_bkg, r_bkg_pix)
    else:
        raise ValueError(
            f"bkg_mode must be 'annulus' or 'manual', not '{bkg_mode}'")

    exp_sum_bkg = np.sum(exp_data[bkg_mask])
    if exp_sum_bkg <= 0:
        raise RuntimeError(
            "Zero summed exposure in background region — "
            "check that the background aperture overlaps the exposure map.")

    return exp_sum_src / exp_sum_bkg


def compute_exposure_stats(exp_data, cx, cy, r_src_pix, fwhm_pix):
    """
    Compute all three exposure summary statistics inside the source aperture.

    Parameters
    ----------
    exp_data  : 2-D float array  — exposure map in seconds
    cx, cy    : float            — source position in exposure-map pixels
    r_src_pix : float            — source aperture radius in pixels
    fwhm_pix  : float            — PSF FWHM in pixels (for psf_weighted)

    Returns
    -------
    stats : dict
        {'median': float, 'mean': float, 'psf_weighted': float}
    meta : dict
        {'n_pix_total': int, 'n_pix_nonzero': int, 'exp_values': ndarray}
        exp_values contains only the non-zero pixel values.
    """
    src_mask   = circle_mask(exp_data.shape, cx, cy, r_src_pix)
    exp_in_apt = exp_data[src_mask]
    good       = exp_in_apt > 0

    if good.sum() == 0:
        raise RuntimeError(
            "No non-zero exposure-map pixels inside the source aperture.\n"
            "Double-check your coordinates and that the exposure map and "
            "event file are aligned.")

    t_median = float(np.median(exp_in_apt[good]))
    t_mean   = float(np.mean(exp_in_apt[good]))

    # PSF-weighted mean — on-axis diagnostic only
    w_map  = gaussian_psf_weights(exp_data.shape, cx, cy, fwhm_pix, src_mask)
    w_vals = w_map[src_mask]
    t_psfw = float(np.sum(w_vals[good] * exp_in_apt[good])
                   / np.sum(w_vals[good]))

    stats = {
        'median':       t_median,
        'mean':         t_mean,
        'psf_weighted': t_psfw,
    }
    meta = {
        'n_pix_total':   int(exp_in_apt.size),
        'n_pix_nonzero': int(good.sum()),
        'exp_values':    exp_in_apt[good],
    }
    return stats, meta
