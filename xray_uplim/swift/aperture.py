"""
xray_uplim.swift.aperture
--------------------------
Circular aperture photometry for Swift XRT event files.

Counts photons within circular source and background apertures directly
from the filtered event table (X, Y sky pixel columns), then computes
effective exposure from the XRT exposure map.

The event file and exposure map share the same sky pixel grid for Swift
XRT (both at ~2.36 arcsec/pix), but we resolve coordinates independently
for each using the appropriate WCS so the code is robust to any pixel
scale or grid offset.

Public API
----------
extract_src_bkg_counts(events, evt_hdr, cfg, mode, bkg_cx_evt, bkg_cy_evt)
    → (n_src, n_bkg, area_ratio, cx_evt, cy_evt, pscale_evt)

extract_exposure(exp_data, exp_hdr, cfg)
    → (exp_stats, exp_meta, cx_exp, cy_exp)
"""

import warnings
import numpy as np

from ..coords  import parse_coord, sky_to_evt_pixel, sky_to_img_pixel
from ..exposure import circle_mask, compute_exposure_stats
from .config   import SwiftConfig


def _arcsec_to_pix(radius_arcsec, pscale):
    if pscale <= 0:
        raise ValueError(f"Invalid pixel scale: {pscale} arcsec/pix")
    return radius_arcsec / pscale


# ---------------------------------------------------------------------------
# Count extraction
# ---------------------------------------------------------------------------

def extract_src_bkg_counts(events, evt_hdr, cfg: SwiftConfig, mode: str,
                           bkg_cx_evt=None, bkg_cy_evt=None):
    """
    Count photons in source and background apertures from a Swift XRT
    event table.

    Source aperture  : circle of radius cfg.src_radius_arcsec centred on
                       (cfg.ra, cfg.dec).
    Background region: controlled by cfg.bkg_mode —
        'annulus' : annulus from (src_radius × bkg_inner_factor) to
                    bkg_radius_arcsec, concentric with the source.
        'manual'  : circle of radius cfg.bkg_radius_arcsec centred on
                    (cfg.bkg_ra, cfg.bkg_dec), or on the pre-computed
                    pixel position (bkg_cx_evt, bkg_cy_evt) if supplied.

    Parameters
    ----------
    events      : astropy.table.Table   filtered events from load_events()
    evt_hdr     : fits.Header           merged event file header
    cfg         : SwiftConfig
    mode        : str                   'PC' or 'WT' (for diagnostics only)
    bkg_cx_evt,
    bkg_cy_evt  : float, optional
        Pre-computed background centre in event-file pixels.
        Only used when cfg.bkg_mode == 'manual'.

    Returns
    -------
    n_src      : int
    n_bkg      : int
    area_ratio : float  — src_area / bkg_area  (for background scaling)
    cx_evt     : float  — source centre X in event-file pixels
    cy_evt     : float  — source centre Y in event-file pixels
    pscale_evt : float  — event-file pixel scale (arcsec/pix)
    """
    src_coord = parse_coord(cfg.ra, cfg.dec)
    cx_evt, cy_evt, pscale_evt = sky_to_evt_pixel(
        src_coord.ra.deg, src_coord.dec.deg, evt_hdr)

    r_src_pix = _arcsec_to_pix(cfg.src_radius_arcsec, pscale_evt)

    x = np.asarray(events['X'], dtype=float)
    y = np.asarray(events['Y'], dtype=float)

    # Source counts
    src_dist2 = (x - cx_evt)**2 + (y - cy_evt)**2
    in_src    = src_dist2 <= r_src_pix**2
    n_src     = int(in_src.sum())

    # Background counts
    if cfg.bkg_mode == 'annulus':
        r_inner_pix = _arcsec_to_pix(
            cfg.src_radius_arcsec * cfg.bkg_inner_factor, pscale_evt)
        r_outer_pix = _arcsec_to_pix(cfg.bkg_radius_arcsec, pscale_evt)
        in_bkg   = (src_dist2 >= r_inner_pix**2) & \
                   (src_dist2 <= r_outer_pix**2)
        area_src = np.pi * r_src_pix**2
        area_bkg = np.pi * (r_outer_pix**2 - r_inner_pix**2)

    elif cfg.bkg_mode == 'manual':
        if bkg_cx_evt is not None and bkg_cy_evt is not None:
            bcx, bcy = bkg_cx_evt, bkg_cy_evt
        else:
            bkg_coord = parse_coord(cfg.bkg_ra, cfg.bkg_dec)
            bcx, bcy, _ = sky_to_evt_pixel(
                bkg_coord.ra.deg, bkg_coord.dec.deg, evt_hdr)

        r_bkg_pix = _arcsec_to_pix(cfg.bkg_radius_arcsec, pscale_evt)
        bkg_dist2 = (x - bcx)**2 + (y - bcy)**2
        in_bkg    = bkg_dist2 <= r_bkg_pix**2
        area_src  = np.pi * r_src_pix**2
        area_bkg  = np.pi * r_bkg_pix**2

    else:
        raise ValueError(
            f"bkg_mode must be 'annulus' or 'manual', not '{cfg.bkg_mode}'.")

    n_bkg = int(in_bkg.sum())

    if area_bkg <= 0:
        raise RuntimeError("Background region has zero area.")

    area_ratio = area_src / area_bkg

    # Diagnostics
    label = f"XRT-{mode}"
    if n_src == 0:
        warnings.warn(
            f"{label}: 0 counts in source aperture "
            f"(r={cfg.src_radius_arcsec}\") — "
            "upper limit will be based on background alone.",
            RuntimeWarning, stacklevel=2)
    if n_bkg == 0:
        warnings.warn(
            f"{label}: 0 counts in background region — "
            "check region placement and bkg_mode.",
            RuntimeWarning, stacklevel=2)

    print(f"  {label}: src={n_src} cts  "
          f"bkg={n_bkg} cts  "
          f"area_ratio={area_ratio:.4f}  "
          f"(net≈{n_src - area_ratio * n_bkg:.1f} cts)")

    return n_src, n_bkg, area_ratio, cx_evt, cy_evt, pscale_evt


# ---------------------------------------------------------------------------
# Exposure extraction
# ---------------------------------------------------------------------------

def extract_exposure(exp_data, exp_hdr, cfg: SwiftConfig):
    """
    Compute effective exposure time inside the source aperture from the
    Swift XRT exposure map.

    Parameters
    ----------
    exp_data : numpy.ndarray  — 2-D exposure map in seconds
    exp_hdr  : fits.Header    — exposure map header
    cfg      : SwiftConfig

    Returns
    -------
    exp_stats : dict   {'median': float, 'mean': float, 'psf_weighted': float}
    exp_meta  : dict   {'n_pix_total': int, 'n_pix_nonzero': int,
                        'exp_values': ndarray}
    cx_exp    : float  — source centre X in exposure-map pixels
    cy_exp    : float  — source centre Y in exposure-map pixels
    pscale_exp: float  — exposure-map pixel scale (arcsec/pix)
    """
    src_coord = parse_coord(cfg.ra, cfg.dec)
    cx_exp, cy_exp, pscale_exp = sky_to_img_pixel(
        src_coord.ra.deg, src_coord.dec.deg, exp_hdr)

    r_src_pix_exp = _arcsec_to_pix(cfg.src_radius_arcsec, pscale_exp)
    psf_fwhm_pix  = _arcsec_to_pix(cfg.psf_fwhm_arcsec,   pscale_exp)

    ny, nx = exp_data.shape
    if not (0 <= cx_exp < nx and 0 <= cy_exp < ny):
        warnings.warn(
            f"Source position ({cx_exp:.1f}, {cy_exp:.1f}) is outside the "
            f"exposure map ({nx}×{ny} pix). "
            "Check RA/Dec and that the exposure map covers the source.",
            RuntimeWarning, stacklevel=2)

    exp_stats, exp_meta = compute_exposure_stats(
        exp_data, cx_exp, cy_exp, r_src_pix_exp, psf_fwhm_pix)

    print(f"  XRT: exposure  "
          f"median={exp_stats['median']:.0f} s  "
          f"mean={exp_stats['mean']:.0f} s  "
          f"({exp_meta['n_pix_nonzero']}/{exp_meta['n_pix_total']} "
          f"pix nonzero)")

    return exp_stats, exp_meta, cx_exp, cy_exp, pscale_exp
