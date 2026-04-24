"""
xray_uplim.xmm.aperture
------------------------
Circular aperture photometry for XMM-Newton EPIC event files.

Counts photons within circular source and background apertures directly
from the filtered event table (X, Y sky pixel columns), then computes
effective exposure from the eexpmap exposure map.

The exposure map and event file may have different pixel scales.
sky_to_evt_pixel()  converts RA/Dec to event-file pixels  (typically
    0.05 arcsec/pix for XMM — very fine because the column values are
    stored in units of 0.05" by SAS).
sky_to_img_pixel()  converts RA/Dec to exposure-map pixels (typically
    2 arcsec/pix from eexpmap with ximagebinsize=40).

All radius conversions are done per-coordinate-system so the two grids
never need to be resampled or aligned.

Public API
----------
extract_src_bkg_counts(events, evt_hdr, cfg, instrument)
    → (n_src, n_bkg, area_ratio, cx_evt, cy_evt, pscale_evt)

extract_exposure(exp_data, exp_hdr, cfg, instrument)
    → (exp_stats, exp_meta, cx_exp, cy_exp)
"""

import warnings
import numpy as np

from ..coords  import parse_coord, sky_to_evt_pixel, sky_to_img_pixel
from ..exposure import circle_mask, compute_exposure_stats
from .config   import XMMConfig


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _arcsec_to_pix(radius_arcsec, pscale):
    """Convert an aperture radius from arcseconds to pixels."""
    if pscale <= 0:
        raise ValueError(f"Invalid pixel scale: {pscale} arcsec/pix")
    return radius_arcsec / pscale


# ---------------------------------------------------------------------------
# Count extraction
# ---------------------------------------------------------------------------

def extract_src_bkg_counts(events, evt_hdr, cfg: XMMConfig, instrument: str,
                           bkg_cx_evt=None, bkg_cy_evt=None):
    """
    Count photons in source and background apertures from an event table.

    Source aperture  : circle of radius cfg.src_radius_arcsec centred on
                       (cfg.ra, cfg.dec).
    Background region: controlled by cfg.bkg_mode —
        'annulus' : annulus from (src_radius * bkg_inner_factor) to
                    bkg_radius_arcsec, concentric with the source.
        'manual'  : circle of radius cfg.bkg_radius_arcsec centred on
                    (cfg.bkg_ra, cfg.bkg_dec) — or on the pre-computed
                    pixel position (bkg_cx_evt, bkg_cy_evt) if supplied.

    Parameters
    ----------
    events      : astropy.table.Table  — filtered events from load_events()
    evt_hdr     : fits.Header           — EVENTS extension header
    cfg         : XMMConfig
    instrument  : 'MOS1', 'MOS2', or 'PN'
    bkg_cx_evt,
    bkg_cy_evt  : float, optional
        Pre-computed background centre in event-file pixels.
        Only used when cfg.bkg_mode == 'manual'.
        If not supplied, the centre is derived from cfg.bkg_ra / cfg.bkg_dec.

    Returns
    -------
    n_src      : int    — raw photon count in source aperture
    n_bkg      : int    — raw photon count in background region
    area_ratio : float  — (src aperture area) / (bkg region area)
                          used to scale background to source area:
                          net_src = n_src - area_ratio * n_bkg
    cx_evt     : float  — source centre in event-file pixels (X)
    cy_evt     : float  — source centre in event-file pixels (Y)
    pscale_evt : float  — pixel scale of event file (arcsec/pix)
    """
    # -- Source position in event-file pixels ---------------------------------
    src_coord = parse_coord(cfg.ra, cfg.dec)
    cx_evt, cy_evt, pscale_evt = sky_to_evt_pixel(
        src_coord.ra.deg, src_coord.dec.deg, evt_hdr)

    r_src_pix = _arcsec_to_pix(cfg.src_radius_arcsec, pscale_evt)

    x = np.asarray(events['X'], dtype=float)
    y = np.asarray(events['Y'], dtype=float)

    # -- Source counts --------------------------------------------------------
    src_dist2 = (x - cx_evt)**2 + (y - cy_evt)**2
    in_src    = src_dist2 <= r_src_pix**2
    n_src     = int(in_src.sum())

    # -- Background counts ----------------------------------------------------
    if cfg.bkg_mode == 'annulus':
        r_bkg_inner_pix = _arcsec_to_pix(
            cfg.src_radius_arcsec * cfg.bkg_inner_factor, pscale_evt)
        r_bkg_outer_pix = _arcsec_to_pix(cfg.bkg_radius_arcsec, pscale_evt)

        in_bkg = (
            (src_dist2 >= r_bkg_inner_pix**2) &
            (src_dist2 <= r_bkg_outer_pix**2)
        )
        area_src = np.pi * r_src_pix**2
        area_bkg = np.pi * (r_bkg_outer_pix**2 - r_bkg_inner_pix**2)

    elif cfg.bkg_mode == 'manual':
        # Resolve background centre
        if bkg_cx_evt is not None and bkg_cy_evt is not None:
            bcx, bcy = bkg_cx_evt, bkg_cy_evt
        else:
            bkg_coord = parse_coord(cfg.bkg_ra, cfg.bkg_dec)
            bcx, bcy, _ = sky_to_evt_pixel(
                bkg_coord.ra.deg, bkg_coord.dec.deg, evt_hdr)

        r_bkg_pix  = _arcsec_to_pix(cfg.bkg_radius_arcsec, pscale_evt)
        bkg_dist2  = (x - bcx)**2 + (y - bcy)**2
        in_bkg     = bkg_dist2 <= r_bkg_pix**2
        area_src   = np.pi * r_src_pix**2
        area_bkg   = np.pi * r_bkg_pix**2

    else:
        raise ValueError(
            f"bkg_mode must be 'annulus' or 'manual', not '{cfg.bkg_mode}'.")

    n_bkg = int(in_bkg.sum())

    if area_bkg <= 0:
        raise RuntimeError("Background region has zero area.")

    area_ratio = area_src / area_bkg

    # -- Diagnostics ----------------------------------------------------------
    if n_src == 0:
        warnings.warn(
            f"{instrument}: 0 counts in source aperture "
            f"(r={cfg.src_radius_arcsec}\") — "
            "upper limit will be based on background alone.",
            RuntimeWarning, stacklevel=2)

    if n_bkg == 0:
        warnings.warn(
            f"{instrument}: 0 counts in background region — "
            "check region placement and bkg_mode.",
            RuntimeWarning, stacklevel=2)

    print(f"  {instrument}: src={n_src} cts  "
          f"bkg={n_bkg} cts  "
          f"area_ratio={area_ratio:.4f}  "
          f"(net≈{n_src - area_ratio * n_bkg:.1f} cts)")

    return n_src, n_bkg, area_ratio, cx_evt, cy_evt, pscale_evt


# ---------------------------------------------------------------------------
# Exposure extraction
# ---------------------------------------------------------------------------

def extract_exposure(exp_data, exp_hdr, cfg: XMMConfig, instrument: str):
    """
    Compute effective exposure time inside the source aperture.

    Converts the source RA/Dec to exposure-map pixels (which have a
    different pixel scale from the event file — typically 2 arcsec/pix)
    and calls the shared compute_exposure_stats().

    Parameters
    ----------
    exp_data   : numpy.ndarray  — 2-D exposure map in seconds (from load_expmap)
    exp_hdr    : fits.Header    — primary image header
    cfg        : XMMConfig
    instrument : 'MOS1', 'MOS2', or 'PN'

    Returns
    -------
    exp_stats : dict
        {'median': float, 'mean': float, 'psf_weighted': float}
        All values in seconds.
    exp_meta  : dict
        {'n_pix_total': int, 'n_pix_nonzero': int, 'exp_values': ndarray}
    cx_exp    : float  — source centre in exposure-map pixels (X)
    cy_exp    : float  — source centre in exposure-map pixels (Y)
    """
    src_coord = parse_coord(cfg.ra, cfg.dec)
    cx_exp, cy_exp, pscale_exp = sky_to_img_pixel(
        src_coord.ra.deg, src_coord.dec.deg, exp_hdr)

    r_src_pix_exp = _arcsec_to_pix(cfg.src_radius_arcsec, pscale_exp)

    # PSF FWHM in exposure-map pixels — use instrument default if cfg is generic
    psf_fwhm_pix = _arcsec_to_pix(
        cfg.PSF_FWHM_DEFAULT.get(instrument, cfg.psf_fwhm_arcsec),
        pscale_exp)

    # Bounds check — warn rather than crash if centre is outside the map
    ny, nx = exp_data.shape
    if not (0 <= cx_exp < nx and 0 <= cy_exp < ny):
        warnings.warn(
            f"{instrument}: source position ({cx_exp:.1f}, {cy_exp:.1f}) "
            f"is outside the exposure map ({nx}×{ny} pix). "
            "Check RA/Dec and that the exposure map covers the source.",
            RuntimeWarning, stacklevel=2)

    exp_stats, exp_meta = compute_exposure_stats(
        exp_data, cx_exp, cy_exp, r_src_pix_exp, psf_fwhm_pix)

    print(f"  {instrument}: exposure  "
          f"median={exp_stats['median']:.0f} s  "
          f"mean={exp_stats['mean']:.0f} s  "
          f"({exp_meta['n_pix_nonzero']}/{exp_meta['n_pix_total']} pix nonzero)")

    return exp_stats, exp_meta, cx_exp, cy_exp
