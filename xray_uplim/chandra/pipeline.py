"""
xray_uplim.chandra.pipeline
----------------------------
Full upper-limit pipeline for Chandra ACIS using CIAO aprates.

Follows the user's standard manual workflow:
    1.  chandra_repro  (optional; auto-run if repro/ not found)
    2.  fluximage      (creates counts image + exposure map)
    3.  Interactive region selector (matplotlib GUI, optional)
    4.  dmlist         (count photons in source and background apertures)
    5.  dmkeypar       (read LIVETIME)
    6.  dmstat         (mean exposure map in source and background)
    7.  Compute geometric areas (arcsec²)
    8.  aprates        (Bayesian upper limit; once per confidence level)
    9.  Marginalized/Gehrels  (cross-check; from our own statistics.py)
    10. EEF correction (Gaussian PSF model)
    11. Print results table
    12. Save diagnostic plots
    13. Write CSV + XLSX

Public API
----------
run_uplim(**kwargs)          — entry point; builds ChandraConfig and runs
process_observation(cfg)     — full pipeline for one or more Chandra observations
"""

import copy
import csv
import math
import os
import tempfile
import warnings

import numpy as np
from astropy.io import fits

from .config     import ChandraConfig
from .eef        import compute_chandra_eef
from .io         import (
    check_ciao, find_ciao_prefix,
    find_repro_dir, find_evt2, find_fluximage_dir,
    find_expmap, find_counts_img, run_chandra_repro, run_fluximage,
    load_evt2_xy, dmkeypar, dmlist_counts, dmstat_mean,
    expmap_aperture_mean, run_aprates,
)
from ..coords    import parse_coord, sky_to_evt_pixel
from ..statistics import net_count_rate, marginalized_upper_limit, gehrels_upper_limit


# =============================================================================
# RESULTS TABLE
# =============================================================================

def _compute_ul_results(N_src, B_scaled, t_eff, N_bkg_raw, area_ratio,
                         confidence_levels,
                         eef=None,
                         aprates_results=None):
    """
    Compute upper limits at every confidence level.  No output printed.

    Returns a list of dicts, one per CL, with both aprates (primary, when
    available) and Kraft/Gehrels (cross-check) results.

    aprates_results : list of (src_rate, err_lo, err_up) tuples, one per CL,
                      or None when CIAO aprates was skipped.
    """
    CR_net, CR_sigma = net_count_rate(N_src, B_scaled, t_eff,
                                       N_bkg_raw, area_ratio)
    results = []
    for i, cl in enumerate(confidence_levels):
        CR_m_ap = marginalized_upper_limit(N_src, N_bkg_raw, area_ratio, t_eff, cl)
        S_g     = gehrels_upper_limit(N_src, B_scaled, cl)
        CR_m_tot = CR_m_ap / eef if (eef is not None and eef > 0) else None
        CR_g_ap  = S_g / t_eff
        CR_g_tot = S_g / (t_eff * eef) if (eef is not None and eef > 0) else None

        # aprates primary result
        if aprates_results is not None and i < len(aprates_results):
            ap_rate, ap_lo, ap_up = aprates_results[i]
            ap_up_tot = ap_up / eef if (eef is not None and eef > 0) else None
        else:
            ap_rate = ap_lo = ap_up = ap_up_tot = None

        results.append({
            'cl'                   : cl,
            'CR_net'               : CR_net,
            'CR_sigma'             : CR_sigma,
            # aprates (primary)
            'aprates_rate'         : ap_rate,
            'aprates_err_lo'       : ap_lo,
            'aprates_ul_aperture'  : ap_up,
            'aprates_ul_total'     : ap_up_tot,
            # Marginalized (cross-check)
            'CR_marg_aperture'     : CR_m_ap,
            'CR_marg_total'        : CR_m_tot,
            # Gehrels (additional cross-check)
            'S_gehrels'            : S_g,
            'CR_gehrels_aperture'  : CR_g_ap,
            'CR_gehrels_total'     : CR_g_tot,
        })
    return results


def _print_results_table(N_src, B_scaled, t_eff, N_bkg_raw, area_ratio,
                          confidence_levels, eef=None, aprates_results=None,
                          psf_fwhm_arcsec=None):
    """
    Compute upper limits and print a concise summary (one line per CL).

    When CIAO aprates ran successfully, prints only the aprates result
    (total source rate if EEF is available, aperture rate otherwise).
    When aprates was not run, falls back to the Bayesian marginalized limit.
    Full results (all methods, all CLs) are always written to the CSV.
    """
    results = _compute_ul_results(
        N_src, B_scaled, t_eff, N_bkg_raw, area_ratio,
        confidence_levels, eef=eef, aprates_results=aprates_results)

    has_aprates = aprates_results is not None
    has_eef     = eef is not None

    if has_aprates and has_eef:
        method_label = "CIAO aprates total source count rate"
    elif has_aprates:
        method_label = "CIAO aprates aperture count rate"
    elif has_eef:
        method_label = "Bayesian marginalized total source count rate"
    else:
        method_label = "Bayesian marginalized aperture count rate"

    print(f"\n  Upper limits  ({method_label}, cts/s):")
    for r in results:
        if has_aprates and has_eef:
            val = (r['aprates_ul_total'] if r['aprates_ul_total'] is not None
                   else r['aprates_ul_aperture'])
        elif has_aprates:
            val = r['aprates_ul_aperture']
        elif has_eef:
            val = r['CR_marg_total']
        else:
            val = r['CR_marg_aperture']
        print(f"    {r['cl']*100:.1f}%:  < {val:.4e}")

    if has_aprates and has_eef:
        print(f"  (EEF = {eef:.4f};  all methods and CLs in CSV)")
    elif has_aprates:
        print(f"  (EEF unavailable — aperture rate only;  all methods and CLs in CSV)")
    elif has_eef:
        print(f"  (EEF = {eef:.4f};  CIAO aprates not run — all CLs in CSV)")
    else:
        print(f"  (CIAO aprates not run;  EEF unavailable — all CLs in CSV)")

    return results


# =============================================================================
# CSV OUTPUT
# =============================================================================

def _build_csv_rows(e_lo, e_hi, N_src, N_bkg_raw, B_scaled, area_ratio,
                    t_eff, E_s, E_b, A_s_arcsec2, A_b_arcsec2,
                    ul_results, eef_info, obsid,
                    result_type='individual', date_obs=''):
    rows = []
    for r in ul_results:
        row = {
            'result_type'          : result_type,
            'obsid'                : obsid,
            'date_obs'             : date_obs,
            'energy_lo_kev'        : e_lo,
            'energy_hi_kev'        : e_hi,
            'N_src'                : N_src,
            'N_bkg_raw'            : N_bkg_raw,
            'B_scaled'             : f"{B_scaled:.4f}",
            'area_ratio'           : f"{area_ratio:.6f}",
            'A_src_arcsec2'        : f"{A_s_arcsec2:.3f}",
            'A_bkg_arcsec2'        : f"{A_b_arcsec2:.3f}",
            't_eff_s'              : f"{t_eff:.3f}",
            'E_s_cm2s'             : f"{E_s:.4e}",
            'E_b_cm2s'             : f"{E_b:.4e}",
            'confidence_level'     : r['cl'],
            'CR_net'               : f"{r['CR_net']:.6e}",
            'CR_sigma'             : f"{r['CR_sigma']:.6e}",
            # aprates
            'aprates_ul_aperture'  : (f"{r['aprates_ul_aperture']:.6e}"
                                      if r['aprates_ul_aperture'] is not None else ''),
            'aprates_ul_total'     : (f"{r['aprates_ul_total']:.6e}"
                                      if r['aprates_ul_total'] is not None else ''),
            # Marginalized cross-check
            'CR_marg_aperture'     : f"{r['CR_marg_aperture']:.6e}",
            'CR_marg_total'        : (f"{r['CR_marg_total']:.6e}"
                                      if r['CR_marg_total'] is not None else ''),
            'S_gehrels'            : f"{r['S_gehrels']:.4f}",
            'CR_gehrels_aperture'  : f"{r['CR_gehrels_aperture']:.6e}",
            'CR_gehrels_total'     : (f"{r['CR_gehrels_total']:.6e}"
                                      if r['CR_gehrels_total'] is not None else ''),
            # EEF
            'eef'                  : '',
            'psf_fwhm_arcsec'      : '',
        }
        if eef_info is not None:
            row['eef']             = f"{eef_info['eef']:.6f}"
            row['psf_fwhm_arcsec'] = f"{eef_info['psf_fwhm_arcsec']:.3f}"
        rows.append(row)
    return rows


def write_results_csv(rows, out_dir, obsid_label):
    """Write upper-limit results to CSV (and XLSX if openpyxl is available)."""
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, f"chandra_uplim_{obsid_label}.csv")

    fieldnames = [
        'result_type', 'obsid', 'date_obs',
        'energy_lo_kev', 'energy_hi_kev',
        'N_src', 'N_bkg_raw', 'B_scaled', 'area_ratio',
        'A_src_arcsec2', 'A_bkg_arcsec2',
        't_eff_s', 'E_s_cm2s', 'E_b_cm2s',
        'eef', 'psf_fwhm_arcsec',
        'confidence_level',
        'CR_net', 'CR_sigma',
        'aprates_ul_aperture', 'aprates_ul_total',
        'CR_marg_aperture', 'CR_marg_total',
        'S_gehrels', 'CR_gehrels_aperture', 'CR_gehrels_total',
    ]

    with open(csv_path, 'w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames,
                                extrasaction='ignore',
                                quoting=csv.QUOTE_NONNUMERIC)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n  Results written to: {csv_path}")

    try:
        from ..output import write_results_xlsx
        xlsx_path = csv_path.replace('.csv', '.xlsx')
        if write_results_xlsx(rows, fieldnames, xlsx_path,
                              text_cols=('result_type', 'obsid')):
            print(f"  Excel file written: {xlsx_path}")
    except Exception:
        pass

    return csv_path


# =============================================================================
# COORDINATE HELPERS
# =============================================================================

def _evt_pixel_to_sky(cx, cy, evt_hdr):
    """Invert sky_to_evt_pixel: event-file sky pixel (cx,cy) → (ra_deg, dec_deg)."""
    x_col = y_col = None
    for i in range(1, 300):
        if f'TTYPE{i}' not in evt_hdr:
            break
        name = evt_hdr[f'TTYPE{i}'].strip().upper()
        if name == 'X':
            x_col = i
        elif name == 'Y':
            y_col = i

    crpx_x = float(evt_hdr[f'TCRPX{x_col}'])
    crvl_x = float(evt_hdr[f'TCRVL{x_col}'])
    cdlt_x = float(evt_hdr[f'TCDLT{x_col}'])
    crpx_y = float(evt_hdr[f'TCRPX{y_col}'])
    crvl_y = float(evt_hdr[f'TCRVL{y_col}'])
    cdlt_y = float(evt_hdr[f'TCDLT{y_col}'])

    cos_dec = np.cos(np.radians(crvl_y))
    ra_deg  = crvl_x + (cx - crpx_x) * cdlt_x / cos_dec
    dec_deg = crvl_y + (cy - crpx_y) * cdlt_y
    return ra_deg, dec_deg


def _geometric_areas(src_r_arcsec, bkg_r_arcsec, bkg_inner_factor, bkg_mode,
                     bkg_r_arcsec_manual=None):
    """
    Return (A_s_arcsec2, A_b_arcsec2, area_ratio) for circle + annulus or
    circle + circle (manual bkg mode).
    """
    A_s = math.pi * src_r_arcsec ** 2
    if bkg_mode == 'annulus':
        r_in = src_r_arcsec * bkg_inner_factor
        A_b  = math.pi * (bkg_r_arcsec ** 2 - r_in ** 2)
    else:
        r_bkg = bkg_r_arcsec_manual if bkg_r_arcsec_manual else bkg_r_arcsec
        A_b   = math.pi * r_bkg ** 2
    area_ratio = A_s / A_b
    return A_s, A_b, area_ratio


# =============================================================================
# INTERNAL HELPERS
# =============================================================================

def _prepare_obs(cfg, obsid_str, env: dict):
    """
    Ensure repro/ and fluximage/ exist for one observation.
    Returns (evt2_path, expmap_path, counts_img_path, repro_dir).
    """
    e_lo_kev, e_hi_kev, e_lo_ev, e_hi_ev, ref_kev = cfg.resolve_energy_band()

    # ---- Reprocessing -------------------------------------------------------
    repro_dir = find_repro_dir(cfg.base_path, obsid_str)
    if repro_dir is None:
        if cfg.run_repro:
            run_chandra_repro(cfg.base_path, obsid_str, env)
            repro_dir = find_repro_dir(cfg.base_path, obsid_str)
        if repro_dir is None:
            raise FileNotFoundError(
                f"No repro/ directory found for obsid {obsid_str} "
                f"under {cfg.base_path}.\n"
                "Run chandra_repro manually or set run_repro=True.")

    # ---- Event file ---------------------------------------------------------
    evt2 = find_evt2(repro_dir)
    if evt2 is None:
        raise FileNotFoundError(
            f"No *repro_evt2.fits found in {repro_dir}. "
            "Run chandra_repro first.")

    # ---- Exposure map -------------------------------------------------------
    fdir   = find_fluximage_dir(repro_dir)
    expmap = find_expmap(fdir, e_lo_kev, e_hi_kev)
    if expmap is None:
        run_fluximage(evt2, fdir, e_lo_kev, e_hi_kev, ref_kev, env)
        expmap = find_expmap(fdir, e_lo_kev, e_hi_kev)
    if expmap is None:
        raise FileNotFoundError(
            f"fluximage ran but no {e_lo_kev}-{e_hi_kev}_thresh.expmap "
            f"found in {fdir}.")

    counts_img = find_counts_img(fdir, e_lo_kev, e_hi_kev)

    return evt2, expmap, counts_img, repro_dir


def _run_gui(cfg, evt2_path, obsid_str):
    """
    Open the interactive region selector for one observation.

    Mutates cfg with the user's chosen aperture + background settings.
    Returns (src_ra_deg, src_dec_deg, bkg_ra_deg, bkg_dec_deg).
    """
    from ..region_selector import select_regions_interactive

    e_lo_kev, e_hi_kev, e_lo_ev, e_hi_ev, ref_kev = cfg.resolve_energy_band()
    src_coord = parse_coord(cfg.ra, cfg.dec)

    evt_x, evt_y, evt_hdr = load_evt2_xy(evt2_path, e_lo_ev, e_hi_ev)

    cx_evt, cy_evt, pscale_evt = sky_to_evt_pixel(
        src_coord.ra.deg, src_coord.dec.deg, evt_hdr)

    print(f"\n  Opening interactive region selector ({obsid_str})…")
    sel = select_regions_interactive(
        evt_x, evt_y, cx_evt, cy_evt, pscale_evt,
        cfg, f"ACIS {obsid_str}")

    cfg.src_radius_arcsec = sel['src_radius_arcsec']
    cfg.bkg_radius_arcsec = sel['bkg_radius_arcsec']
    cfg.bkg_inner_factor  = sel['bkg_inner_factor']

    bkg_cx = sel['bkg_cx']
    bkg_cy = sel['bkg_cy']
    bkg_moved = (abs(bkg_cx - cx_evt) > 1.0 or abs(bkg_cy - cy_evt) > 1.0)

    src_ra_deg  = src_coord.ra.deg
    src_dec_deg = src_coord.dec.deg
    bkg_ra_deg  = src_ra_deg
    bkg_dec_deg = src_dec_deg

    if bkg_moved:
        try:
            bkg_ra_deg, bkg_dec_deg = _evt_pixel_to_sky(bkg_cx, bkg_cy, evt_hdr)
            cfg.bkg_mode = 'manual'
            cfg.bkg_ra   = str(float(bkg_ra_deg))
            cfg.bkg_dec  = str(float(bkg_dec_deg))
            print(f"  [GUI] Background → manual mode: "
                  f"RA={bkg_ra_deg:.5f}  Dec={bkg_dec_deg:.5f}")
        except Exception as exc:
            warnings.warn(
                f"Could not convert background pixel to RA/Dec ({exc}). "
                "Falling back to annulus mode.",
                RuntimeWarning, stacklevel=2)
            cfg.bkg_mode = 'annulus'

    return src_ra_deg, src_dec_deg, bkg_ra_deg, bkg_dec_deg


def _extract_one_obs(cfg, obsid_str,
                     src_ra_deg, src_dec_deg,
                     env: dict,
                     bkg_ra_deg=None, bkg_dec_deg=None):
    """
    Steps 4–7 for one observation: count photons, read exposure map,
    get LIVETIME.

    If bkg_ra_deg/bkg_dec_deg are None, falls back to cfg.bkg_ra/bkg_dec
    (manual mode) or source centre (annulus mode).

    Returns a dict of raw per-obs quantities.
    """
    e_lo_kev, e_hi_kev, e_lo_ev, e_hi_ev, _ = cfg.resolve_energy_band()
    evt2, expmap, counts_img, repro_dir = _prepare_obs(cfg, obsid_str, env)

    # Resolve background centre
    if cfg.bkg_mode == 'annulus':
        bkg_ra  = src_ra_deg
        bkg_dec = src_dec_deg
    else:
        if bkg_ra_deg is not None and bkg_dec_deg is not None:
            bkg_ra  = bkg_ra_deg
            bkg_dec = bkg_dec_deg
        else:
            bkg_coord = parse_coord(cfg.bkg_ra, cfg.bkg_dec)
            bkg_ra    = bkg_coord.ra.deg
            bkg_dec   = bkg_coord.dec.deg

    # Aperture geometry
    A_s, A_b, area_ratio = _geometric_areas(
        cfg.src_radius_arcsec, cfg.bkg_radius_arcsec,
        cfg.bkg_inner_factor, cfg.bkg_mode)

    print(f"\n  Source aperture  : r = {cfg.src_radius_arcsec:.1f}\"  "
          f"(A_s = {A_s:.1f} arcsec²)")
    if cfg.bkg_mode == 'annulus':
        r_in = cfg.src_radius_arcsec * cfg.bkg_inner_factor
        print(f"  Bkg annulus      : {r_in:.1f}\" – {cfg.bkg_radius_arcsec:.1f}\"  "
              f"(A_b = {A_b:.1f} arcsec²)")
    else:
        print(f"  Bkg circle       : r = {cfg.bkg_radius_arcsec:.1f}\"  "
              f"RA={bkg_ra:.5f}  Dec={bkg_dec:.5f}  (A_b = {A_b:.1f} arcsec²)")
    print(f"  Area ratio (A_s/A_b) : {area_ratio:.5f}")

    # ---- Step 4: Count photons ----------------------------------------------
    N_src = dmlist_counts(
        evt2, src_ra_deg, src_dec_deg,
        cfg.src_radius_arcsec, e_lo_ev, e_hi_ev, env)

    if cfg.bkg_mode == 'annulus':
        r_in = cfg.src_radius_arcsec * cfg.bkg_inner_factor
        N_bkg = dmlist_counts(
            evt2, src_ra_deg, src_dec_deg,
            cfg.bkg_radius_arcsec, e_lo_ev, e_hi_ev, env,
            inner_arcsec=r_in)
    else:
        N_bkg = dmlist_counts(
            evt2, bkg_ra, bkg_dec,
            cfg.bkg_radius_arcsec, e_lo_ev, e_hi_ev, env)

    B_scaled = N_bkg * area_ratio
    print(f"\n  N_src            : {N_src}")
    print(f"  N_bkg_raw        : {N_bkg}")
    print(f"  B_scaled (exp.)  : {B_scaled:.3f}")
    print(f"  Net (N_src − B)  : {N_src - B_scaled:.3f}")

    # ---- Step 5: LIVETIME ---------------------------------------------------
    livetime_str = dmkeypar(evt2, 'LIVETIME', env)
    T_s = float(livetime_str)
    print(f"\n  LIVETIME         : {T_s:.3f} s  ({T_s/1e3:.3f} ks)")

    # ---- Step 6: Exposure map means -----------------------------------------
    # Use pure-Python WCS aperture photometry — the CIAO dmstat (ra,dec)=circle()
    # filter works on event files but not on image FITS files like the expmap.
    E_s = expmap_aperture_mean(
        expmap, src_ra_deg, src_dec_deg, cfg.src_radius_arcsec)

    if cfg.bkg_mode == 'annulus':
        r_in = cfg.src_radius_arcsec * cfg.bkg_inner_factor
        E_b = expmap_aperture_mean(
            expmap, src_ra_deg, src_dec_deg,
            cfg.bkg_radius_arcsec, inner_arcsec=r_in)
    else:
        E_b = expmap_aperture_mean(
            expmap, bkg_ra, bkg_dec, cfg.bkg_radius_arcsec)

    print(f"\n  E_s (mean expmap src) : {E_s:.4e} cm²·s")
    print(f"  E_b (mean expmap bkg) : {E_b:.4e} cm²·s")

    # ---- DATE-OBS from event header -----------------------------------------
    try:
        date_obs = dmkeypar(evt2, 'DATE-OBS', env)
    except Exception:
        date_obs = ''

    return dict(
        obsid_str=obsid_str,
        evt2=evt2, expmap=expmap, counts_img=counts_img,
        src_ra_deg=src_ra_deg, src_dec_deg=src_dec_deg,
        bkg_ra_deg=bkg_ra, bkg_dec_deg=bkg_dec,
        N_src=N_src, N_bkg_raw=N_bkg, B_scaled=B_scaled,
        area_ratio=area_ratio, A_s=A_s, A_b=A_b,
        T_s=T_s, E_s=E_s, E_b=E_b,
        date_obs=date_obs,
    )


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def process_observation(cfg: ChandraConfig):
    """
    Full extraction and upper-limit calculation for one or more Chandra
    ACIS observations.

    For multiple obsids, counts are summed and exposures accumulated
    before computing a single combined upper limit with aprates.  Per-obs
    individual rows are also written to the CSV.

    Returns
    -------
    dict with keys:
        obsids, N_src, N_bkg_raw, B_scaled, area_ratio, t_eff_s,
        E_s, E_b, A_s, A_b, ul, energy, eef_info, csv_rows
    """
    e_lo_kev, e_hi_kev, e_lo_ev, e_hi_ev, ref_kev = cfg.resolve_energy_band()
    obsids      = cfg.obsids
    n_obs       = len(obsids)
    obsid_label = obsids[0] if n_obs == 1 else "+".join(obsids)
    out_dir     = os.path.join(cfg.base_path, obsid_label, 'ul_products')
    os.makedirs(out_dir, exist_ok=True)

    # Save original aperture/background settings for per-obs GUI restore
    _orig = {k: getattr(cfg, k) for k in (
        'src_radius_arcsec', 'bkg_radius_arcsec', 'bkg_inner_factor',
        'bkg_mode', 'bkg_ra', 'bkg_dec')}

    # Locate CIAO once — env dict is threaded to all CIAO calls below
    env = check_ciao(cfg.ciao_prefix)

    # Resolve source RA/Dec once (used as default; may be overridden by GUI)
    src_coord   = parse_coord(cfg.ra, cfg.dec)
    src_ra_deg  = src_coord.ra.deg
    src_dec_deg = src_coord.dec.deg
    bkg_ra_deg  = src_ra_deg   # default (annulus mode)
    bkg_dec_deg = src_dec_deg

    # ------------------------------------------------------------------ #
    # Loop over observations                                               #
    # ------------------------------------------------------------------ #
    per_obs = []

    for i, obsid_str in enumerate(obsids):
        print(f"\n{'='*70}")
        if n_obs > 1:
            print(f"  Chandra ACIS  —  Observation {i+1}/{n_obs}  ({obsid_str})")
        else:
            print(f"  Chandra ACIS  ({obsid_str})")
        print(f"{'='*70}")

        # ── GUI logic ────────────────────────────────────────────────── #
        if cfg.use_gui and cfg.gui_per_obs:
            cfg_obs = copy.copy(cfg)
            for k, v in _orig.items():
                setattr(cfg_obs, k, v)
            evt2_path = find_evt2(
                find_repro_dir(cfg.base_path, obsid_str))
            obs_src_ra, obs_src_dec, obs_bkg_ra, obs_bkg_dec = \
                _run_gui(cfg_obs, evt2_path, obsid_str)

        elif cfg.use_gui and i == 0:
            evt2_path = find_evt2(
                find_repro_dir(cfg.base_path, obsid_str))
            obs_src_ra, obs_src_dec, obs_bkg_ra, obs_bkg_dec = \
                _run_gui(cfg, evt2_path, obsid_str)
            # Store sky coords for subsequent obs in shared mode
            bkg_ra_deg  = obs_bkg_ra
            bkg_dec_deg = obs_bkg_dec
            cfg_obs = cfg

        elif cfg.use_gui and i > 0:
            # Shared mode: GUI already ran; reuse cfg (apertures + bkg sky coords)
            cfg_obs     = cfg
            obs_src_ra  = src_ra_deg
            obs_src_dec = src_dec_deg
            obs_bkg_ra  = bkg_ra_deg
            obs_bkg_dec = bkg_dec_deg

        else:
            cfg_obs     = cfg
            obs_src_ra  = src_ra_deg
            obs_src_dec = src_dec_deg
            if cfg.bkg_mode == 'manual':
                bkg_coord   = parse_coord(cfg.bkg_ra, cfg.bkg_dec)
                obs_bkg_ra  = bkg_coord.ra.deg
                obs_bkg_dec = bkg_coord.dec.deg
            else:
                obs_bkg_ra  = src_ra_deg
                obs_bkg_dec = src_dec_deg

        raw = _extract_one_obs(
            cfg_obs, obsid_str,
            obs_src_ra, obs_src_dec,
            env,
            bkg_ra_deg=obs_bkg_ra, bkg_dec_deg=obs_bkg_dec)
        raw['cfg_obs'] = cfg_obs
        per_obs.append(raw)

    # ------------------------------------------------------------------ #
    # Accumulate across observations                                        #
    # ------------------------------------------------------------------ #
    N_src_total     = sum(r['N_src']     for r in per_obs)
    N_bkg_raw_total = sum(r['N_bkg_raw'] for r in per_obs)
    T_total         = sum(r['T_s']       for r in per_obs)
    E_s_total       = sum(r['E_s']       for r in per_obs)
    E_b_total       = sum(r['E_b']       for r in per_obs)
    area_ratio      = (sum(r['B_scaled'] for r in per_obs) / N_bkg_raw_total
                       if N_bkg_raw_total > 0
                       else per_obs[0]['area_ratio'])
    A_s             = per_obs[0]['A_s']
    A_b             = per_obs[0]['A_b']
    B_scaled_total  = N_bkg_raw_total * area_ratio

    # ---- EEF ----------------------------------------------------------------
    eef_info = compute_chandra_eef(cfg.src_radius_arcsec, cfg.psf_fwhm_arcsec)
    eef       = eef_info['eef']
    print(f"\n  -- EEF (Encircled Energy Fraction) ----------------------------")
    print(f"    PSF FWHM (Gaussian)  : {cfg.psf_fwhm_arcsec:.3f} arcsec")
    print(f"    Aperture radius      : {cfg.src_radius_arcsec:.3f} arcsec")
    print(f"    EEF                  : {eef:.6f}")

    # ---- Step 8: aprates (once per confidence level) -------------------------
    aprates_results = None
    if cfg.use_aprates:
        try:
            aprates_results = []
            with tempfile.TemporaryDirectory() as tmpdir:
                for cl in cfg.confidence_levels:
                    outfile = os.path.join(
                        tmpdir, f"aprates_{cl:.4f}.par")
                    ap_rate, ap_lo, ap_up = run_aprates(
                        n=N_src_total, m=N_bkg_raw_total,
                        A_s=A_s, A_b=A_b,
                        T_s=T_total, T_b=T_total,
                        E_s=E_s_total, E_b=E_b_total,
                        conf=cl, outfile=outfile, env=env)
                    aprates_results.append((ap_rate, ap_lo, ap_up))
                    print(f"\n  aprates (CL={cl:.4f}):")
                    print(f"    src_rate         : {ap_rate:.4e} cts/s")
                    print(f"    src_rate_err_up  : {ap_up:.4e} cts/s  "
                          f"← aperture UL (use in PIMMS)")
                    if eef > 0:
                        print(f"    UL_tot (EEF corr): {ap_up/eef:.4e} cts/s")
        except RuntimeError as exc:
            warnings.warn(
                f"aprates failed — falling back to Kraft only.\n  {exc}",
                RuntimeWarning, stacklevel=2)
            aprates_results = None

    # ---- Step 9: results table ----------------------------------------------
    if n_obs > 1:
        print(f"\n{'='*70}")
        print(f"  Combined totals across {n_obs} observations")
        print(f"{'='*70}")
        print(f"  N_src total      : {N_src_total}")
        print(f"  N_bkg_raw total  : {N_bkg_raw_total}")
        print(f"  B_scaled total   : {B_scaled_total:.3f} cts")
        print(f"  T_eff total      : {T_total:.3f} s  ({T_total/1e3:.3f} ks)")
        print(f"  E_s total        : {E_s_total:.4e} cm²·s")
        print(f"  E_b total        : {E_b_total:.4e} cm²·s")

    ul_results = _print_results_table(
        N_src_total, B_scaled_total, T_total,
        N_bkg_raw_total, area_ratio,
        cfg.confidence_levels, eef=eef,
        aprates_results=aprates_results,
        psf_fwhm_arcsec=cfg.psf_fwhm_arcsec)

    # ---- Step 10: per-obs individual UL rows --------------------------------
    if n_obs > 1:
        print(f"\n{'='*70}")
        print(f"  Per-observation summary  ({n_obs} observations)")
        print(f"{'='*70}")
        print(f"  {'Obs ID':>14}  {'N_src':>6}  {'B_scaled':>9}  "
              f"{'T_eff (ks)':>11}  {'E_s (cm²s)':>12}")
        print("  " + "-" * 62)

    for r in per_obs:
        r_eef      = eef_info
        r_area     = r['area_ratio']
        r_B        = r['N_bkg_raw'] * r_area

        r_ap_list = None
        if aprates_results is not None:
            with tempfile.TemporaryDirectory() as tmpdir2:
                r_ap_list = []
                for cl in cfg.confidence_levels:
                    outfile = os.path.join(tmpdir2, f"ap_{cl:.4f}.par")
                    ap = run_aprates(
                        n=r['N_src'], m=r['N_bkg_raw'],
                        A_s=r['A_s'], A_b=r['A_b'],
                        T_s=r['T_s'], T_b=r['T_s'],
                        E_s=r['E_s'], E_b=r['E_b'],
                        conf=cl, outfile=outfile, env=env)
                    r_ap_list.append(ap)

        r['ul'] = _compute_ul_results(
            r['N_src'], r_B, r['T_s'],
            r['N_bkg_raw'], r_area, cfg.confidence_levels,
            eef=eef, aprates_results=r_ap_list)
        r['csv_rows'] = _build_csv_rows(
            e_lo_kev, e_hi_kev,
            r['N_src'], r['N_bkg_raw'], r_B, r_area,
            r['T_s'], r['E_s'], r['E_b'],
            r['A_s'], r['A_b'],
            r['ul'], r_eef, r['obsid_str'],
            result_type='individual', date_obs=r.get('date_obs', ''))

        if n_obs > 1:
            print(f"  {r['obsid_str']:>14}  {r['N_src']:>6}  "
                  f"{r_B:>9.2f}  {r['T_s']/1e3:>11.3f}  "
                  f"{r['E_s']:>12.4e}")

    # ---- Step 11: diagnostic plots ------------------------------------------
    if cfg.save_plots:
        for r in per_obs:
            _save_plots(r, cfg, out_dir, e_lo_kev, e_hi_kev)

    # ---- Step 12: CSV -------------------------------------------------------
    obsid_label = obsids[0] if n_obs == 1 else '+'.join(obsids)
    combined_date = per_obs[0].get('date_obs', '') if per_obs else ''

    combined_csv_rows = _build_csv_rows(
        e_lo_kev, e_hi_kev,
        N_src_total, N_bkg_raw_total, B_scaled_total, area_ratio,
        T_total, E_s_total, E_b_total, A_s, A_b,
        ul_results, eef_info, obsid_label,
        result_type='combined', date_obs=combined_date)

    all_csv_rows = []
    if n_obs > 1:
        for r in per_obs:
            all_csv_rows.extend(r['csv_rows'])
    all_csv_rows.extend(combined_csv_rows)

    write_results_csv(all_csv_rows, out_dir, obsid_label)

    return {
        'obsids'    : obsids,
        'N_src'     : N_src_total,
        'N_bkg_raw' : N_bkg_raw_total,
        'B_scaled'  : B_scaled_total,
        'area_ratio': area_ratio,
        't_eff_s'   : T_total,
        'E_s'       : E_s_total,
        'E_b'       : E_b_total,
        'A_s'       : A_s,
        'A_b'       : A_b,
        'ul'        : ul_results,
        'energy'    : (e_lo_kev, e_hi_kev),
        'eef_info'  : eef_info,
        'csv_rows'  : all_csv_rows,
        'per_obs'   : per_obs,
    }


# =============================================================================
# DIAGNOSTIC PLOTS
# =============================================================================

def _save_plots(obs_raw, cfg, out_dir, e_lo, e_hi):
    """Save diagnostic region image for one observation."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        from astropy.visualization import ZScaleInterval
    except ImportError:
        return

    counts_img = obs_raw.get('counts_img')
    if counts_img is None or not os.path.isfile(counts_img):
        return

    try:
        with fits.open(counts_img) as hdul:
            img_data = hdul[0].data.astype(float)
            img_hdr  = hdul[0].header

        from ..coords import sky_to_img_pixel
        src_cx, src_cy, pscale = sky_to_img_pixel(
            obs_raw['src_ra_deg'], obs_raw['src_dec_deg'], img_hdr)
        bkg_cx, bkg_cy, _      = sky_to_img_pixel(
            obs_raw['bkg_ra_deg'], obs_raw['bkg_dec_deg'], img_hdr)

        src_r_pix = cfg.src_radius_arcsec / pscale
        bkg_r_pix = cfg.bkg_radius_arcsec / pscale
        bkg_in_pix = cfg.src_radius_arcsec * cfg.bkg_inner_factor / pscale

        zscale = ZScaleInterval()
        vmin, vmax = zscale.get_limits(img_data)
        vmax = max(vmax, vmin + 1)

        fig, ax = plt.subplots(figsize=(5.5, 5.0))
        ax.imshow(img_data, origin='lower', cmap='gray',
                  vmin=vmin, vmax=vmax, interpolation='none')

        ax.add_patch(mpatches.Circle(
            (src_cx, src_cy), src_r_pix,
            color='tomato', fill=False, lw=2.0, label='Source'))
        if cfg.bkg_mode == 'annulus':
            ax.add_patch(mpatches.Circle(
                (src_cx, src_cy), bkg_in_pix,
                color='orange', fill=False, lw=1.5, linestyle='--'))
            ax.add_patch(mpatches.Circle(
                (src_cx, src_cy), bkg_r_pix,
                color='orange', fill=False, lw=2.0, label='Bkg'))
        else:
            ax.add_patch(mpatches.Circle(
                (bkg_cx, bkg_cy), bkg_r_pix,
                color='orange', fill=False, lw=2.0, label='Bkg'))

        ax.set_title(
            f"Chandra ACIS  {obs_raw['obsid_str']}  "
            f"{e_lo:.1f}–{e_hi:.1f} keV\n"
            f"N_src={obs_raw['N_src']}  N_bkg={obs_raw['N_bkg_raw']}  "
            f"B={obs_raw['B_scaled']:.1f}", fontsize=9)
        ax.legend(fontsize=8, loc='upper right')
        ax.set_xlabel('Image X (pixel)')
        ax.set_ylabel('Image Y (pixel)')

        fname = os.path.join(
            out_dir,
            f"chandra_regions_{obs_raw['obsid_str']}_{e_lo:.1f}-{e_hi:.1f}keV.pdf")
        fig.savefig(fname, bbox_inches='tight')
        plt.close(fig)
        print(f"  Region image saved: {os.path.basename(fname)}")

    except Exception as exc:
        warnings.warn(f"Diagnostic plot skipped: {exc}", RuntimeWarning)


# =============================================================================
# ENTRY POINT
# =============================================================================

def run_uplim(base_path, obsid, ra, dec, **kwargs):
    """
    Run the full Chandra ACIS upper-limit pipeline.

    Parameters
    ----------
    base_path : str
        Root directory containing one sub-directory per obsid.
        Each obsid directory should contain the original Chandra
        data as downloaded from the CDA (primary/ and secondary/ sub-dirs
        expected by chandra_repro, or a pre-existing repro/ directory).
    obsid : str or list of str
        Chandra observation ID(s).  Pass a list to co-add multiple obs.
    ra  : str or float  — source RA  ("HH:MM:SS" or decimal degrees)
    dec : str or float  — source Dec ("±DD:MM:SS" or decimal degrees)
    **kwargs : any ChandraConfig field, e.g.
                   energy_band='soft',
                   src_radius_arcsec=5.0,
                   psf_fwhm_arcsec=0.9,
                   confidence_levels=[0.9973],
                   use_aprates=True,
                   save_plots=True

    Returns
    -------
    dict — result from process_observation()
    """
    cfg = ChandraConfig(base_path=base_path, obsid=obsid, ra=ra, dec=dec,
                        **kwargs)
    cfg.validate()

    e_lo_kev, e_hi_kev, *_ = cfg.resolve_energy_band()
    src_coord   = parse_coord(cfg.ra, cfg.dec)
    obsids      = cfg.obsids
    obsid_label = obsids[0] if len(obsids) == 1 else "+".join(obsids)
    out_dir     = os.path.join(cfg.base_path, obsid_label, 'ul_products')

    print("Chandra ACIS Non-Detection Upper Limit")
    print("=" * 70)
    print(f"Source       :  RA = {src_coord.ra.deg:.6f} deg  "
          f"Dec = {src_coord.dec.deg:.6f} deg")
    if isinstance(cfg.energy_band, tuple):
        band_label = f"{e_lo_kev:.2f}–{e_hi_kev:.2f} keV (custom)"
    else:
        band_label = f"'{cfg.energy_band}'  ({e_lo_kev:.2f}–{e_hi_kev:.2f} keV)"
    print(f"Energy band  :  {band_label}")
    print(f"Bkg mode     :  {cfg.bkg_mode}")
    print(f"Base path    :  {cfg.base_path}")
    if len(obsids) == 1:
        print(f"Obs ID       :  {obsids[0]}")
    else:
        print(f"Obs IDs      :  {len(obsids)} observations co-added")
        for oid in obsids:
            print(f"               {oid}")
    print(f"PSF FWHM     :  {cfg.psf_fwhm_arcsec:.2f} arcsec (Gaussian EEF model)")
    if cfg.use_aprates:
        print("Statistics   :  CIAO aprates  (marginalized as cross-check)")
    else:
        print("Statistics   :  marginalized/Gehrels  (no aprates)")
    print()

    result = process_observation(cfg)

    # ---- Final summary ------------------------------------------------------
    print(f"\n{'='*70}")
    print("  SUMMARY")
    print(f"{'='*70}")
    n_obs     = len(obsids)
    obs_label = obsids[0] if n_obs == 1 else f"{n_obs} obs co-added"
    eef_str   = f"{result['eef_info']['eef']:.3f}"
    ul_row    = next((u for u in result['ul'] if u['cl'] >= 0.997),
                     result['ul'][-1])

    print(f"  {'Obs':>20}  {'N_src':>6}  {'B_scaled':>9}  "
          f"{'T_eff (ks)':>11}  {'EEF':>6}")
    print("  " + "-" * 62)
    print(f"  {obs_label:>20}  {result['N_src']:>6}  "
          f"{result['B_scaled']:>9.2f}  "
          f"{result['t_eff_s']/1e3:>11.3f}  "
          f"{eef_str:>6}")

    if ul_row['aprates_ul_aperture'] is not None:
        print(f"\n  aprates UL (3σ aperture) : {ul_row['aprates_ul_aperture']:.4e} cts/s")
        if ul_row['aprates_ul_total'] is not None:
            print(f"  aprates UL (3σ EEF-corr) : {ul_row['aprates_ul_total']:.4e} cts/s")
        print(f"  → Use aperture UL in PIMMS (Chandra ACIS, same band)")
    else:
        print(f"\n  Marg UL (3σ aperture)    : {ul_row['CR_marg_aperture']:.4e} cts/s")

    print()
    return result
