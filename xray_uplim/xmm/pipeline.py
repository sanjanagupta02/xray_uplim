"""
xray_uplim.xmm.pipeline
------------------------
Per-instrument extraction and upper-limit calculation for XMM-Newton EPIC.

Each instrument (MOS1, MOS2, PN) is processed independently.  Results are
never combined across instruments — PN and MOS have different effective areas,
response matrices, and PSF shapes.

Multi-obsid co-adding sums counts and exposures across observations for each
instrument independently, yielding a single combined upper limit per instrument.

Public API
----------
run_uplim(**kwargs)               — entry point; builds XMMConfig and calls
                                    process_observations
process_observations(cfg)         — main pipeline; handles single and multi-obs
process_instrument(instrument, cfg)
                                  — thin wrapper for single-obsid backward compat
"""

import copy
import csv
import os
import warnings
import numpy as np

from .config   import XMMConfig
from .io       import locate_files, load_events, load_expmap
from .aperture import extract_src_bkg_counts, extract_exposure
from .eef      import compute_xmm_eef
from ..coords  import parse_coord, sky_to_img_pixel
from ..exposure import compute_exposure_area_ratio
from ..statistics import net_count_rate, marginalized_upper_limit, gehrels_upper_limit


# =============================================================================
# PURE COMPUTATION — no printing
# =============================================================================

def _compute_ul_results(N_src, B_scaled, t_eff, N_bkg_raw, area_ratio,
                         confidence_levels, eef=None):
    """
    Compute upper limits at every confidence level.

    Parameters
    ----------
    N_src, B_scaled, t_eff, N_bkg_raw, area_ratio : as usual
    confidence_levels : list of float
    eef : float or None

    Returns
    -------
    list of dicts with keys:
        cl, CR_net, CR_sigma, CR_marg_aperture, CR_marg_total,
        S_gehrels, CR_gehrels_aperture, CR_gehrels_total
    """
    CR_net, CR_sigma = net_count_rate(N_src, B_scaled, t_eff,
                                       N_bkg_raw, area_ratio)
    results = []
    for cl in confidence_levels:
        CR_m_ap = marginalized_upper_limit(N_src, N_bkg_raw, area_ratio, t_eff, cl)
        S_g     = gehrels_upper_limit(N_src, B_scaled, cl)
        CR_g_ap = S_g / t_eff
        if eef is not None and eef > 0:
            # Total source rate with EEF folded into the effective exposure.
            # Equivalent to CR_m_ap / EEF (linear change of variables), but
            # makes the physical model explicit (expected counts = S_tot × EEF × t).
            CR_m_tot = marginalized_upper_limit(
                N_src, N_bkg_raw, area_ratio, t_eff * eef, cl)
            CR_g_tot = S_g / (t_eff * eef)
        else:
            CR_m_tot = None
            CR_g_tot = None

        results.append({
            'cl':                     cl,
            'CR_net':                 CR_net,
            'CR_sigma':               CR_sigma,
            'CR_marg_aperture':       CR_m_ap,
            'CR_marg_total':          CR_m_tot,
            'S_gehrels':              S_g,
            'CR_gehrels_aperture':    CR_g_ap,
            'CR_gehrels_total':       CR_g_tot,
        })
    return results


# =============================================================================
# RESULTS TABLE
# =============================================================================

def _print_results_table(N_src, B_scaled, t_eff, N_bkg_raw, area_ratio,
                          confidence_levels, eef=None):
    """
    Compute upper limits and print a concise summary (one line per CL).

    Displays the Bayesian marginalized total source count rate when EEF is
    available, otherwise the aperture count rate.  Full results (all methods,
    all CLs) are always written to the CSV.

    Returns
    -------
    list of dicts — one per confidence level
    """
    results = _compute_ul_results(N_src, B_scaled, t_eff, N_bkg_raw, area_ratio,
                                   confidence_levels, eef=eef)

    use_total = eef is not None
    rate_label = "total source count rate" if use_total else "aperture count rate"
    print(f"\n  Upper limits  (Bayesian marginalized {rate_label}, cts/s):")
    for r in results:
        val = r['CR_marg_total'] if use_total else r['CR_marg_aperture']
        print(f"    {r['cl']*100:.1f}%:  < {val:.4e}")
    if use_total:
        print(f"  (EEF = {eef:.4f};  all methods and CLs in CSV)")
    else:
        print(f"  (EEF unavailable — aperture rate only;  all CLs in CSV)")

    return results


# =============================================================================
# CSV OUTPUT
# =============================================================================

def _build_csv_rows(instrument, e_lo, e_hi, N_src, N_bkg_raw, B_scaled,
                    area_ratio, t_eff, ul_results, eef_info, obsid,
                    date_obs='', result_type='individual'):
    """Build a list of CSV row dicts (one per confidence level) for one instrument."""
    rows = []
    for r in ul_results:
        row = {
            'result_type':        result_type,
            'obsid':              obsid,
            'date_obs':           date_obs,
            'instrument':         instrument,
            'energy_lo_kev':      e_lo,
            'energy_hi_kev':      e_hi,
            'N_src':              N_src,
            'N_bkg_raw':          N_bkg_raw,
            'B_scaled':           f"{B_scaled:.4f}",
            'area_ratio':         f"{area_ratio:.6f}",
            't_eff_s':            f"{t_eff:.2f}",
            'confidence_level':   r['cl'],
            'CR_net':              f"{r['CR_net']:.6e}",
            'CR_sigma':            f"{r['CR_sigma']:.6e}",
            'CR_marg_aperture':    f"{r['CR_marg_aperture']:.6e}",
            'S_gehrels':           f"{r['S_gehrels']:.4f}",
            'CR_gehrels_aperture': f"{r['CR_gehrels_aperture']:.6e}",
            # EEF fields (empty when EEF not available)
            'theta_arcmin':        '',
            'eef':                 '',
            'energy_ev':           '',
            'psf_file':            '',
            'eef_extrapolated':    '',
            'eef_capped':          '',
            'CR_marg_total':       '',
            'CR_gehrels_total':    '',
        }
        if eef_info is not None:
            row['theta_arcmin']     = f"{eef_info['theta_arcmin']:.4f}" if eef_info['theta_arcmin'] is not None else ''
            row['eef']              = f"{eef_info['eef']:.6f}"
            row['energy_ev']        = f"{eef_info['energy_ev']:.0f}" if eef_info['energy_ev'] is not None else ''
            row['psf_file']         = os.path.basename(eef_info['psf_file']) if eef_info['psf_file'] else ''
            row['eef_extrapolated'] = str(eef_info['extrapolated'])
            row['eef_capped']       = (f"{eef_info['eef_capped']:.6f}"
                                       if eef_info['eef_capped'] is not None else '')
            if r['CR_marg_total'] is not None:
                row['CR_marg_total']    = f"{r['CR_marg_total']:.6e}"
                row['CR_gehrels_total'] = f"{r['CR_gehrels_total']:.6e}"
        rows.append(row)
    return rows


def write_results_csv(rows, out_dir, obsid):
    """
    Write upper-limit results to a CSV file in out_dir.

    Parameters
    ----------
    rows    : list of dicts from _build_csv_rows()
    out_dir : str  — output directory (created if absent)
    obsid   : str

    Returns
    -------
    csv_path : str
    """
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, f"xmm_uplim_{obsid}.csv")

    fieldnames = [
        'result_type',
        'obsid', 'date_obs', 'instrument', 'energy_lo_kev', 'energy_hi_kev',
        'N_src', 'N_bkg_raw', 'B_scaled', 'area_ratio',
        't_eff_s',
        'theta_arcmin', 'eef', 'energy_ev', 'psf_file', 'eef_extrapolated',
        'eef_capped',
        'confidence_level',
        'CR_net', 'CR_sigma',
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

    # Also write Excel-native .xlsx so obsids with leading zeros display correctly
    from ..output import write_results_xlsx
    xlsx_path = csv_path.replace('.csv', '.xlsx')
    if write_results_xlsx(rows, fieldnames, xlsx_path, text_cols=('obsid',)):
        print(f"  Excel file written: {xlsx_path}")

    return csv_path


# =============================================================================
# INTERNAL HELPERS
# =============================================================================

def _evt_pixel_to_sky(cx, cy, evt_hdr):
    """
    Invert sky_to_evt_pixel: convert event-file pixel (cx, cy) back to
    (ra_deg, dec_deg).

    Uses the same per-column WCS keywords (TCRPXn, TCRVLn, TCDLTn) as
    sky_to_evt_pixel, just solving for ra/dec instead of pixel.
    """
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


# =============================================================================
# DIAGNOSTIC PLOTS
# =============================================================================

def _save_plots(evt_x, evt_y, cx_evt, cy_evt, pscale_evt,
                exp_meta, exp_stats,
                instrument, e_lo, e_hi, cfg, out_dir,
                src_coord, bkg_cx_evt, bkg_cy_evt, obsid_str):
    """
    Save diagnostic plots for one instrument.

    Reuses the shared plot functions from xray_uplim.plots.
    These functions accept (evt_x, evt_y, cx, cy, pscale, label, ...) and are
    instrument-agnostic — originally written for NuSTAR but work for any
    mission whose event file exposes X/Y sky pixel columns.
    """
    try:
        from ..plots import radial_profile, exposure_histogram, region_image
    except ImportError:
        warnings.warn("Diagnostic plots skipped (xray_uplim.plots import failed).",
                      RuntimeWarning, stacklevel=2)
        return

    radial_profile(
        evt_x, evt_y, cx_evt, cy_evt, pscale_evt,
        instrument, e_lo, e_hi, obsid_str, cfg, out_dir)

    exposure_histogram(exp_meta, exp_stats, instrument, cfg, out_dir)

    region_image(
        evt_x, evt_y, cx_evt, cy_evt, pscale_evt,
        instrument, e_lo, e_hi, obsid_str, cfg, out_dir,
        src_ra_deg  = src_coord.ra.deg,
        src_dec_deg = src_coord.dec.deg,
        bkg_cx_evt  = bkg_cx_evt,
        bkg_cy_evt  = bkg_cy_evt)


# =============================================================================
# PER-OBS / PER-INSTRUMENT EXTRACTION
# =============================================================================

def _load_and_extract_instrument(instrument, obsid_str, obs_data_dir, cfg,
                                  run_gui=False):
    """
    Load event file and exposure map, extract counts and exposure for one
    (instrument, obsid) pair.

    Parameters
    ----------
    instrument    : 'MOS1', 'MOS2', or 'PN'
    obsid_str     : str — single obsid string
    obs_data_dir  : str — directory containing this obs's data files
    cfg           : XMMConfig (may be a shallow copy with per-obs overrides)
    run_gui       : bool — whether to open the interactive region selector

    Returns
    -------
    dict with keys: instrument, obsid_str, date_obs, N_src, N_bkg_raw,
        B_scaled, area_ratio, t_eff, exp_stats, eef_info, e_lo, e_hi,
        bkg_cx_evt, bkg_cy_evt
    """
    from ..coords import sky_to_evt_pixel, sky_to_img_pixel

    e_lo, e_hi = cfg.resolve_energy_band()
    out_dir    = os.path.join(obs_data_dir, "ul_products")

    print(f"\n{'='*70}")
    print(f"  {instrument}  [obs: {obsid_str}]")
    print(f"{'='*70}")

    # -- Step 1: locate files -------------------------------------------------
    evt_file, exp_file = locate_files(obs_data_dir, obsid_str, instrument, cfg)
    print(f"  Event file : {os.path.basename(evt_file)}")
    print(f"  Expmap     : {os.path.basename(exp_file)}")

    # -- Step 2: load events --------------------------------------------------
    events, evt_hdr, pi_lo, pi_hi = load_events(cfg, evt_file, instrument)
    date_obs = str(evt_hdr.get('DATE-OBS', '')).strip()

    # -- Step 3: load exposure map --------------------------------------------
    exp_data, exp_hdr = load_expmap(exp_file)

    # -- Step 4: source pixel position ----------------------------------------
    src_coord = parse_coord(cfg.ra, cfg.dec)

    cx_evt, cy_evt, pscale_evt = sky_to_evt_pixel(
        src_coord.ra.deg, src_coord.dec.deg, evt_hdr)
    cx_exp, cy_exp, pscale_exp = sky_to_img_pixel(
        src_coord.ra.deg, src_coord.dec.deg, exp_hdr)

    evt_x = np.asarray(events['X'], dtype=float)
    evt_y = np.asarray(events['Y'], dtype=float)

    print(f"\n  Event X range       : [{evt_x.min():.0f}, {evt_x.max():.0f}]")
    print(f"  Event Y range       : [{evt_y.min():.0f}, {evt_y.max():.0f}]")
    print(f"  Source pixel (evt)  : ({cx_evt:.1f}, {cy_evt:.1f})")
    print(f"  Source pixel (exp)  : ({cx_exp:.1f}, {cy_exp:.1f})")
    print(f"  Pixel scale (evt)   : {pscale_evt:.4f} \"/pix")
    print(f"  Pixel scale (exp)   : {pscale_exp:.3f} \"/pix")

    x_ok = evt_x.min() <= cx_evt <= evt_x.max()
    y_ok = evt_y.min() <= cy_evt <= evt_y.max()
    if not (x_ok and y_ok):
        print(f"  !! WARNING: source pixel is OUTSIDE the event X/Y range — "
              f"check your coordinates!")
    else:
        print(f"  Source position is inside the event image. Good.")

    # -- Step 5: interactive region selector (optional) -----------------------
    bkg_cx_evt = cx_evt
    bkg_cy_evt = cy_evt

    if run_gui:
        from ..region_selector import select_regions_interactive
        print(f"\n  Opening interactive region selector for {instrument}...")
        sel = select_regions_interactive(
            evt_x, evt_y, cx_evt, cy_evt, pscale_evt, cfg, instrument)

        cx_evt = sel['cx']
        cy_evt = sel['cy']
        bkg_cx_evt = sel['bkg_cx']
        bkg_cy_evt = sel['bkg_cy']

        cfg.src_radius_arcsec = sel['src_radius_arcsec']
        cfg.bkg_radius_arcsec = sel['bkg_radius_arcsec']
        cfg.bkg_inner_factor  = sel['bkg_inner_factor']

        # If background was moved to a separate position, switch to manual mode
        bkg_moved = (abs(bkg_cx_evt - cx_evt) > 1.0 or
                     abs(bkg_cy_evt - cy_evt) > 1.0)
        if bkg_moved:
            try:
                bkg_ra, bkg_dec = _evt_pixel_to_sky(bkg_cx_evt, bkg_cy_evt,
                                                     evt_hdr)
                cfg.bkg_mode = 'manual'
                cfg.bkg_ra   = str(float(bkg_ra))
                cfg.bkg_dec  = str(float(bkg_dec))
                print(f"  [GUI] Background → manual mode: "
                      f"RA={bkg_ra:.5f}  Dec={bkg_dec:.5f}")
            except Exception as exc:
                warnings.warn(
                    f"Could not convert background pixel to RA/Dec ({exc}). "
                    "Falling back to annulus mode.",
                    RuntimeWarning, stacklevel=2)
                cfg.bkg_mode = 'annulus'
                bkg_cx_evt   = cx_evt
                bkg_cy_evt   = cy_evt

    elif cfg.bkg_mode == 'manual' and cfg.bkg_ra and cfg.bkg_dec:
        # Reproject manual bkg sky coords to this obs's pixel frame
        bkg_coord = parse_coord(cfg.bkg_ra, cfg.bkg_dec)
        bkg_cx_evt, bkg_cy_evt, _ = sky_to_evt_pixel(
            bkg_coord.ra.deg, bkg_coord.dec.deg, evt_hdr)

    print(f"\n  Src aperture : {cfg.src_radius_arcsec:.1f}\"")
    if cfg.bkg_mode == 'annulus':
        r_in = cfg.src_radius_arcsec * cfg.bkg_inner_factor
        print(f"  Bkg annulus  : {r_in:.1f}\" — {cfg.bkg_radius_arcsec:.1f}\"")
    else:
        print(f"  Bkg circle   : r={cfg.bkg_radius_arcsec:.1f}\"  (manual centre)")

    # -- Step 6: source and background counts ---------------------------------
    # extract_src_bkg_counts prints the geometric ratio for reference; the
    # exposure-weighted ratio (Tier 1) is computed after loading the expmap.
    print()
    N_src, N_bkg_raw, area_ratio_geom, cx_evt, cy_evt, pscale_evt = \
        extract_src_bkg_counts(events, evt_hdr, cfg, instrument,
                               bkg_cx_evt=bkg_cx_evt, bkg_cy_evt=bkg_cy_evt)

    # -- Step 7: effective exposure -------------------------------------------
    print()
    exp_stats, exp_meta, cx_exp, cy_exp = extract_exposure(
        exp_data, exp_hdr, cfg, instrument)

    # -- Tier 1: exposure-weighted area ratio ---------------------------------
    r_src_exp = cfg.src_radius_arcsec / pscale_exp
    try:
        if cfg.bkg_mode == 'annulus':
            r_inner_exp = (cfg.src_radius_arcsec *
                           cfg.bkg_inner_factor) / pscale_exp
            r_outer_exp = cfg.bkg_radius_arcsec / pscale_exp
            area_ratio = compute_exposure_area_ratio(
                exp_data, cx_exp, cy_exp, r_src_exp,
                'annulus',
                r_bkg_inner_pix=r_inner_exp,
                r_bkg_outer_pix=r_outer_exp)
        else:  # manual
            bkg_coord_exp = parse_coord(cfg.bkg_ra, cfg.bkg_dec)
            cx_bkg_exp, cy_bkg_exp, _ = sky_to_img_pixel(
                bkg_coord_exp.ra.deg, bkg_coord_exp.dec.deg, exp_hdr)
            r_bkg_exp = cfg.bkg_radius_arcsec / pscale_exp
            area_ratio = compute_exposure_area_ratio(
                exp_data, cx_exp, cy_exp, r_src_exp,
                'manual',
                cx_bkg=cx_bkg_exp, cy_bkg=cy_bkg_exp,
                r_bkg_pix=r_bkg_exp)
    except RuntimeError as _exc:
        warnings.warn(
            f"Exposure-weighted area ratio failed for {instrument} "
            f"({_exc}); using geometric ratio.",
            UserWarning, stacklevel=2)
        area_ratio = area_ratio_geom

    B_scaled = N_bkg_raw * area_ratio

    print(f"  Area ratio   (exp-weighted) : {area_ratio:.5f}")
    print(f"  Scaled bkg   B             : {B_scaled:.3f} cts")
    print(f"  Net counts   (N_src - B)   : {N_src - B_scaled:.3f} cts")

    print(f"\n  -- Exposure statistics ------------------------------------------")
    for key, lbl in [('median',       'Median        [RECOMMENDED]        '),
                     ('mean',         'Mean          [diagnostic]         '),
                     ('psf_weighted', 'PSF-wtd mean  [on-axis diag. only] ')]:
        tag = ' <-- PRIMARY' if key == cfg.exp_stat else ''
        print(f"    {lbl} : {exp_stats[key]/1e3:7.3f} ks{tag}")

    t_eff = exp_stats[cfg.exp_stat]
    print(f"\n  Using t_eff = {t_eff/1e3:.3f} ks  ({cfg.exp_stat})")

    # -- Step 8: EEF from CCF PSF ---------------------------------------------
    eef_info = None
    try:
        eef_info = compute_xmm_eef(
            cfg, instrument, evt_hdr,
            cfg.src_radius_arcsec, e_lo, e_hi)

        print(f"\n  -- EEF (Encircled Energy Fraction) ----------------------------")
        print(f"    Off-axis angle   : {eef_info['theta_arcmin']:.3f} arcmin")
        print(f"    Pointing         : RA={eef_info['pointing_ra']:.5f}  "
              f"Dec={eef_info['pointing_dec']:.5f}")
        print(f"    PSF file         : {os.path.basename(eef_info['psf_file'])}")
        print(f"    Band-centre E    : {eef_info['energy_ev']:.0f} eV")
        print(f"    EEF at {cfg.src_radius_arcsec:.0f}\"       : {eef_info['eef']:.4f}")
        if eef_info['extrapolated']:
            print(f"    !! Off-axis angle exceeds CCF limit (15'). "
                  f"EEF capped at 15' = {eef_info['eef_capped']:.4f}")

    except (RuntimeError, FileNotFoundError, KeyError) as exc:
        warnings.warn(
            f"EEF computation skipped for {instrument}: {exc}\n"
            "Set psf_dir= in your config to enable EEF-corrected upper limits.\n"
            "Download XMM PSF CCF files from:\n"
            "  https://www.cosmos.esa.int/web/xmm-newton/current-calibration-files",
            UserWarning, stacklevel=2)

    # -- Step 9: diagnostic plots ---------------------------------------------
    if cfg.save_plots:
        _save_plots(evt_x, evt_y, cx_evt, cy_evt, pscale_evt,
                    exp_meta, exp_stats,
                    instrument, e_lo, e_hi, cfg, out_dir,
                    src_coord, bkg_cx_evt, bkg_cy_evt, obsid_str=obsid_str)

    return {
        'instrument':  instrument,
        'obsid_str':   obsid_str,
        'date_obs':    date_obs,
        'N_src':       N_src,
        'N_bkg_raw':   N_bkg_raw,
        'B_scaled':    B_scaled,
        'area_ratio':  area_ratio,
        't_eff':       t_eff,
        'exp_stats':   exp_stats,
        'eef_info':    eef_info,
        'e_lo':        e_lo,
        'e_hi':        e_hi,
        'bkg_cx_evt':  bkg_cx_evt,
        'bkg_cy_evt':  bkg_cy_evt,
    }


# =============================================================================
# PER-INSTRUMENT PIPELINE (single-obsid wrapper)
# =============================================================================

def process_instrument(instrument: str, cfg: XMMConfig):
    """
    Full extraction and upper-limit calculation for one EPIC instrument.

    Thin wrapper around _load_and_extract_instrument for single-obsid
    backward compatibility.

    Parameters
    ----------
    instrument : 'MOS1', 'MOS2', or 'PN'
    cfg        : XMMConfig (validated before calling)

    Returns
    -------
    dict with keys:
        instrument, date_obs, N_src, N_bkg_raw, B_scaled, area_ratio,
        net_counts, t_eff_s, exp_stats, ul, energy, eef_info, csv_rows
    """
    obsid_str    = cfg.obsids[0]
    obs_data_dir = cfg.data_dir  # single-obs: data_dir IS the obs dir

    raw = _load_and_extract_instrument(instrument, obsid_str, obs_data_dir,
                                        cfg, run_gui=cfg.use_gui)

    e_lo, e_hi = raw['e_lo'], raw['e_hi']
    eef_val    = raw['eef_info']['eef'] if raw['eef_info'] is not None else None

    ul_results = _print_results_table(raw['N_src'], raw['B_scaled'], raw['t_eff'],
                                       raw['N_bkg_raw'], raw['area_ratio'],
                                       cfg.confidence_levels, eef=eef_val)

    csv_rows = _build_csv_rows(instrument, e_lo, e_hi, raw['N_src'], raw['N_bkg_raw'],
                                raw['B_scaled'], raw['area_ratio'], raw['t_eff'],
                                ul_results, raw['eef_info'], obsid_str,
                                date_obs=raw['date_obs'], result_type='individual')

    return {
        'instrument': instrument,
        'date_obs':   raw['date_obs'],
        'N_src':      raw['N_src'],
        'N_bkg_raw':  raw['N_bkg_raw'],
        'B_scaled':   raw['B_scaled'],
        'area_ratio': raw['area_ratio'],
        'net_counts': raw['N_src'] - raw['B_scaled'],
        't_eff_s':    raw['t_eff'],
        'exp_stats':  raw['exp_stats'],
        'ul':         ul_results,
        'energy':     (e_lo, e_hi),
        'eef_info':   raw['eef_info'],
        'csv_rows':   csv_rows,
    }


# =============================================================================
# MAIN PIPELINE — handles single and multi-obsid
# =============================================================================

def process_observations(cfg: XMMConfig):
    """
    Main pipeline entry point.  Handles single-obsid (backward-compatible)
    and multi-obsid co-adding.

    For multi-obsid mode, counts and exposures are summed per instrument
    across all observations that have valid data for that instrument.
    Individual per-observation upper limits are also reported and written
    to CSV when n_obs > 1.

    Parameters
    ----------
    cfg : XMMConfig (already validated)

    Returns
    -------
    per_obs_raw : dict[obsid_str][instrument] — raw extraction dicts
    """
    obsids = cfg.obsids
    n_obs  = len(obsids)
    src_coord = parse_coord(cfg.ra, cfg.dec)
    e_lo, e_hi = cfg.resolve_energy_band()

    # obsid_label for filenames: single string or "obs1+obs2"
    obsid_label = cfg.obsid if isinstance(cfg.obsid, str) else '+'.join(obsids)

    # Output dir
    out_dir = os.path.join(cfg.data_dir, "ul_products")

    # Save original aperture settings for gui_per_obs
    _orig_aperture = {k: getattr(cfg, k) for k in (
        'src_radius_arcsec', 'bkg_radius_arcsec', 'bkg_inner_factor',
        'bkg_mode', 'bkg_ra', 'bkg_dec')}

    # -- Print header ---------------------------------------------------------
    print("XMM-Newton EPIC Non-Detection Upper Limit")
    print("=" * 70)
    print(f"Source      :  RA = {src_coord.ra.deg:.6f} deg  "
          f"Dec = {src_coord.dec.deg:.6f} deg")
    if isinstance(cfg.energy_band, tuple):
        band_label = f"{e_lo:.2f}–{e_hi:.2f} keV (custom)"
    else:
        band_label = f"'{cfg.energy_band}'  ({e_lo:.2f}–{e_hi:.2f} keV)"
    print(f"Energy band :  {band_label}")
    print(f"Instruments :  {', '.join(cfg.instruments)}")
    print(f"Exp stat    :  {cfg.exp_stat}  (primary)")
    print(f"Bkg mode    :  {cfg.bkg_mode}")
    print(f"Data dir    :  {cfg.data_dir}")
    if n_obs > 1:
        print(f"Obs IDs     :  {', '.join(obsids)}  [{n_obs} observations — co-adding]")
    else:
        print(f"Obs ID      :  {obsids[0]}")
    print()

    # -- Per-obs data directory -----------------------------------------------
    def _obs_data_dir(obsid_str):
        if n_obs == 1:
            return cfg.data_dir
        return os.path.join(cfg.data_dir, obsid_str)

    # -- Load and extract all (obs, instrument) pairs -------------------------
    # per_obs_raw[obsid_str][instrument] = dict from _load_and_extract_instrument
    per_obs_raw = {}

    for i, obsid_str in enumerate(obsids):
        obs_data_dir = _obs_data_dir(obsid_str)

        if n_obs > 1:
            print(f"\n{'#'*70}")
            print(f"  Observation {i+1}/{n_obs}:  {obsid_str}")
            print(f"{'#'*70}")

        # GUI mode for this obs
        if cfg.use_gui and cfg.gui_per_obs:
            cfg_obs = copy.copy(cfg)
            for k, v in _orig_aperture.items():
                setattr(cfg_obs, k, v)
            run_gui_this = True
        elif cfg.use_gui and i == 0:
            cfg_obs = cfg
            run_gui_this = True
        else:
            cfg_obs = cfg
            run_gui_this = False

        per_obs_raw[obsid_str] = {}
        for instrument in cfg.instruments:
            try:
                raw = _load_and_extract_instrument(
                    instrument, obsid_str, obs_data_dir, cfg_obs,
                    run_gui=run_gui_this)
                per_obs_raw[obsid_str][instrument] = raw
            except FileNotFoundError as exc:
                warnings.warn(
                    f"\nSkipping {instrument} obs {obsid_str}: {exc}",
                    UserWarning, stacklevel=2)
                continue

    # -- Per-instrument combined results --------------------------------------
    all_csv_rows = []

    for instrument in cfg.instruments:
        # Collect obs that have data for this instrument
        obs_with_data = [oid for oid in obsids
                         if instrument in per_obs_raw.get(oid, {})]
        if not obs_with_data:
            continue

        print(f"\n{'='*70}")
        if n_obs > 1:
            print(f"  {instrument} — co-added across {len(obs_with_data)} observations")
        else:
            print(f"  {instrument} — results")
        print(f"{'='*70}")

        # Individual per-obs ULs (only when n_obs > 1)
        if n_obs > 1:
            print(f"\n  -- Individual per-observation upper limits ({instrument}) --")
            for obsid_str in obs_with_data:
                raw = per_obs_raw[obsid_str][instrument]
                eef_ind = raw['eef_info']['eef'] if raw['eef_info'] else None
                ul_ind = _compute_ul_results(raw['N_src'], raw['B_scaled'],
                                              raw['t_eff'], raw['N_bkg_raw'],
                                              raw['area_ratio'],
                                              cfg.confidence_levels, eef=eef_ind)
                print(f"\n  Obs {obsid_str}:  N_src={raw['N_src']}  "
                      f"B={raw['B_scaled']:.2f}  t_eff={raw['t_eff']/1e3:.3f} ks")
                for r in ul_ind:
                    val = (r['CR_marg_total'] if r['CR_marg_total'] is not None
                           else r['CR_marg_aperture'])
                    print(f"    {r['cl']*100:.1f}%:  < {val:.3e} cts/s")

                ind_rows = _build_csv_rows(
                    instrument, raw['e_lo'], raw['e_hi'],
                    raw['N_src'], raw['N_bkg_raw'], raw['B_scaled'],
                    raw['area_ratio'], raw['t_eff'], ul_ind, raw['eef_info'],
                    obsid_str, date_obs=raw['date_obs'], result_type='individual')
                all_csv_rows.extend(ind_rows)

        # Combined counts / exposure across observations for this instrument
        N_total     = sum(per_obs_raw[oid][instrument]['N_src']     for oid in obs_with_data)
        B_total     = sum(per_obs_raw[oid][instrument]['B_scaled']  for oid in obs_with_data)
        N_bkg_total = sum(per_obs_raw[oid][instrument]['N_bkg_raw'] for oid in obs_with_data)
        area_ratio  = per_obs_raw[obs_with_data[0]][instrument]['area_ratio']
        t_total     = sum(per_obs_raw[oid][instrument]['t_eff']     for oid in obs_with_data)
        e_lo_i      = per_obs_raw[obs_with_data[0]][instrument]['e_lo']
        e_hi_i      = per_obs_raw[obs_with_data[0]][instrument]['e_hi']

        # EEF: exposure-weighted average
        all_eef = [per_obs_raw[oid][instrument]['eef_info'] for oid in obs_with_data]
        if all(e is not None for e in all_eef):
            t_eef_sum = sum(
                per_obs_raw[oid][instrument]['t_eff'] *
                per_obs_raw[oid][instrument]['eef_info']['eef']
                for oid in obs_with_data)
            eef_avg = t_eef_sum / t_total if t_total > 0 else None
            eef_combined_info = {
                'eef':           eef_avg,
                'theta_arcmin':  None,
                'pointing_ra':   None,
                'pointing_dec':  None,
                'psf_file':      all_eef[0]['psf_file'],   # representative
                'energy_ev':     all_eef[0]['energy_ev'],
                'extrapolated':  any(e['extrapolated'] for e in all_eef),
                'eef_capped':    None,
            }
            eef_val = eef_avg

            if n_obs > 1:
                print(f"\n  -- EEF across {len(obs_with_data)} observations "
                      f"({instrument}) --")
                for oid in obs_with_data:
                    raw = per_obs_raw[oid][instrument]
                    ei  = raw['eef_info']
                    print(f"    {oid}: theta={ei['theta_arcmin']:.3f}'  "
                          f"EEF={ei['eef']:.4f}  "
                          f"t*EEF={raw['t_eff']*ei['eef']/1e3:.3f} ks")
                print(f"    Exposure-weighted EEF = {eef_avg:.4f}")
        else:
            eef_combined_info = None
            eef_val = None

        if n_obs > 1:
            print(f"\n  -- Combined ({len(obs_with_data)} obs, {instrument}): "
                  f"N_src={N_total}  B={B_total:.2f}  t_eff={t_total/1e3:.3f} ks --")

        ul_combined = _print_results_table(
            N_total, B_total, t_total, N_bkg_total, area_ratio,
            cfg.confidence_levels, eef=eef_val)

        date_obs_first = per_obs_raw[obs_with_data[0]][instrument].get('date_obs', '')
        comb_obsid = obsid_label if n_obs > 1 else obsids[0]
        comb_rtype = 'combined' if n_obs > 1 else 'individual'

        comb_rows = _build_csv_rows(
            instrument, e_lo_i, e_hi_i, N_total, N_bkg_total, B_total,
            area_ratio, t_total, ul_combined, eef_combined_info,
            comb_obsid, date_obs=date_obs_first, result_type=comb_rtype)
        all_csv_rows.extend(comb_rows)

    # -- Summary table --------------------------------------------------------
    # Collect combined results for summary display
    combined_results = {}
    for instrument in cfg.instruments:
        obs_with_data = [oid for oid in obsids
                         if instrument in per_obs_raw.get(oid, {})]
        if not obs_with_data:
            continue
        N_total = sum(per_obs_raw[oid][instrument]['N_src']    for oid in obs_with_data)
        B_total = sum(per_obs_raw[oid][instrument]['B_scaled'] for oid in obs_with_data)
        t_total = sum(per_obs_raw[oid][instrument]['t_eff']    for oid in obs_with_data)
        eef_infos = [per_obs_raw[oid][instrument]['eef_info'] for oid in obs_with_data]
        eef_str = 'N/A'
        if all(e is not None for e in eef_infos):
            t_eef_sum = sum(per_obs_raw[oid][instrument]['t_eff'] *
                            per_obs_raw[oid][instrument]['eef_info']['eef']
                            for oid in obs_with_data)
            eef_avg = t_eef_sum / t_total if t_total > 0 else None
            if eef_avg is not None:
                eef_str = f"{eef_avg:.3f}"
        combined_results[instrument] = {
            'N_src':   N_total,
            'B_scaled': B_total,
            't_eff_s':  t_total,
            'eef_str':  eef_str,
        }
        # Attach ul_results for this instrument from all_csv_rows (recompute)
        area_ratio = per_obs_raw[obs_with_data[0]][instrument]['area_ratio']
        N_bkg_total = sum(per_obs_raw[oid][instrument]['N_bkg_raw'] for oid in obs_with_data)
        eef_val2 = float(eef_str) if eef_str != 'N/A' else None
        ul_sum = _compute_ul_results(N_total, B_total, t_total, N_bkg_total,
                                      area_ratio, cfg.confidence_levels, eef=eef_val2)
        combined_results[instrument]['ul'] = ul_sum

    if combined_results:
        print(f"\n{'='*70}")
        print("  SUMMARY")
        print(f"{'='*70}")
        print(f"  {'Instrument':<8}  {'N_src':>6}  {'B_scaled':>9}  "
              f"{'t_eff (ks)':>11}  {'EEF':>6}  "
              f"{'Marg CR_ap (3σ)':>18}")
        print("  " + "-" * 68)
        for instrument, cr in combined_results.items():
            ul_row = next((u for u in cr['ul'] if u['cl'] >= 0.997), cr['ul'][-1])
            print(f"  {instrument:<8}  {cr['N_src']:>6}  "
                  f"{cr['B_scaled']:>9.2f}  "
                  f"{cr['t_eff_s']/1e3:>11.3f}  "
                  f"{cr['eef_str']:>6}  "
                  f"{ul_row['CR_marg_aperture']:>18.4e}")
        print()

    # -- Write CSV + XLSX -----------------------------------------------------
    os.makedirs(out_dir, exist_ok=True)
    if all_csv_rows:
        write_results_csv(all_csv_rows, out_dir, obsid_label)

    print("\nDone.")
    return per_obs_raw


# =============================================================================
# ENTRY POINT
# =============================================================================

def run_uplim(data_dir, obsid, ra, dec, **kwargs):
    """
    Run the full XMM-Newton upper-limit pipeline.

    Parameters
    ----------
    data_dir : str           — ODF working directory (epproc/emproc output)
    obsid    : str or list   — XMM observation ID(s).  Pass a list to co-add.
    ra       : str or float  — source RA  ("HH:MM:SS" or decimal degrees)
    dec      : str or float  — source Dec ("±DD:MM:SS" or decimal degrees)
    **kwargs : any XMMConfig field, e.g.
                   instruments=['MOS1', 'PN'],
                   energy_band='soft',
                   src_radius_arcsec=20.0,
                   confidence_levels=[0.9973],
                   psf_dir='/path/to/ccf',
                   save_plots=True

    Returns
    -------
    per_obs_raw : dict[obsid_str][instrument]
    """
    cfg = XMMConfig(data_dir=data_dir, obsid=obsid, ra=ra, dec=dec, **kwargs)
    cfg.validate()
    return process_observations(cfg)
