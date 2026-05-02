"""
xray_uplim.swift.pipeline
--------------------------
Full upper-limit pipeline for Swift XRT.

Readout mode (PC / WT) is auto-detected from the event files present on
disk.  Results are reported for the single XRT instrument.

Public API
----------
run_uplim(**kwargs)          — entry point; builds SwiftConfig and runs
process_observation(cfg)     — full pipeline for one Swift observation
"""

import copy
import csv
import os
import warnings
import numpy as np

from .config   import SwiftConfig
from .io       import locate_files, load_events, load_expmap
from .aperture import extract_src_bkg_counts, extract_exposure
from .eef      import compute_swift_eef
from ..coords  import parse_coord, sky_to_evt_pixel, sky_to_img_pixel
from ..statistics import net_count_rate, kraft_upper_limit, gehrels_upper_limit


# =============================================================================
# RESULTS TABLE
# =============================================================================

def _compute_ul_results(N_src, B_scaled, t_eff, N_bkg_raw, area_ratio,
                         confidence_levels, eef=None):
    """
    Compute upper limits at every confidence level.  No output printed.
    Returns a list of dicts, one per CL.
    """
    CR_net, CR_sigma = net_count_rate(N_src, B_scaled, t_eff,
                                       N_bkg_raw, area_ratio)
    results = []
    for cl in confidence_levels:
        S_k  = kraft_upper_limit(N_src, B_scaled, cl)
        S_g  = gehrels_upper_limit(N_src, B_scaled, cl)
        CR_k_ap = S_k / t_eff
        CR_g_ap = S_g / t_eff
        CR_k_tot = S_k / (t_eff * eef) if (eef is not None and eef > 0) else None
        CR_g_tot = S_g / (t_eff * eef) if (eef is not None and eef > 0) else None
        results.append({
            'cl':                  cl,
            'CR_net':              CR_net,
            'CR_sigma':            CR_sigma,
            'S_kraft':             S_k,
            'CR_kraft_aperture':   CR_k_ap,
            'CR_kraft_total':      CR_k_tot,
            'S_gehrels':           S_g,
            'CR_gehrels_aperture': CR_g_ap,
            'CR_gehrels_total':    CR_g_tot,
        })
    return results


def _print_results_table(N_src, B_scaled, t_eff, N_bkg_raw, area_ratio,
                          confidence_levels, eef=None):
    """Compute and print upper limits at every confidence level."""
    results = _compute_ul_results(N_src, B_scaled, t_eff, N_bkg_raw,
                                   area_ratio, confidence_levels, eef)
    CR_net   = results[0]['CR_net']
    CR_sigma = results[0]['CR_sigma']

    print(f"\n  Point estimate  (N_src - B) / t_eff  [NOT an upper limit]")
    print(f"    = ({N_src} - {B_scaled:.1f}) / {t_eff:.1f} s")
    print(f"    = {CR_net:+.4e} cts/s  ±  {CR_sigma:.4e}  (1-sigma Poisson)")
    if CR_net < 0:
        print(f"    (Negative — source aperture below expected background: "
              f"clean non-detection)")

    print(f"\n  Upper limits:")
    if eef is not None:
        header = (
            f"  {'CL':>8}  {'Net CR':>13}  "
            f"{'Kraft S_ul':>10}  {'Kraft CR_ap':>13}  {'Kraft CR_tot':>13}  "
            f"{'Geh S_ul':>10}  {'Geh CR_ap':>13}  {'Geh CR_tot':>13}"
        )
    else:
        header = (
            f"  {'CL':>8}  {'Net CR':>13}  "
            f"{'Kraft S_ul':>10}  {'Kraft CR_ap':>13}  "
            f"{'Geh S_ul':>10}  {'Geh CR_ap':>13}"
        )
    divider = "  " + "-" * (len(header) - 2)
    print(header)
    print(divider)

    for r in results:
        if eef is not None:
            print(
                f"  {r['cl']:8.4f}  {CR_net:+13.4e}  "
                f"{r['S_kraft']:10.3f}  {r['CR_kraft_aperture']:13.4e}  "
                f"{r['CR_kraft_total']:13.4e}  "
                f"{r['S_gehrels']:10.3f}  {r['CR_gehrels_aperture']:13.4e}  "
                f"{r['CR_gehrels_total']:13.4e}"
            )
        else:
            print(
                f"  {r['cl']:8.4f}  {CR_net:+13.4e}  "
                f"{r['S_kraft']:10.3f}  {r['CR_kraft_aperture']:13.4e}  "
                f"{r['S_gehrels']:10.3f}  {r['CR_gehrels_aperture']:13.4e}"
            )

    print(divider)
    if eef is not None:
        print(f"  CR_ap  = aperture count-rate upper limit = S_ul / t_eff.")
        print(f"  CR_tot = EEF-corrected total source rate = S_ul / (t_eff × EEF).")
        print(f"  EEF used: {eef:.4f}")
    else:
        print(f"  CR_ap is the aperture count-rate upper limit.")
        print(f"  EEF correction skipped (psfconst_xrt.fits not found).")

    return results


# =============================================================================
# CSV OUTPUT
# =============================================================================

def _build_csv_rows(mode, e_lo, e_hi, N_src, N_bkg_raw, B_scaled,
                    area_ratio, t_eff, ul_results, eef_info, obsid,
                    result_type='individual', date_obs=''):
    """
    Build a list of CSV row dicts (one per confidence level).

    result_type : 'individual' for per-obs rows, 'combined' for the stacked total.
    date_obs    : DATE-OBS from the event file header (ISO 8601 string).
                  For combined rows this is the date of the first observation.
    """
    rows = []
    for r in ul_results:
        row = {
            'result_type':         result_type,
            'obsid':               obsid,
            'date_obs':            date_obs,
            'mode':                mode,
            'energy_lo_kev':       e_lo,
            'energy_hi_kev':       e_hi,
            'N_src':               N_src,
            'N_bkg_raw':           N_bkg_raw,
            'B_scaled':            f"{B_scaled:.4f}",
            'area_ratio':          f"{area_ratio:.6f}",
            't_eff_s':             f"{t_eff:.2f}",
            'confidence_level':    r['cl'],
            'CR_net':              f"{r['CR_net']:.6e}",
            'CR_sigma':            f"{r['CR_sigma']:.6e}",
            'S_kraft':             f"{r['S_kraft']:.4f}",
            'CR_kraft_aperture':   f"{r['CR_kraft_aperture']:.6e}",
            'S_gehrels':           f"{r['S_gehrels']:.4f}",
            'CR_gehrels_aperture': f"{r['CR_gehrels_aperture']:.6e}",
            # EEF fields (empty when EEF skipped)
            'theta_arcmin':        '',
            'eef':                 '',
            'energy_kev':          '',
            'psf_file':            '',
            'eef_extrapolated':    '',
            'eef_capped':          '',
            'CR_kraft_total':      '',
            'CR_gehrels_total':    '',
        }
        if eef_info is not None:
            row['theta_arcmin']     = f"{eef_info['theta_arcmin']:.4f}"
            row['eef']              = f"{eef_info['eef']:.6f}"
            row['energy_kev']       = f"{eef_info['energy_kev']:.3f}"
            row['psf_file']         = os.path.basename(eef_info['psf_file'])
            row['eef_extrapolated'] = str(eef_info['extrapolated'])
            row['eef_capped']       = (f"{eef_info['eef_capped']:.6f}"
                                       if eef_info['eef_capped'] is not None
                                       else '')
            if r['CR_kraft_total'] is not None:
                row['CR_kraft_total']   = f"{r['CR_kraft_total']:.6e}"
                row['CR_gehrels_total'] = f"{r['CR_gehrels_total']:.6e}"
        rows.append(row)
    return rows


def write_results_csv(rows, out_dir, obsid):
    """Write upper-limit results to CSV."""
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, f"swift_uplim_{obsid}.csv")

    fieldnames = [
        'result_type', 'obsid', 'date_obs', 'mode', 'energy_lo_kev', 'energy_hi_kev',
        'N_src', 'N_bkg_raw', 'B_scaled', 'area_ratio',
        't_eff_s',
        'theta_arcmin', 'eef', 'energy_kev', 'psf_file',
        'eef_extrapolated', 'eef_capped',
        'confidence_level',
        'CR_net', 'CR_sigma',
        'S_kraft',   'CR_kraft_aperture',   'CR_kraft_total',
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
    if write_results_xlsx(rows, fieldnames, xlsx_path,
                          text_cols=('result_type', 'obsid', 'mode')):
        print(f"  Excel file written: {xlsx_path}")

    return csv_path


# =============================================================================
# INTERNAL HELPERS
# =============================================================================

def _evt_pixel_to_sky(cx, cy, evt_hdr):
    """Invert sky_to_evt_pixel: event-file pixel (cx, cy) → (ra_deg, dec_deg)."""
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


def _load_one_obs(cfg, obs_root, obsid_str):
    """
    Steps 1–4 for one observation: locate → load events → load expmap →
    compute source pixel position.

    Returns a dict with all raw products for that observation.
    """
    # Step 1
    evt_file, exp_file, mode = locate_files(obs_root, obsid_str, cfg)

    # Step 2
    events, evt_hdr, pi_lo, pi_hi = load_events(cfg, evt_file, mode)

    # Step 3
    if exp_file is not None:
        exp_data, exp_hdr = load_expmap(exp_file)
    else:
        exp_data = exp_hdr = None
        warnings.warn(
            "No exposure map found — effective exposure will be estimated "
            "from the event file ONTIME header keyword.",
            UserWarning, stacklevel=2)

    # Step 4
    src_coord = parse_coord(cfg.ra, cfg.dec)
    cx_evt, cy_evt, pscale_evt = sky_to_evt_pixel(
        src_coord.ra.deg, src_coord.dec.deg, evt_hdr)

    evt_x = np.asarray(events['X'], dtype=float)
    evt_y = np.asarray(events['Y'], dtype=float)

    print(f"\n  Event X range       : [{evt_x.min():.0f}, {evt_x.max():.0f}]")
    print(f"  Event Y range       : [{evt_y.min():.0f}, {evt_y.max():.0f}]")
    print(f"  Source pixel (evt)  : ({cx_evt:.1f}, {cy_evt:.1f})")
    print(f"  Pixel scale (evt)   : {pscale_evt:.4f} \"/pix")

    x_ok = evt_x.min() <= cx_evt <= evt_x.max()
    y_ok = evt_y.min() <= cy_evt <= evt_y.max()
    if not (x_ok and y_ok):
        print("  !! WARNING: source pixel is OUTSIDE the event X/Y range — "
              "check your coordinates!")
    else:
        print("  Source position is inside the event image. Good.")

    date_obs = str(evt_hdr.get('DATE-OBS', '')).strip()

    return dict(
        mode=mode, events=events, evt_hdr=evt_hdr,
        evt_x=evt_x, evt_y=evt_y,
        cx_evt=cx_evt, cy_evt=cy_evt, pscale_evt=pscale_evt,
        exp_data=exp_data, exp_hdr=exp_hdr,
        pi_lo=pi_lo, pi_hi=pi_hi,
        src_coord=src_coord,
        date_obs=date_obs,
    )


def _run_gui_first_obs(cfg, obs):
    """
    Run the interactive region selector for the FIRST observation.

    Mutates cfg with the user's chosen aperture and background settings.
    Returns (bkg_cx_evt, bkg_cy_evt) in event-file pixels for this obs,
    and stores sky-coordinate background position in cfg so later obs
    can re-project it themselves.
    """
    label      = f"XRT-{obs['mode']}"
    bkg_cx_evt = obs['cx_evt']
    bkg_cy_evt = obs['cy_evt']

    from ..region_selector import select_regions_interactive
    print(f"\n  Opening interactive region selector for {label}...")
    sel = select_regions_interactive(
        obs['evt_x'], obs['evt_y'],
        obs['cx_evt'], obs['cy_evt'], obs['pscale_evt'],
        cfg, label)

    bkg_cx_evt = sel['bkg_cx']
    bkg_cy_evt = sel['bkg_cy']

    cfg.src_radius_arcsec = sel['src_radius_arcsec']
    cfg.bkg_radius_arcsec = sel['bkg_radius_arcsec']
    cfg.bkg_inner_factor  = sel['bkg_inner_factor']

    bkg_moved = (abs(bkg_cx_evt - obs['cx_evt']) > 1.0 or
                 abs(bkg_cy_evt - obs['cy_evt']) > 1.0)
    if bkg_moved:
        try:
            bkg_ra, bkg_dec = _evt_pixel_to_sky(bkg_cx_evt, bkg_cy_evt,
                                                  obs['evt_hdr'])
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
            bkg_cx_evt   = obs['cx_evt']
            bkg_cy_evt   = obs['cy_evt']

    return bkg_cx_evt, bkg_cy_evt


def _extract_counts_exposure_eef(cfg, obs, bkg_cx_evt, bkg_cy_evt, e_lo, e_hi):
    """
    Steps 6–8: extract counts, effective exposure, and EEF for one observation.

    For subsequent (non-GUI) observations, bkg_cx_evt / bkg_cy_evt should be
    None so that extract_src_bkg_counts re-projects cfg.bkg_ra / bkg_dec.

    Returns a dict with all per-obs quantities needed for accumulation.
    """
    mode = obs['mode']
    label = f"XRT-{mode}"

    # Step 6
    N_src, N_bkg_raw, area_ratio, cx_out, cy_out, pscale_out = \
        extract_src_bkg_counts(obs['events'], obs['evt_hdr'], cfg, mode,
                               bkg_cx_evt=bkg_cx_evt, bkg_cy_evt=bkg_cy_evt)
    B_scaled = N_bkg_raw * area_ratio

    print(f"  Area ratio   (src / bkg) : {area_ratio:.5f}")
    print(f"  Scaled bkg   B           : {B_scaled:.3f} cts")
    print(f"  Net counts   (N_src - B) : {N_src - B_scaled:.3f} cts")

    # Step 7
    print()
    if obs['exp_data'] is not None:
        exp_stats, exp_meta, cx_exp, cy_exp = extract_exposure(
            obs['exp_data'], obs['exp_hdr'], cfg)
        print(f"\n  -- Exposure statistics ------------------------------------------")
        for key, lbl in [('median',       'Median        [RECOMMENDED]        '),
                         ('mean',         'Mean          [diagnostic]         '),
                         ('psf_weighted', 'PSF-wtd mean  [on-axis diag. only] ')]:
            tag = ' <-- PRIMARY' if key == cfg.exp_stat else ''
            print(f"    {lbl} : {exp_stats[key]/1e3:7.3f} ks{tag}")
        t_eff = exp_stats[cfg.exp_stat]
    else:
        ontime = float(obs['evt_hdr'].get('ONTIME',
                       obs['evt_hdr'].get('EXPOSURE', 0.0)))
        t_eff = ontime
        exp_stats = {'median': ontime, 'mean': ontime, 'psf_weighted': ontime}
        exp_meta  = None
        cx_exp = cy_exp = None
        warnings.warn(
            f"No exposure map — using ONTIME={ontime:.0f} s from event header. "
            "This ignores bad columns and vignetting. Run xrtexpomap for accuracy.",
            UserWarning, stacklevel=2)

    print(f"\n  Using t_eff = {t_eff/1e3:.3f} ks  ({cfg.exp_stat})")

    # Step 8
    eef_info = None
    try:
        eef_info = compute_swift_eef(
            cfg, obs['evt_hdr'], cfg.src_radius_arcsec, e_lo, e_hi)
        print(f"\n  -- EEF (Encircled Energy Fraction) ----------------------------")
        print(f"    Off-axis angle   : {eef_info['theta_arcmin']:.3f} arcmin")
        print(f"    Pointing         : RA={eef_info['pointing_ra']:.5f}  "
              f"Dec={eef_info['pointing_dec']:.5f}")
        print(f"    PSF file         : {os.path.basename(eef_info['psf_file'])}")
        print(f"    Band-centre E    : {eef_info['energy_kev']:.3f} keV")
        print(f"    EEF at {cfg.src_radius_arcsec:.0f}\"       : {eef_info['eef']:.4f}")
        if eef_info['extrapolated']:
            print(f"    !! Off-axis angle exceeds XRT FOV ({eef_info['theta_arcmin']:.1f}'). "
                  f"EEF capped at 12' = {eef_info['eef_capped']:.4f}")
    except (RuntimeError, FileNotFoundError, KeyError) as exc:
        warnings.warn(
            f"EEF computation skipped: {exc}\n"
            "The bundled psfconst_xrt.fits may be missing. Copy it from "
            "<heasoft>/image/ximage/cal/swift/xrt/ or set psf_file= in config.",
            UserWarning, stacklevel=2)

    return dict(
        N_src=N_src, N_bkg_raw=N_bkg_raw, area_ratio=area_ratio,
        B_scaled=B_scaled, t_eff=t_eff,
        exp_stats=exp_stats, exp_meta=exp_meta,
        cx_evt=cx_out, cy_evt=cy_out, pscale_evt=pscale_out,
        cx_exp=cx_exp, cy_exp=cy_exp,
        eef_info=eef_info,
        mode=mode,
        bkg_cx_evt=bkg_cx_evt if bkg_cx_evt is not None else cx_out,
        bkg_cy_evt=bkg_cy_evt if bkg_cy_evt is not None else cy_out,
    )


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def process_observation(cfg: SwiftConfig):
    """
    Full extraction and upper-limit calculation for one Swift XRT observation,
    or for multiple co-added observations when cfg.obsid is a list.

    For multiple obsids:
    - The interactive region selector (GUI) is shown for the FIRST observation
      only; the chosen aperture sizes and background position are then applied
      to all subsequent observations.
    - Counts and effective exposures are summed across observations.
    - The EEF is averaged weighted by effective exposure time (to account for
      slightly different off-axis angles between pointings).
    - A single combined upper limit is computed at the end.

    Steps per observation
    ---------------------
    1.  Locate event file and exposure map (auto-detect PC / WT mode).
    2.  Load and filter events (PI, grade).
    3.  Load exposure map.
    4.  Convert source RA/Dec to event-file pixel coords.
    5.  (First obs only, optional) Open interactive region selector GUI.
    6.  Extract source and background counts.
    7.  Compute effective exposure from exposure map (or ONTIME fallback).
    8.  Compute EEF from psfconst_xrt.fits.

    Then (once, on accumulated totals):
    9.  Print combined results table.
    10. Save diagnostic plots (per observation).
    11. Write combined CSV.

    Parameters
    ----------
    cfg : SwiftConfig (validated before calling)

    Returns
    -------
    dict with keys:
        obsids, mode, N_src, N_bkg_raw, B_scaled, area_ratio,
        net_counts, t_eff_s, exp_stats, ul, energy, eef_info, csv_rows
    """
    e_lo, e_hi = cfg.resolve_energy_band()
    out_dir    = os.path.join(cfg.data_dir, "ul_products")
    os.makedirs(out_dir, exist_ok=True)

    obsids = cfg.obsids
    n_obs  = len(obsids)

    # Save the original aperture/background settings so we can restore them
    # for each observation in per_obs GUI mode.
    _orig = {k: getattr(cfg, k) for k in (
        'src_radius_arcsec', 'bkg_radius_arcsec', 'bkg_inner_factor',
        'bkg_mode', 'bkg_ra', 'bkg_dec')}

    # ------------------------------------------------------------------ #
    # Loop over observations                                               #
    # ------------------------------------------------------------------ #
    per_obs = []
    bkg_cx_first = bkg_cy_first = None   # pixel coords from GUI (first obs, shared mode)

    for i, obsid_str in enumerate(obsids):
        obs_root = os.path.join(cfg.data_dir, obsid_str)

        print(f"\n{'='*70}")
        if n_obs > 1:
            print(f"  Swift XRT  —  Observation {i+1}/{n_obs}  ({obsid_str})")
        else:
            print(f"  Swift XRT")
        print(f"{'='*70}")

        obs = _load_one_obs(cfg, obs_root, obsid_str)

        # ── GUI logic ────────────────────────────────────────────────── #
        if cfg.use_gui and cfg.gui_per_obs:
            # Independent GUI for every observation.
            # Work on a per-obs copy so we don't pollute the shared cfg.
            cfg_obs = copy.copy(cfg)
            for k, v in _orig.items():
                setattr(cfg_obs, k, v)   # reset to original settings each time
            bkg_cx_evt, bkg_cy_evt = _run_gui_first_obs(cfg_obs, obs)

        elif cfg.use_gui and i == 0:
            # Shared mode: GUI only for the first observation.
            # Mutates cfg so that subsequent obs inherit aperture + bkg sky coords.
            cfg_obs = cfg
            bkg_cx_evt, bkg_cy_evt = _run_gui_first_obs(cfg_obs, obs)
            bkg_cx_first = bkg_cx_evt
            bkg_cy_first = bkg_cy_evt

        elif cfg.use_gui and i > 0:
            # Shared mode, subsequent obs: cfg already has bkg_ra/bkg_dec set
            # from the first-obs GUI.  Pass None for pixel coords so that
            # extract_src_bkg_counts re-projects the sky position into this
            # observation's pixel frame.
            cfg_obs = cfg
            bkg_cx_evt = None
            bkg_cy_evt = None

        else:
            # No GUI: use cfg as-is for all observations.
            cfg_obs = cfg
            bkg_cx_evt = None
            bkg_cy_evt = None

        # Print aperture summary after possible GUI
        print(f"\n  Src aperture : {cfg_obs.src_radius_arcsec:.1f}\"")
        if cfg_obs.bkg_mode == 'annulus':
            r_in = cfg_obs.src_radius_arcsec * cfg_obs.bkg_inner_factor
            print(f"  Bkg annulus  : {r_in:.1f}\" — {cfg_obs.bkg_radius_arcsec:.1f}\"")
        else:
            print(f"  Bkg circle   : r={cfg_obs.bkg_radius_arcsec:.1f}\"  "
                  f"(manual centre  RA={cfg_obs.bkg_ra}  Dec={cfg_obs.bkg_dec})")

        print()
        raw = _extract_counts_exposure_eef(
            cfg_obs, obs, bkg_cx_evt, bkg_cy_evt, e_lo, e_hi)
        raw['obs_root']   = obs_root
        raw['obsid_str']  = obsid_str
        raw['date_obs']   = obs.get('date_obs', '')
        raw['evt_x']      = obs['evt_x']
        raw['evt_y']      = obs['evt_y']
        raw['src_coord']  = obs['src_coord']
        raw['cfg_obs']    = cfg_obs
        # Pixel coords for plots: use GUI result for this obs (or best available)
        raw['bkg_cx_for_plot'] = (bkg_cx_evt if bkg_cx_evt is not None
                                   else raw['bkg_cx_evt'])
        raw['bkg_cy_for_plot'] = (bkg_cy_evt if bkg_cy_evt is not None
                                   else raw['bkg_cy_evt'])
        per_obs.append(raw)

    # ------------------------------------------------------------------ #
    # Accumulate across observations                                        #
    # ------------------------------------------------------------------ #
    N_src_total     = sum(r['N_src']     for r in per_obs)
    N_bkg_raw_total = sum(r['N_bkg_raw'] for r in per_obs)
    T_eff_total     = sum(r['t_eff']     for r in per_obs)
    area_ratio      = per_obs[0]['area_ratio']   # geometric — same for all
    B_scaled_total  = N_bkg_raw_total * area_ratio

    # Exposure-weighted EEF average
    eef_infos = [r['eef_info'] for r in per_obs]
    if all(e is not None for e in eef_infos):
        eef_avg = (sum(r['t_eff'] * r['eef_info']['eef'] for r in per_obs)
                   / T_eff_total)
        eef_info = dict(per_obs[0]['eef_info'])
        eef_info['eef'] = eef_avg
        if n_obs > 1:
            print(f"\n  EEF averaged across {n_obs} observations "
                  f"(exposure-weighted): {eef_avg:.4f}")
    else:
        eef_info = None

    # Composite exp_stats (sum)
    exp_stats_total = {
        'median'      : sum(r['exp_stats']['median']       for r in per_obs),
        'mean'        : sum(r['exp_stats']['mean']         for r in per_obs),
        'psf_weighted': sum(r['exp_stats']['psf_weighted'] for r in per_obs),
    }

    mode = per_obs[0]['mode']

    # ------------------------------------------------------------------ #
    # Step 9a: per-obs UL (computed quietly; printed as brief summary)    #
    # ------------------------------------------------------------------ #
    if n_obs > 1:
        print(f"\n{'='*70}")
        print(f"  Per-observation summary  ({n_obs} observations)")
        print(f"{'='*70}")
        ul3_col = 'CR_kraft_aperture'
        print(f"  {'Obs ID':>14}  {'N_src':>6}  {'B_scaled':>9}  "
              f"{'t_eff (ks)':>11}  {'EEF':>6}  {'Kraft CR_ap (3σ)':>18}")
        print("  " + "-" * 76)

    for r in per_obs:
        eef_val_i = r['eef_info']['eef'] if r['eef_info'] is not None else None
        r['ul'] = _compute_ul_results(
            r['N_src'], r['B_scaled'], r['t_eff'],
            r['N_bkg_raw'], r['area_ratio'],
            cfg.confidence_levels, eef=eef_val_i)
        r['csv_rows'] = _build_csv_rows(
            r['mode'], e_lo, e_hi,
            r['N_src'], r['N_bkg_raw'], r['B_scaled'],
            r['area_ratio'], r['t_eff'],
            r['ul'], r['eef_info'], r['obsid_str'],
            result_type='individual',
            date_obs=r.get('date_obs', ''))

        if n_obs > 1:
            ul3_i = next((u for u in r['ul'] if u['cl'] >= 0.997), r['ul'][-1])
            eef_str_i = (f"{r['eef_info']['eef']:.3f}"
                         if r['eef_info'] is not None else "  N/A")
            print(f"  {r['obsid_str']:>14}  {r['N_src']:>6}  "
                  f"{r['B_scaled']:>9.2f}  {r['t_eff']/1e3:>11.3f}  "
                  f"{eef_str_i:>6}  "
                  f"{ul3_i['CR_kraft_aperture']:>18.4e}")

    # ------------------------------------------------------------------ #
    # Step 9b: results table on combined counts                            #
    # ------------------------------------------------------------------ #
    if n_obs > 1:
        print(f"\n{'='*70}")
        print(f"  Combined totals across {n_obs} observations")
        print(f"{'='*70}")
        print(f"  N_src total      : {N_src_total}")
        print(f"  N_bkg_raw total  : {N_bkg_raw_total}")
        print(f"  B_scaled total   : {B_scaled_total:.3f} cts")
        print(f"  T_eff total      : {T_eff_total/1e3:.3f} ks")

    eef_val    = eef_info['eef'] if eef_info is not None else None
    ul_results = _print_results_table(
        N_src_total, B_scaled_total, T_eff_total,
        N_bkg_raw_total, area_ratio,
        cfg.confidence_levels, eef=eef_val)

    # ------------------------------------------------------------------ #
    # Step 10: diagnostic plots  (one set per observation)                  #
    # ------------------------------------------------------------------ #
    if cfg.save_plots:
        for r in per_obs:
            _save_plots(
                r['evt_x'], r['evt_y'], r['cx_evt'], r['cy_evt'], r['pscale_evt'],
                r['exp_meta'], r['exp_stats'],
                f"XRT-{r['mode']}", e_lo, e_hi, r['cfg_obs'], out_dir,
                r['src_coord'],
                bkg_cx_evt=r['bkg_cx_for_plot'],
                bkg_cy_evt=r['bkg_cy_for_plot'],
                obsid_str=r['obsid_str'])

    # ------------------------------------------------------------------ #
    # Step 11: CSV — per-obs rows first, then combined row                  #
    # ------------------------------------------------------------------ #
    obsid_label = (obsids[0] if n_obs == 1 else "+".join(obsids))
    # For combined date_obs use the first observation's date
    combined_date = per_obs[0].get('date_obs', '') if per_obs else ''
    combined_csv_rows = _build_csv_rows(
        mode, e_lo, e_hi,
        N_src_total, N_bkg_raw_total, B_scaled_total,
        area_ratio, T_eff_total, ul_results, eef_info, obsid_label,
        result_type='combined', date_obs=combined_date)

    all_csv_rows = []
    if n_obs > 1:
        for r in per_obs:
            all_csv_rows.extend(r['csv_rows'])
    all_csv_rows.extend(combined_csv_rows)

    return {
        'obsids'     : obsids,
        'mode'       : mode,
        'N_src'      : N_src_total,
        'N_bkg_raw'  : N_bkg_raw_total,
        'B_scaled'   : B_scaled_total,
        'area_ratio' : area_ratio,
        'net_counts' : N_src_total - B_scaled_total,
        't_eff_s'    : T_eff_total,
        'exp_stats'  : exp_stats_total,
        'ul'         : ul_results,
        'energy'     : (e_lo, e_hi),
        'eef_info'   : eef_info,
        'csv_rows'   : all_csv_rows,
        'per_obs'    : per_obs,
    }


# =============================================================================
# DIAGNOSTIC PLOTS
# =============================================================================

def _save_plots(evt_x, evt_y, cx_evt, cy_evt, pscale_evt,
                exp_meta, exp_stats, label, e_lo, e_hi, cfg, out_dir,
                src_coord, bkg_cx_evt, bkg_cy_evt, obsid_str=None):
    try:
        from ..plots import radial_profile, exposure_histogram, region_image
    except ImportError:
        warnings.warn("Diagnostic plots skipped (import failed).",
                      RuntimeWarning, stacklevel=2)
        return

    # Use the per-observation obsid for file naming (avoids collisions when
    # multiple obsids are co-added)
    plot_obsid = obsid_str if obsid_str is not None else cfg.obsids[0]

    radial_profile(
        evt_x, evt_y, cx_evt, cy_evt, pscale_evt,
        label, e_lo, e_hi, plot_obsid, cfg, out_dir)

    if exp_meta is not None:
        exposure_histogram(exp_meta, exp_stats, label, cfg, out_dir)

    region_image(
        evt_x, evt_y, cx_evt, cy_evt, pscale_evt,
        label, e_lo, e_hi, plot_obsid, cfg, out_dir,
        src_ra_deg  = src_coord.ra.deg,
        src_dec_deg = src_coord.dec.deg,
        bkg_cx_evt  = bkg_cx_evt,
        bkg_cy_evt  = bkg_cy_evt)


# =============================================================================
# ENTRY POINT
# =============================================================================

def run_uplim(data_dir, obsid, ra, dec, **kwargs):
    """
    Run the full Swift XRT upper-limit pipeline.

    Parameters
    ----------
    data_dir : str or path
        Parent directory that contains one sub-directory per observation,
        named by obsid (e.g. '/path/to/swift/data/' which contains
        '03000397004/', '03000397005/', …).
        Each obsid sub-directory must contain the standard Swift archive
        layout (xrt/event/ and xrt/products/).

    obsid : str or list of str
        One observation ID, or a list of IDs to co-add.
        When a list is supplied, counts and exposures from all observations
        are summed before computing the upper limit — equivalent to a
        stacked analysis with no time-resolved information.
        Example: obsid='03000397004'
                 obsid=['03000397001', '03000397002', '03000397004']

    ra  : str or float  — source RA  ("HH:MM:SS" or decimal degrees)
    dec : str or float  — source Dec ("±DD:MM:SS" or decimal degrees)
    **kwargs : any SwiftConfig field, e.g.
                   energy_band='soft',
                   src_radius_arcsec=20.0,
                   confidence_levels=[0.9973],
                   caldb_dir='/path/to/caldb',
                   save_plots=True

    Returns
    -------
    dict — result from process_observation()
    """
    cfg = SwiftConfig(data_dir=data_dir, obsid=obsid, ra=ra, dec=dec, **kwargs)
    cfg.validate()

    e_lo, e_hi = cfg.resolve_energy_band()
    src_coord  = parse_coord(cfg.ra, cfg.dec)
    out_dir    = os.path.join(cfg.data_dir, "ul_products")
    obsids     = cfg.obsids

    print("Swift XRT Non-Detection Upper Limit")
    print("=" * 70)
    print(f"Source      :  RA = {src_coord.ra.deg:.6f} deg  "
          f"Dec = {src_coord.dec.deg:.6f} deg")
    if isinstance(cfg.energy_band, tuple):
        band_label = f"{e_lo:.2f}–{e_hi:.2f} keV (custom)"
    else:
        band_label = f"'{cfg.energy_band}'  ({e_lo:.2f}–{e_hi:.2f} keV)"
    print(f"Energy band :  {band_label}")
    print(f"Exp stat    :  {cfg.exp_stat}  (primary)")
    print(f"Bkg mode    :  {cfg.bkg_mode}")
    print(f"Data dir    :  {cfg.data_dir}")
    if len(obsids) == 1:
        print(f"Obs ID      :  {obsids[0]}")
    else:
        print(f"Obs IDs     :  {len(obsids)} observations  "
              f"[{obsids[0]} … {obsids[-1]}]")
        for oid in obsids:
            print(f"               {oid}")
    if cfg.psf_file:
        print(f"PSF file    :  {cfg.psf_file}  (user override)")
    print()

    result = process_observation(cfg)

    # -- Summary --------------------------------------------------------------
    print(f"\n{'='*70}")
    print("  SUMMARY")
    print(f"{'='*70}")
    eef_str = (f"{result['eef_info']['eef']:.3f}"
               if result['eef_info'] is not None else "  N/A")
    ul_row = next((u for u in result['ul'] if u['cl'] >= 0.997),
                  result['ul'][-1])

    n_obs = len(obsids)
    obs_label = obsids[0] if n_obs == 1 else f"{n_obs} obs co-added"

    print(f"  {'Obs':>20}  {'Mode':<5}  {'N_src':>6}  {'B_scaled':>9}  "
          f"{'t_eff (ks)':>11}  {'EEF':>6}  "
          f"{'Kraft CR_ap (3σ)':>18}")
    print("  " + "-" * 84)
    print(f"  {obs_label:>20}  {result['mode']:<5}  {result['N_src']:>6}  "
          f"{result['B_scaled']:>9.2f}  "
          f"{result['t_eff_s']/1e3:>11.3f}  "
          f"{eef_str:>6}  "
          f"{ul_row['CR_kraft_aperture']:>18.4e}")
    print()

    # -- Write CSV ------------------------------------------------------------
    obsid_label = obsids[0] if n_obs == 1 else "+".join(obsids)
    write_results_csv(result['csv_rows'], out_dir, obsid_label)

    return result
