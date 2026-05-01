"""
xray_uplim.nustar.pipeline
--------------------------
Top-level orchestration: per-module extraction and combined results.

Public API
----------
run_uplim(...)        — convenience wrapper; builds a Config and runs everything
process_module(...)   — extract counts + exposure for one FPM, return results dict
combine_modules(...)  — sum across FPMs and print combined upper limits
"""

import csv
import os
import warnings
import numpy as np

from .config      import Config
from .io          import locate_files, load_events, load_expmap
from ..coords     import parse_coord, sky_to_evt_pixel, sky_to_img_pixel
from ..eef        import compute_eef
from ..exposure   import compute_exposure_stats
from ..statistics import net_count_rate, kraft_upper_limit, gehrels_upper_limit
from ..plots      import radial_profile, exposure_histogram, region_image


# =============================================================================
# RESULTS TABLE  (shared by process_module and combine_modules)
# =============================================================================

def print_results_table(N_src, B_scaled, t_eff, N_bkg_raw, area_ratio,
                        confidence_levels, eef=None):
    """
    Compute and print all three methods at every confidence level.

    Aperture columns (always printed)
    ----------------------------------
    CL              — one-sided confidence level
    Net CR          — (N_src - B_scaled) / t_eff   [point estimate, not a UL]
    Kraft S_ul      — Kraft+91 upper limit in counts
    Kraft CR_ap     — Kraft+91 aperture count-rate upper limit (cts/s)
    Gehrels S_ul / CR_ap — Gehrels 1986 cross-check (aperture)

    EEF-corrected columns (printed only when eef is not None)
    ----------------------------------------------------------
    Kraft CR_tot    — EEF-corrected total source rate  = S_ul / (t_eff * EEF)
    Gehrels CR_tot  — same for Gehrels

    Parameters
    ----------
    eef : float or None
        Encircled energy fraction at the source aperture radius.  If None,
        EEF-corrected columns are omitted.

    Returns
    -------
    list of dicts, one per confidence level, with keys:
        cl, CR_net, CR_sigma,
        S_kraft,   CR_kraft_aperture,   CR_kraft_total (None if no EEF),
        S_gehrels, CR_gehrels_aperture, CR_gehrels_total (None if no EEF)
    """
    CR_net, CR_sigma = net_count_rate(N_src, B_scaled, t_eff,
                                      N_bkg_raw, area_ratio)

    print(f"\n  Point estimate  (N_src - B) / t_eff  [NOT an upper limit]")
    print(f"    = ({N_src} - {B_scaled:.1f}) / {t_eff:.1f} s")
    print(f"    = {CR_net:+.4e} cts/s  ±  {CR_sigma:.4e}  (1-sigma Poisson)")
    if CR_net < 0:
        print(f"    (Negative — source aperture below expected background: "
              f"clean non-detection)")

    # -- Header ---------------------------------------------------------------
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

    results = []
    for cl in confidence_levels:
        S_k  = kraft_upper_limit(N_src, B_scaled, cl)
        S_g  = gehrels_upper_limit(N_src, B_scaled, cl)
        CR_k_ap = S_k / t_eff
        CR_g_ap = S_g / t_eff

        if eef is not None and eef > 0:
            CR_k_tot = S_k / (t_eff * eef)
            CR_g_tot = S_g / (t_eff * eef)
        else:
            CR_k_tot = None
            CR_g_tot = None

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

        if eef is not None:
            print(
                f"  {cl:8.4f}  {CR_net:+13.4e}  "
                f"{S_k:10.3f}  {CR_k_ap:13.4e}  {CR_k_tot:13.4e}  "
                f"{S_g:10.3f}  {CR_g_ap:13.4e}  {CR_g_tot:13.4e}"
            )
        else:
            print(
                f"  {cl:8.4f}  {CR_net:+13.4e}  "
                f"{S_k:10.3f}  {CR_k_ap:13.4e}  "
                f"{S_g:10.3f}  {CR_g_ap:13.4e}"
            )

    print(divider)
    if eef is not None:
        print(f"  CR_ap  = aperture count-rate upper limit = S_ul / t_eff.")
        print(f"  CR_tot = EEF-corrected total source rate = S_ul / (t_eff * EEF).")
        print(f"  EEF used: {eef:.4f}")
    else:
        print(f"  CR_ap is the aperture count-rate upper limit.")
        print(f"  EEF correction skipped (set caldb_dir to enable).")

    return results


# =============================================================================
# CSV OUTPUT
# =============================================================================

def write_results_csv(rows, out_dir, obsid):
    """
    Write upper-limit results to a CSV file.

    One row per (module, confidence-level) combination, plus combined rows
    (module='AB') when both FPMs are processed.

    Parameters
    ----------
    rows    : list of dicts — each dict is one row; see _build_csv_rows()
    out_dir : str           — output directory (created if absent)
    obsid   : str

    Returns
    -------
    csv_path : str — absolute path of the written file
    """
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, f"nustar_uplim_{obsid}.csv")

    fieldnames = [
        'obsid', 'date_obs', 'module', 'energy_lo_kev', 'energy_hi_kev',
        'N_src', 'N_bkg_raw', 'B_scaled', 'area_ratio',
        't_eff_s',
        'theta_arcmin', 'eef', 'psf_file', 'eef_extrapolated',
        'eef_capped', 'eef_extrap',
        'confidence_level',
        'CR_net', 'CR_sigma',
        'S_kraft', 'CR_kraft_aperture', 'CR_kraft_total',
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


def _build_csv_rows(module, e_lo, e_hi, N_src, N_bkg_raw, B_scaled,
                    area_ratio, t_eff, ul_results, eef_info, obsid,
                    date_obs=''):
    """
    Build a list of CSV row dicts (one per confidence level) for one module.

    Parameters
    ----------
    eef_info : dict or None — return value of compute_eef(), or None if skipped
    date_obs : str          — DATE-OBS from event file header (ISO 8601)
    """
    rows = []
    for r in ul_results:
        row = {
            'obsid':              obsid,
            'date_obs':           date_obs,
            'module':             module,
            'energy_lo_kev':      e_lo,
            'energy_hi_kev':      e_hi,
            'N_src':              N_src,
            'N_bkg_raw':          N_bkg_raw,
            'B_scaled':           f"{B_scaled:.4f}",
            'area_ratio':         f"{area_ratio:.6f}",
            't_eff_s':            f"{t_eff:.2f}",
            'confidence_level':   r['cl'],
            'CR_net':             f"{r['CR_net']:.6e}",
            'CR_sigma':           f"{r['CR_sigma']:.6e}",
            'S_kraft':            f"{r['S_kraft']:.4f}",
            'CR_kraft_aperture':  f"{r['CR_kraft_aperture']:.6e}",
            'S_gehrels':          f"{r['S_gehrels']:.4f}",
            'CR_gehrels_aperture': f"{r['CR_gehrels_aperture']:.6e}",
        }
        if eef_info is not None:
            row['theta_arcmin']    = (f"{eef_info['theta_arcmin']:.4f}"
                                      if eef_info['theta_arcmin'] is not None else '')
            row['eef']             = f"{eef_info['eef']:.6f}"
            row['psf_file']        = '; '.join(
                                         os.path.basename(f)
                                         for f in eef_info['psf_files'])
            row['eef_extrapolated'] = eef_info['extrapolated']
            row['eef_capped']      = (f"{eef_info['eef_capped']:.6f}"
                                      if eef_info['eef_capped'] is not None else '')
            row['eef_extrap']      = (f"{eef_info['eef_extrap']:.6f}"
                                      if eef_info['eef_extrap'] is not None else '')
            if r['CR_kraft_total'] is not None:
                row['CR_kraft_total']   = f"{r['CR_kraft_total']:.6e}"
                row['CR_gehrels_total'] = f"{r['CR_gehrels_total']:.6e}"
        rows.append(row)
    return rows


# =============================================================================
# PER-MODULE PIPELINE
# =============================================================================

def process_module(module, src_coord, cfg):
    """
    Full extraction and result calculation for one FPM.

    Parameters
    ----------
    module    : str              — 'A' or 'B'
    src_coord : SkyCoord         — source sky position
    cfg       : Config

    Returns
    -------
    dict with keys:
        module, N_src, N_bkg_raw, B_scaled, area_ratio, net_counts,
        t_eff_s, exp_stats, ul, energy, eef_info, csv_rows
    """
    e_lo, e_hi = cfg.resolve_energy_band()
    out_dir    = os.path.join(cfg.base_path, cfg.obsid, "ul_products")
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  FPM{module}")
    print(f"{'='*70}")

    # -- Locate and load files ------------------------------------------------
    evt_file, exp_file = locate_files(cfg.base_path, cfg.obsid, module)
    print(f"  Event file  : {os.path.basename(evt_file)}")
    print(f"  Expo map    : {os.path.basename(exp_file)}")

    evts, evt_hdr, PI_lo, PI_hi = load_events(evt_file, e_lo, e_hi)
    date_obs = str(evt_hdr.get('DATE-OBS', '')).strip()
    print(f"  Energy filter [{e_lo:.1f}-{e_hi:.1f} keV]  "
          f"PI=[{PI_lo},{PI_hi}]  ->  {len(evts):,} events")

    exp_data, exp_hdr = load_expmap(exp_file)

    # -- Source pixel positions -----------------------------------------------
    cx_evt, cy_evt, pscale_evt = sky_to_evt_pixel(
        src_coord.ra.deg, src_coord.dec.deg, evt_hdr)
    cx_exp, cy_exp, pscale_exp = sky_to_img_pixel(
        src_coord.ra.deg, src_coord.dec.deg, exp_hdr)

    # -- Sanity check ---------------------------------------------------------
    evt_x = evts['X'].astype(float)
    evt_y = evts['Y'].astype(float)
    print(f"\n  Event X range       : [{evt_x.min():.0f}, {evt_x.max():.0f}]")
    print(f"  Event Y range       : [{evt_y.min():.0f}, {evt_y.max():.0f}]")
    print(f"  Source pixel (evt)  : ({cx_evt:.1f}, {cy_evt:.1f})")
    print(f"  Source pixel (exp)  : ({cx_exp:.1f}, {cy_exp:.1f})")
    x_ok = evt_x.min() <= cx_evt <= evt_x.max()
    y_ok = evt_y.min() <= cy_evt <= evt_y.max()
    if not (x_ok and y_ok):
        print(f"  !! WARNING: source pixel is OUTSIDE the event X/Y range — "
              f"check your coordinates!")
    else:
        print(f"  Source position is inside the event image. Good.")

    # -- Interactive region selector ------------------------------------------
    # bkg_cx_evt / bkg_cy_evt track the background centre in event pixel space.
    # They equal the source centre in annulus mode, or differ in manual mode.
    bkg_cx_evt = cx_evt
    bkg_cy_evt = cy_evt

    if cfg.use_gui:
        from ..region_selector import select_regions_interactive
        print(f"\n  Opening interactive region selector for FPM{module}...")
        sel = select_regions_interactive(
            evt_x, evt_y, cx_evt, cy_evt, pscale_evt, cfg, f'FPM{module}')

        cx_evt     = sel['cx']
        cy_evt     = sel['cy']
        bkg_cx_evt = sel['bkg_cx']
        bkg_cy_evt = sel['bkg_cy']

        # Write confirmed radii back to cfg so plots/EEF see updated values
        cfg.src_radius_arcsec = sel['src_radius_arcsec']
        cfg.bkg_radius_arcsec = sel['bkg_radius_arcsec']
        cfg.bkg_inner_factor  = sel['bkg_inner_factor']

        # If user moved background to a different location, switch to manual mode
        bkg_moved = (abs(bkg_cx_evt - cx_evt) > 1.0 or
                     abs(bkg_cy_evt - cy_evt) > 1.0)
        if bkg_moved:
            try:
                from astropy.wcs import WCS as _WCS
                _wcs = _WCS(evt_hdr, naxis=2)
                # pixel_to_world_values always returns plain floats (avoids
                # the 'list has no attribute ra' issue with multi-axis WCS)
                _bkg_ra, _bkg_dec = _wcs.pixel_to_world_values(
                    bkg_cx_evt - 1, bkg_cy_evt - 1)
                # Only commit to manual mode after all conversions succeed
                cfg.bkg_mode = 'manual'
                cfg.bkg_ra   = str(float(_bkg_ra))
                cfg.bkg_dec  = str(float(_bkg_dec))
                print(f"  [GUI] Background → manual mode: "
                      f"RA={float(_bkg_ra):.5f}  Dec={float(_bkg_dec):.5f}")
            except Exception as _e:
                warnings.warn(
                    f"Could not convert background pixel to RA/Dec ({_e}). "
                    "Falling back to annulus mode.")
                # Make sure bkg_mode is NOT left as 'manual'
                cfg.bkg_mode = 'annulus'
                bkg_cx_evt   = cx_evt
                bkg_cy_evt   = cy_evt

    src_radius_arcsec = cfg.src_radius_arcsec
    bkg_radius_arcsec = cfg.bkg_radius_arcsec
    bkg_inner_factor  = cfg.bkg_inner_factor

    # -- Pixel radii ----------------------------------------------------------
    r_src_evt        = src_radius_arcsec / pscale_evt
    r_src_exp        = src_radius_arcsec / pscale_exp
    fwhm_pix         = cfg.psf_fwhm_arcsec / pscale_exp
    r_bkg_in_arcsec  = src_radius_arcsec * bkg_inner_factor
    r_bkg_out_arcsec = bkg_radius_arcsec
    r_bkg_in_evt     = r_bkg_in_arcsec  / pscale_evt
    r_bkg_out_evt    = r_bkg_out_arcsec / pscale_evt

    print(f"  Pixel scale (evt)   : {pscale_evt:.3f} \"/pix")
    print(f"  Pixel scale (exp)   : {pscale_exp:.3f} \"/pix")
    print(f"  Src aperture        : {src_radius_arcsec:.1f}\" = {r_src_evt:.1f} pix")
    print(f"  Bkg annulus         : {r_bkg_in_arcsec:.1f}\" -- {r_bkg_out_arcsec:.1f}\"")

    # -- Source counts --------------------------------------------------------
    d_src = np.sqrt((evt_x - cx_evt)**2 + (evt_y - cy_evt)**2)
    N_src = int(np.sum(d_src <= r_src_evt))

    # -- Background counts ----------------------------------------------------
    if cfg.bkg_mode == 'annulus':
        in_annulus = (d_src > r_bkg_in_evt) & (d_src <= r_bkg_out_evt)
        N_bkg_raw  = int(np.sum(in_annulus))
        area_src   = np.pi * r_src_evt**2
        area_bkg   = np.pi * (r_bkg_out_evt**2 - r_bkg_in_evt**2)

    elif cfg.bkg_mode == 'manual':
        bkg_coord      = parse_coord(cfg.bkg_ra, cfg.bkg_dec)
        cx_b, cy_b, _  = sky_to_evt_pixel(
            bkg_coord.ra.deg, bkg_coord.dec.deg, evt_hdr)
        r_bkg_circ     = bkg_radius_arcsec / pscale_evt
        d_bkg          = np.sqrt((evt_x - cx_b)**2 + (evt_y - cy_b)**2)
        N_bkg_raw      = int(np.sum(d_bkg <= r_bkg_circ))
        area_src       = np.pi * r_src_evt**2
        area_bkg       = np.pi * r_bkg_circ**2
    else:
        raise ValueError(f"Unknown bkg_mode: '{cfg.bkg_mode}'")

    area_ratio = area_src / area_bkg
    B_scaled   = N_bkg_raw * area_ratio

    print(f"\n  Source counts  (N_src)        : {N_src}")
    print(f"  Background counts (raw)       : {N_bkg_raw}")
    print(f"  Area ratio  (src / bkg)       : {area_ratio:.5f}")
    print(f"  Scaled background B           : {B_scaled:.3f} cts")
    print(f"  Net counts  (N_src - B)       : {N_src - B_scaled:.3f} cts")

    # -- Effective exposure ---------------------------------------------------
    exp_stats, exp_meta = compute_exposure_stats(
        exp_data, cx_exp, cy_exp, r_src_exp, fwhm_pix)

    print(f"\n  Pixels in aperture (total)    : {exp_meta['n_pix_total']}")
    print(f"  Pixels in aperture (non-zero) : {exp_meta['n_pix_nonzero']}")
    print(f"  -- Exposure statistics ----------------------------------------")
    for key, label in [
            ('median',       'Median        [RECOMMENDED]          '),
            ('mean',         'Mean          [diagnostic]           '),
            ('psf_weighted', 'PSF-wtd mean  [on-axis diag. only]   ')]:
        tag = ' <-- PRIMARY' if key == cfg.exp_stat else ''
        print(f"    {label} : {exp_stats[key]/1e3:7.3f} ks{tag}")

    t_eff = exp_stats[cfg.exp_stat]
    print(f"\n  Using t_eff = {t_eff/1e3:.3f} ks  ({cfg.exp_stat})")

    # -- EEF from CALDB PSF ---------------------------------------------------
    eef_info = None
    caldb_dir = cfg.caldb_dir if cfg.caldb_dir else None

    try:
        eef_info = compute_eef(
            src_coord.ra.deg, src_coord.dec.deg, evt_hdr,
            src_radius_arcsec, e_lo, e_hi,
            module, caldb_dir=caldb_dir, gamma=cfg.psf_gamma)

        print(f"\n  -- EEF (Encircled Energy Fraction) ----------------------------")
        print(f"    Off-axis angle   : {eef_info['theta_arcmin']:.3f} arcmin")
        print(f"    Pointing used    : RA={eef_info['pointing_ra']:.5f}  "
              f"Dec={eef_info['pointing_dec']:.5f}")
        print(f"    PSF file(s)      : "
              f"{', '.join(os.path.basename(f) for f in eef_info['psf_files'])}")
        print(f"    PSF pixel scale  : {eef_info['pix_scale_arcsec']:.4f} \"/pix")
        print(f"    EEF at {src_radius_arcsec:.0f}\"        : {eef_info['eef']:.4f}")
        if eef_info['extrapolated']:
            print(f"    !! Off-axis angle exceeds CalDB limit (7'). "
                  f"EEF (capped at 7') = {eef_info['eef_capped']:.4f}  "
                  f"EEF (extrapolated) = {eef_info['eef_extrap']:.4f}")

    except (RuntimeError, FileNotFoundError) as exc:
        warnings.warn(
            f"EEF computation skipped for FPM{module}: {exc}\n"
            "Set caldb_dir= (or $CALDB) to enable EEF-corrected upper limits.",
            UserWarning, stacklevel=2)

    # -- Results table --------------------------------------------------------
    eef_val = eef_info['eef'] if eef_info is not None else None
    ul_results = print_results_table(
        N_src, B_scaled, t_eff, N_bkg_raw, area_ratio,
        cfg.confidence_levels, eef=eef_val)

    # -- CSV rows -------------------------------------------------------------
    csv_rows = _build_csv_rows(
        module, e_lo, e_hi, N_src, N_bkg_raw, B_scaled,
        area_ratio, t_eff, ul_results, eef_info, cfg.obsid,
        date_obs=date_obs)

    # -- Diagnostic plots -----------------------------------------------------
    if cfg.save_plots:
        radial_profile(evt_x, evt_y, cx_evt, cy_evt, pscale_evt,
                       f'FPM{module}', e_lo, e_hi, cfg.obsid, cfg, out_dir)
        exposure_histogram(exp_meta, exp_stats, f'FPM{module}', cfg, out_dir)
        region_image(evt_x, evt_y, cx_evt, cy_evt, pscale_evt,
                     f'FPM{module}', e_lo, e_hi, cfg.obsid, cfg, out_dir,
                     src_ra_deg=src_coord.ra.deg,
                     src_dec_deg=src_coord.dec.deg,
                     bkg_cx_evt=bkg_cx_evt,
                     bkg_cy_evt=bkg_cy_evt)

    return {
        'module':     module,
        'date_obs':   date_obs,
        'N_src':      N_src,
        'N_bkg_raw':  N_bkg_raw,
        'B_scaled':   B_scaled,
        'area_ratio': area_ratio,
        'net_counts': N_src - B_scaled,
        't_eff_s':    t_eff,
        'exp_stats':  exp_stats,
        'ul':         ul_results,
        'energy':     (e_lo, e_hi),
        'eef_info':   eef_info,
        'csv_rows':   csv_rows,
    }


# =============================================================================
# COMBINED (FPM-A + FPM-B)
# =============================================================================

def combine_modules(results_list, cfg):
    """
    Sum counts across FPMs and compute combined results.

    Combining strategy
    ------------------
    N_total  = sum(N_src)       counts are additive across independent detectors
    B_total  = sum(B_scaled)    each B already corrected to source aperture area
    t_comb   = sum(t_eff)       exposures add — correct for additive counts

    For the EEF-corrected combined rate, the correct denominator is:

        sum_i( t_eff_i * EEF_i )

    because each FPM contributes counts collected over its own exposure and
    through its own aperture EEF.

    Parameters
    ----------
    results_list : list of dicts returned by process_module()
    cfg          : Config

    Returns
    -------
    list of combined CSV row dicts
    """
    print(f"\n{'='*70}")
    print("  COMBINED  FPM-A + FPM-B")
    print(f"{'='*70}")

    N_total     = sum(r['N_src']     for r in results_list)
    B_total     = sum(r['B_scaled']  for r in results_list)
    N_bkg_total = sum(r['N_bkg_raw'] for r in results_list)
    area_ratio  = results_list[0]['area_ratio']   # same aperture geometry
    t_vals      = [r['t_eff_s'] for r in results_list]
    t_comb      = float(np.sum(t_vals))           # SUM, not mean

    print(f"  Combined N_src    : {N_total}")
    print(f"  Combined B_scaled : {B_total:.3f} cts")
    for r in results_list:
        print(f"  t_eff FPM-{r['module']}       : {r['t_eff_s']/1e3:.3f} ks")
    print(f"  t_eff (combined)  : {t_comb/1e3:.3f} ks  "
          f"[sum — correct for additive counts]")

    # -- Combined EEF denominator ---------------------------------------------
    # EEF-weighted combined exposure: sum(t_i * EEF_i)
    eef_combined_info = None
    all_have_eef = all(r['eef_info'] is not None for r in results_list)

    if all_have_eef:
        t_eef_sum = sum(
            r['t_eff_s'] * r['eef_info']['eef'] for r in results_list)
        # Store a synthetic eef_info for the combined result so CSV and
        # print_results_table can use it.  We fabricate a single effective EEF
        # as t_eef_sum / t_comb so that S_ul / (t_comb * eef_eff) equals the
        # correct expression S_ul / t_eef_sum.
        eef_eff = t_eef_sum / t_comb if t_comb > 0 else None

        # Report individual EEFs
        print(f"\n  -- Combined EEF -------------------------------------------")
        for r in results_list:
            ei = r['eef_info']
            print(f"    FPM-{r['module']}: theta = {ei['theta_arcmin']:.3f}'  "
                  f"EEF = {ei['eef']:.4f}  "
                  f"t_eff * EEF = {r['t_eff_s'] * ei['eef'] / 1e3:.3f} ks")
        print(f"    Combined t_eff * EEF denominator : {t_eef_sum/1e3:.3f} ks")
        print(f"    Effective EEF (= sum(t*EEF)/t_comb) : {eef_eff:.4f}")

        # Build a minimal eef_info dict for _build_csv_rows.
        # Collect all PSF files used across both FPMs for the CSV record.
        all_psf_files = []
        for r in results_list:
            for f in r['eef_info']['psf_files']:
                if f not in all_psf_files:
                    all_psf_files.append(f)

        eef_combined_info = {
            'eef':          eef_eff,
            'theta_arcmin': None,   # per-FPM angles are in individual rows
            'pointing_ra':  None,
            'pointing_dec': None,
            'psf_files':    all_psf_files,
            'pix_scale_arcsec': None,
            'extrapolated': any(r['eef_info']['extrapolated']
                                for r in results_list),
            'eef_capped':   None,
            'eef_extrap':   None,
        }
    else:
        eef_eff = None

    # -- Combined results table -----------------------------------------------
    e_lo, e_hi = results_list[0]['energy']
    ul_results = print_results_table(
        N_total, B_total, t_comb, N_bkg_total, area_ratio,
        cfg.confidence_levels, eef=eef_eff)

    # -- Combined CSV rows ----------------------------------------------------
    # Use date from first module's result (same observation, same date)
    date_obs_combined = results_list[0].get('date_obs', '')
    csv_rows = _build_csv_rows(
        'AB', e_lo, e_hi, N_total, N_bkg_total, B_total,
        area_ratio, t_comb, ul_results, eef_combined_info, cfg.obsid,
        date_obs=date_obs_combined)

    return csv_rows


# =============================================================================
# CONVENIENCE ENTRY POINT
# =============================================================================

def run_uplim(base_path, obsid, ra, dec, **kwargs):
    """
    Run the full upper-limit pipeline with minimal boilerplate.

    Parameters
    ----------
    base_path : str   — root data directory
    obsid     : str   — NuSTAR observation ID
    ra        : str or float  — source RA
    dec       : str or float  — source Dec
    **kwargs  : any Config field by name, e.g.
                    src_radius_arcsec=30.0,
                    energy_band='soft',
                    confidence_levels=[0.9973],
                    caldb_dir='/path/to/caldb',
                    save_plots=False

    Returns
    -------
    list of result dicts (one per module processed)

    Example
    -------
    >>> from nustar_uplim import run_uplim
    >>> results = run_uplim(
    ...     base_path = "/data/NuSTAR/2017gas/",
    ...     obsid     = "80202052002",
    ...     ra        = "20:17:11.360",
    ...     dec       = "+58:12:08.10",
    ...     energy_band       = 'soft',
    ...     confidence_levels = [0.9545, 0.9973],
    ...     caldb_dir         = "/path/to/caldb",
    ... )
    """
    cfg = Config(base_path=base_path, obsid=obsid, ra=ra, dec=dec, **kwargs)
    cfg.validate()

    e_lo, e_hi = cfg.resolve_energy_band()
    src_coord  = parse_coord(cfg.ra, cfg.dec)
    out_dir    = os.path.join(cfg.base_path, cfg.obsid, "ul_products")

    print("NuSTAR Non-Detection Upper Limit")
    print("=" * 70)
    print(f"Source  :  RA = {src_coord.ra.deg:.6f} deg  "
          f"Dec = {src_coord.dec.deg:.6f} deg")
    if isinstance(cfg.energy_band, tuple):
        band_label = f"{e_lo:.1f}-{e_hi:.1f} keV (custom)"
    else:
        band_label = f"'{cfg.energy_band}'  ({e_lo:.1f}-{e_hi:.1f} keV)"
    print(f"Band    :  {band_label}")
    print(f"Modules :  {', '.join(f'FPM{m}' for m in cfg.modules)}")
    print(f"Exp stat:  {cfg.exp_stat}  (primary)")
    if cfg.caldb_dir:
        print(f"CALDB   :  {cfg.caldb_dir}")
    else:
        caldb_env = os.environ.get('CALDB', '')
        if caldb_env:
            print(f"CALDB   :  {caldb_env}  ($CALDB)")
        else:
            print(f"CALDB   :  not set — EEF correction will be skipped")

    # -- Per-module -----------------------------------------------------------
    all_results = []
    all_csv_rows = []
    for mod in cfg.modules:
        result = process_module(mod, src_coord, cfg)
        all_results.append(result)
        all_csv_rows.extend(result['csv_rows'])

    # -- Combined (if both FPMs processed) ------------------------------------
    if len(all_results) > 1:
        combined_rows = combine_modules(all_results, cfg)
        all_csv_rows.extend(combined_rows)

    # -- Write CSV ------------------------------------------------------------
    write_results_csv(all_csv_rows, out_dir, obsid)

    print("\nDone.")
    return all_results
