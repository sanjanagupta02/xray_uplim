"""
xray_uplim.nustar.pipeline
--------------------------
Top-level orchestration: per-module extraction and combined results.

Public API
----------
run_uplim(...)           — convenience wrapper; builds a Config and calls process_observations
process_observations(cfg)— main entry point; handles single- and multi-obsid co-adding
process_module(...)      — extract counts + exposure for one FPM (single-obsid, backward compat)
combine_modules(...)     — sum across FPMs and print combined upper limits
"""

import copy
import csv
import os
import warnings
import numpy as np

from .config      import Config
from .io          import locate_files, load_events, load_expmap
from ..coords     import parse_coord, sky_to_evt_pixel, sky_to_img_pixel
from ..eef        import compute_eef
from ..exposure   import compute_exposure_stats, compute_exposure_area_ratio
from ..statistics import net_count_rate, marginalized_upper_limit, gehrels_upper_limit
from ..plots      import radial_profile, exposure_histogram, region_image


# =============================================================================
# CORE COMPUTATION (no printing, no I/O)
# =============================================================================

def _compute_ul_results(N_src, B_scaled, t_eff, N_bkg_raw, area_ratio,
                        confidence_levels, eef=None):
    """
    Compute upper-limit results at every confidence level.  No printing.

    Returns
    -------
    list of dicts, one per CL, with keys:
        cl, CR_net, CR_sigma,
        CR_marg_aperture,   CR_marg_total   (None if no EEF),
        S_gehrels, CR_gehrels_aperture, CR_gehrels_total (None if no EEF)
    """
    CR_net, CR_sigma = net_count_rate(N_src, B_scaled, t_eff,
                                      N_bkg_raw, area_ratio)
    results = []
    for cl in confidence_levels:
        CR_m_ap = marginalized_upper_limit(N_src, N_bkg_raw, area_ratio, t_eff, cl)
        S_g     = gehrels_upper_limit(N_src, B_scaled, cl)
        CR_g_ap = S_g / t_eff
        if eef is not None and eef > 0:
            # Total source rate: fold EEF into the effective exposure.
            # marginalized_upper_limit(N, B, α, t×EEF, cl) is mathematically
            # identical to CR_m_ap / EEF (the Poisson CDF transforms linearly
            # under the S_ap → S_tot = S_ap/EEF change of variables), but this
            # form makes the physical model explicit: the expected source counts
            # are S_tot × EEF × t_eff, and we integrate directly over S_tot.
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

def print_results_table(N_src, B_scaled, t_eff, N_bkg_raw, area_ratio,
                        confidence_levels, eef=None):
    """
    Compute and print all three methods at every confidence level.

    Parameters
    ----------
    eef : float or None
        Encircled energy fraction.  If None, EEF-corrected columns are omitted.

    Returns
    -------
    list of dicts — same as _compute_ul_results()
    """
    results = _compute_ul_results(N_src, B_scaled, t_eff, N_bkg_raw,
                                  area_ratio, confidence_levels, eef=eef)
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
            f"{'Marg CR_ap':>13}  {'Marg CR_tot':>13}  "
            f"{'Geh S_ul':>10}  {'Geh CR_ap':>13}  {'Geh CR_tot':>13}"
        )
    else:
        header = (
            f"  {'CL':>8}  {'Net CR':>13}  "
            f"{'Marg CR_ap':>13}  "
            f"{'Geh S_ul':>10}  {'Geh CR_ap':>13}"
        )
    divider = "  " + "-" * (len(header) - 2)
    print(header)
    print(divider)

    for r in results:
        if eef is not None:
            print(
                f"  {r['cl']:8.4f}  {r['CR_net']:+13.4e}  "
                f"{r['CR_marg_aperture']:13.4e}  "
                f"{r['CR_marg_total']:13.4e}  "
                f"{r['S_gehrels']:10.3f}  {r['CR_gehrels_aperture']:13.4e}  "
                f"{r['CR_gehrels_total']:13.4e}"
            )
        else:
            print(
                f"  {r['cl']:8.4f}  {r['CR_net']:+13.4e}  "
                f"{r['CR_marg_aperture']:13.4e}  "
                f"{r['S_gehrels']:10.3f}  {r['CR_gehrels_aperture']:13.4e}"
            )

    print(divider)
    if eef is not None:
        print(f"  Marg CR_ap  = marginalized aperture count-rate UL (cts/s).")
        print(f"  Marg CR_tot = total source rate UL; computed via Bayesian integral")
        print(f"                with effective exposure t_eff × EEF (EEF={eef:.4f}).")
    else:
        print(f"  Marg CR_ap is the marginalized aperture count-rate upper limit.")
        print(f"  EEF correction skipped (set caldb_dir to enable).")

    return results


# =============================================================================
# CSV OUTPUT
# =============================================================================

def write_results_csv(rows, out_dir, obsid_label):
    """
    Write upper-limit results to a CSV (and .xlsx) file.

    Parameters
    ----------
    rows        : list of dicts
    out_dir     : str  — output directory (created if absent)
    obsid_label : str  — used in the output filename

    Returns
    -------
    csv_path : str
    """
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, f"nustar_uplim_{obsid_label}.csv")

    fieldnames = [
        'result_type',
        'obsid', 'date_obs', 'module', 'energy_lo_kev', 'energy_hi_kev',
        'N_src', 'N_bkg_raw', 'B_scaled', 'area_ratio',
        't_eff_s',
        'theta_arcmin', 'eef', 'psf_file', 'eef_extrapolated',
        'eef_capped', 'eef_extrap',
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

    from ..output import write_results_xlsx
    xlsx_path = csv_path.replace('.csv', '.xlsx')
    if write_results_xlsx(rows, fieldnames, xlsx_path, text_cols=('obsid',)):
        print(f"  Excel file written: {xlsx_path}")

    return csv_path


def _build_csv_rows(module, e_lo, e_hi, N_src, N_bkg_raw, B_scaled,
                    area_ratio, t_eff, ul_results, eef_info, obsid,
                    date_obs='', result_type='individual'):
    """
    Build a list of CSV row dicts (one per confidence level) for one module.

    Parameters
    ----------
    eef_info    : dict or None — return value of compute_eef(), or None if skipped
    date_obs    : str          — DATE-OBS from event file header (ISO 8601)
    result_type : str          — 'individual' (per-obs) or 'combined' (co-added)
    """
    rows = []
    for r in ul_results:
        row = {
            'result_type':        result_type,
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
            'CR_net':              f"{r['CR_net']:.6e}",
            'CR_sigma':            f"{r['CR_sigma']:.6e}",
            'CR_marg_aperture':    f"{r['CR_marg_aperture']:.6e}",
            'S_gehrels':           f"{r['S_gehrels']:.4f}",
            'CR_gehrels_aperture': f"{r['CR_gehrels_aperture']:.6e}",
        }
        if eef_info is not None:
            row['theta_arcmin']     = (f"{eef_info['theta_arcmin']:.4f}"
                                       if eef_info['theta_arcmin'] is not None else '')
            row['eef']              = f"{eef_info['eef']:.6f}"
            row['psf_file']         = '; '.join(
                                          os.path.basename(f)
                                          for f in eef_info['psf_files'])
            row['eef_extrapolated'] = eef_info['extrapolated']
            row['eef_capped']       = (f"{eef_info['eef_capped']:.6f}"
                                       if eef_info['eef_capped'] is not None else '')
            row['eef_extrap']       = (f"{eef_info['eef_extrap']:.6f}"
                                       if eef_info['eef_extrap'] is not None else '')
            if r['CR_marg_total'] is not None:
                row['CR_marg_total']    = f"{r['CR_marg_total']:.6e}"
                row['CR_gehrels_total'] = f"{r['CR_gehrels_total']:.6e}"
        rows.append(row)
    return rows


# =============================================================================
# PER-MODULE EXTRACTION (single obsid, internal helper)
# =============================================================================

def _load_and_extract_module(module, obsid_str, obs_root, src_coord, cfg,
                              run_gui=False):
    """
    Load files and extract raw counts / exposure / EEF for one FPM of one
    observation.  Prints diagnostics but does NOT print the UL table.

    Parameters
    ----------
    module    : str      — 'A' or 'B'
    obsid_str : str      — observation ID (used for file lookup and plot names)
    obs_root  : str      — <base_path>/<obsid_str>
    src_coord : SkyCoord
    cfg       : Config   — may be mutated by the GUI
    run_gui   : bool     — whether to open the interactive region selector

    Returns
    -------
    dict with keys:
        module, obsid_str, date_obs,
        N_src, N_bkg_raw, B_scaled, area_ratio,
        t_eff, exp_stats, eef_info,
        e_lo, e_hi,
        bkg_cx_evt, bkg_cy_evt
    """
    e_lo, e_hi = cfg.resolve_energy_band()
    out_dir    = os.path.join(obs_root, "ul_products")
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  FPM{module}  [obs: {obsid_str}]")
    print(f"{'='*70}")

    # -- Locate and load files ------------------------------------------------
    evt_file, exp_file = locate_files(cfg.base_path, obsid_str, module)
    print(f"  Event file  : {os.path.basename(evt_file)}")
    print(f"  Expo map    : {os.path.basename(exp_file)}")

    evts, evt_hdr, PI_lo, PI_hi = load_events(evt_file, e_lo, e_hi)
    date_obs = str(evt_hdr.get('DATE-OBS', '')).strip()
    print(f"  Date-Obs    : {date_obs or '(not in header)'}")
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
    bkg_cx_evt = cx_evt
    bkg_cy_evt = cy_evt

    if run_gui:
        from ..region_selector import select_regions_interactive
        print(f"\n  Opening interactive region selector for FPM{module}...")
        sel = select_regions_interactive(
            evt_x, evt_y, cx_evt, cy_evt, pscale_evt, cfg, f'FPM{module}')

        cx_evt     = sel['cx']
        cy_evt     = sel['cy']
        bkg_cx_evt = sel['bkg_cx']
        bkg_cy_evt = sel['bkg_cy']

        # Write confirmed radii back to cfg so subsequent obs use them
        cfg.src_radius_arcsec = sel['src_radius_arcsec']
        cfg.bkg_radius_arcsec = sel['bkg_radius_arcsec']
        cfg.bkg_inner_factor  = sel['bkg_inner_factor']

        # If user moved background, switch to manual mode (sky coords stored in cfg)
        bkg_moved = (abs(bkg_cx_evt - cx_evt) > 1.0 or
                     abs(bkg_cy_evt - cy_evt) > 1.0)
        if bkg_moved:
            try:
                from astropy.wcs import WCS as _WCS
                _wcs = _WCS(evt_hdr, naxis=2)
                _bkg_ra, _bkg_dec = _wcs.pixel_to_world_values(
                    bkg_cx_evt - 1, bkg_cy_evt - 1)
                cfg.bkg_mode = 'manual'
                cfg.bkg_ra   = str(float(_bkg_ra))
                cfg.bkg_dec  = str(float(_bkg_dec))
                print(f"  [GUI] Background → manual mode: "
                      f"RA={float(_bkg_ra):.5f}  Dec={float(_bkg_dec):.5f}")
            except Exception as _e:
                warnings.warn(
                    f"Could not convert background pixel to RA/Dec ({_e}). "
                    "Falling back to annulus mode.")
                cfg.bkg_mode = 'annulus'
                bkg_cx_evt   = cx_evt
                bkg_cy_evt   = cy_evt

    elif cfg.bkg_mode == 'manual' and cfg.bkg_ra and cfg.bkg_dec:
        # Re-project manual background sky coords into this obs's pixel frame
        bkg_coord = parse_coord(cfg.bkg_ra, cfg.bkg_dec)
        bkg_cx_evt, bkg_cy_evt, _ = sky_to_evt_pixel(
            bkg_coord.ra.deg, bkg_coord.dec.deg, evt_hdr)

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
        d_bkg_center = np.sqrt((evt_x - bkg_cx_evt)**2 +
                               (evt_y - bkg_cy_evt)**2)
        in_annulus = (d_bkg_center > r_bkg_in_evt) & (d_bkg_center <= r_bkg_out_evt)
        N_bkg_raw  = int(np.sum(in_annulus))

    elif cfg.bkg_mode == 'manual':
        bkg_coord      = parse_coord(cfg.bkg_ra, cfg.bkg_dec)
        cx_b, cy_b, _  = sky_to_evt_pixel(
            bkg_coord.ra.deg, bkg_coord.dec.deg, evt_hdr)
        r_bkg_circ     = bkg_radius_arcsec / pscale_evt
        d_bkg          = np.sqrt((evt_x - cx_b)**2 + (evt_y - cy_b)**2)
        N_bkg_raw      = int(np.sum(d_bkg <= r_bkg_circ))
    else:
        raise ValueError(f"Unknown bkg_mode: '{cfg.bkg_mode}'")

    # -- Tier 1: exposure-map-weighted area ratio ------------------------------
    # Using the exposure map instead of pure geometry accounts for vignetting
    # differences between source and background apertures.
    r_bkg_in_exp  = r_bkg_in_arcsec  / pscale_exp
    r_bkg_out_exp = r_bkg_out_arcsec / pscale_exp

    try:
        if cfg.bkg_mode == 'annulus':
            area_ratio = compute_exposure_area_ratio(
                exp_data, cx_exp, cy_exp, r_src_exp,
                'annulus',
                r_bkg_inner_pix=r_bkg_in_exp,
                r_bkg_outer_pix=r_bkg_out_exp)
        else:  # manual
            bkg_coord_exp  = parse_coord(cfg.bkg_ra, cfg.bkg_dec)
            cx_bkg_exp, cy_bkg_exp, _ = sky_to_img_pixel(
                bkg_coord_exp.ra.deg, bkg_coord_exp.dec.deg, exp_hdr)
            r_bkg_circ_exp = bkg_radius_arcsec / pscale_exp
            area_ratio = compute_exposure_area_ratio(
                exp_data, cx_exp, cy_exp, r_src_exp,
                'manual',
                cx_bkg=cx_bkg_exp, cy_bkg=cy_bkg_exp,
                r_bkg_pix=r_bkg_circ_exp)
    except RuntimeError as _exc:
        warnings.warn(
            f"Exposure-weighted area ratio failed for FPM{module} "
            f"({_exc}); falling back to geometric ratio.",
            UserWarning, stacklevel=2)
        # geometric fallback
        if cfg.bkg_mode == 'annulus':
            area_ratio = (r_src_evt**2 /
                          (r_bkg_out_evt**2 - r_bkg_in_evt**2))
        else:
            area_ratio = (r_src_evt / r_bkg_circ)**2

    B_scaled   = N_bkg_raw * area_ratio

    print(f"\n  Source counts  (N_src)        : {N_src}")
    print(f"  Background counts (raw)       : {N_bkg_raw}")
    print(f"  Area ratio  (exp-weighted)    : {area_ratio:.5f}")
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
    eef_info  = None
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

    # -- Diagnostic plots -----------------------------------------------------
    if cfg.save_plots:
        radial_profile(evt_x, evt_y, cx_evt, cy_evt, pscale_evt,
                       f'FPM{module}', e_lo, e_hi, obsid_str, cfg, out_dir)
        exposure_histogram(exp_meta, exp_stats, f'FPM{module}', cfg, out_dir)
        region_image(evt_x, evt_y, cx_evt, cy_evt, pscale_evt,
                     f'FPM{module}', e_lo, e_hi, obsid_str, cfg, out_dir,
                     src_ra_deg=src_coord.ra.deg,
                     src_dec_deg=src_coord.dec.deg,
                     bkg_cx_evt=bkg_cx_evt,
                     bkg_cy_evt=bkg_cy_evt)

    return {
        'module':     module,
        'obsid_str':  obsid_str,
        'date_obs':   date_obs,
        'N_src':      N_src,
        'N_bkg_raw':  N_bkg_raw,
        'B_scaled':   B_scaled,
        'area_ratio': area_ratio,
        't_eff':      t_eff,
        'exp_stats':  exp_stats,
        'eef_info':   eef_info,
        'e_lo':       e_lo,
        'e_hi':       e_hi,
        'bkg_cx_evt': bkg_cx_evt,
        'bkg_cy_evt': bkg_cy_evt,
    }


# =============================================================================
# COMBINED (FPM-A + FPM-B)
# =============================================================================

def combine_modules(results_list, cfg, obsid_label=None):
    """
    Sum counts across FPMs and compute combined results.

    Combining strategy
    ------------------
    N_total  = sum(N_src)      — counts additive across independent detectors
    B_total  = sum(B_scaled)   — each B already scaled to source aperture area
    t_comb   = sum(t_eff_s)    — exposures add (correct for additive counts)

    EEF denominator:  sum_i(t_eff_i * EEF_i)
    — each FPM contributes counts collected through its own EEF.
    Represented as an effective EEF = sum(t*EEF) / t_comb.

    Parameters
    ----------
    results_list : list of dicts from process_module() or per-module combined dicts
    cfg          : Config
    obsid_label  : str or None — obsid string for CSV rows; defaults to cfg.obsid

    Returns
    -------
    list of combined CSV row dicts
    """
    if obsid_label is None:
        obsid_label = (cfg.obsid if isinstance(cfg.obsid, str)
                       else '+'.join(cfg.obsids))

    print(f"\n{'='*70}")
    print("  COMBINED  FPM-A + FPM-B")
    print(f"{'='*70}")

    N_total     = sum(r['N_src']     for r in results_list)
    B_total     = sum(r['B_scaled']  for r in results_list)
    N_bkg_total = sum(r['N_bkg_raw'] for r in results_list)
    area_ratio  = results_list[0]['area_ratio']
    t_vals      = [r['t_eff_s'] for r in results_list]
    t_comb      = float(np.sum(t_vals))

    print(f"  Combined N_src    : {N_total}")
    print(f"  Combined B_scaled : {B_total:.3f} cts")
    for r in results_list:
        print(f"  t_eff FPM-{r['module']}       : {r['t_eff_s']/1e3:.3f} ks")
    print(f"  t_eff (combined)  : {t_comb/1e3:.3f} ks  "
          f"[sum — correct for additive counts]")

    # -- Combined EEF ---------------------------------------------------------
    eef_combined_info = None
    all_have_eef = all(r['eef_info'] is not None for r in results_list)

    if all_have_eef:
        t_eef_sum = sum(
            r['t_eff_s'] * r['eef_info']['eef'] for r in results_list)
        eef_eff = t_eef_sum / t_comb if t_comb > 0 else None

        print(f"\n  -- Combined EEF -------------------------------------------")
        for r in results_list:
            ei = r['eef_info']
            theta_str = (f"{ei['theta_arcmin']:.3f}'"
                         if ei['theta_arcmin'] is not None else 'N/A')
            print(f"    FPM-{r['module']}: theta = {theta_str}  "
                  f"EEF = {ei['eef']:.4f}  "
                  f"t_eff * EEF = {r['t_eff_s'] * ei['eef'] / 1e3:.3f} ks")
        print(f"    Combined t_eff * EEF denominator : {t_eef_sum/1e3:.3f} ks")
        print(f"    Effective EEF (= sum(t*EEF)/t_comb) : {eef_eff:.4f}")

        all_psf_files = []
        for r in results_list:
            for f in r['eef_info']['psf_files']:
                if f not in all_psf_files:
                    all_psf_files.append(f)

        eef_combined_info = {
            'eef':          eef_eff,
            'theta_arcmin': None,
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
    date_obs_combined = results_list[0].get('date_obs', '')
    csv_rows = _build_csv_rows(
        'AB', e_lo, e_hi, N_total, N_bkg_total, B_total,
        area_ratio, t_comb, ul_results, eef_combined_info, obsid_label,
        date_obs=date_obs_combined, result_type='combined')

    return csv_rows


# =============================================================================
# SINGLE-MODULE PUBLIC WRAPPER  (backward compat)
# =============================================================================

def process_module(module, src_coord, cfg):
    """
    Full extraction and result calculation for one FPM (single-obsid).

    This is a thin wrapper around _load_and_extract_module that also prints
    the UL table and builds CSV rows.  Kept for backward compatibility.

    Parameters
    ----------
    module    : str    — 'A' or 'B'
    src_coord : SkyCoord
    cfg       : Config

    Returns
    -------
    dict compatible with combine_modules() and the old API.
    """
    obsid_str = cfg.obsids[0]
    obs_root  = os.path.join(cfg.base_path, obsid_str)

    raw = _load_and_extract_module(
        module, obsid_str, obs_root, src_coord, cfg, run_gui=cfg.use_gui)

    e_lo, e_hi  = raw['e_lo'], raw['e_hi']
    eef_val     = raw['eef_info']['eef'] if raw['eef_info'] is not None else None

    ul_results = print_results_table(
        raw['N_src'], raw['B_scaled'], raw['t_eff'], raw['N_bkg_raw'],
        raw['area_ratio'], cfg.confidence_levels, eef=eef_val)

    csv_rows = _build_csv_rows(
        module, e_lo, e_hi, raw['N_src'], raw['N_bkg_raw'], raw['B_scaled'],
        raw['area_ratio'], raw['t_eff'], ul_results, raw['eef_info'], obsid_str,
        date_obs=raw['date_obs'], result_type='individual')

    return {
        'module':     module,
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
# MULTI-OBSID MAIN ENTRY POINT
# =============================================================================

def process_observations(cfg):
    """
    Main entry point for NuSTAR upper-limit calculation.

    Handles both single-obsid and multi-obsid (co-adding) runs transparently.
    When multiple obsids are given, per-observation results are reported
    alongside the combined co-added upper limit.

    Co-adding is exact Bayesian Poisson statistics:
        N_src_total = Σ N_src_i    (sum of independent Poisson r.v.s → Poisson)
        B_total     = Σ B_scaled_i
        t_total     = Σ t_eff_i
    Kraft+1991 is then applied to the totals — this is not an approximation.

    Parameters
    ----------
    cfg : Config

    Returns
    -------
    per_obs_raw : dict  {obsid_str: {module: raw_data_dict}}
    """
    obsids   = cfg.obsids
    n_obs    = len(obsids)
    src_coord = parse_coord(cfg.ra, cfg.dec)
    e_lo, e_hi = cfg.resolve_energy_band()

    # Label used in output filenames
    obsid_label = (cfg.obsid if isinstance(cfg.obsid, str)
                   else '+'.join(obsids))
    out_dir_main = os.path.join(cfg.base_path, obsids[0], "ul_products")

    # Save original aperture settings so gui_per_obs=True can restore them
    _orig_aperture = {k: getattr(cfg, k) for k in (
        'src_radius_arcsec', 'bkg_radius_arcsec', 'bkg_inner_factor',
        'bkg_mode', 'bkg_ra', 'bkg_dec')}

    # -- Print header ---------------------------------------------------------
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
    if n_obs > 1:
        print(f"Obs IDs :  {', '.join(obsids)}  [{n_obs} observations — co-adding]")
    else:
        print(f"Obs ID  :  {obsids[0]}")

    # -- Load and extract every (obs, module) pair ----------------------------
    # per_obs_raw[obsid_str][module] = dict from _load_and_extract_module
    per_obs_raw = {}

    for i, obsid_str in enumerate(obsids):
        obs_root = os.path.join(cfg.base_path, obsid_str)

        if n_obs > 1:
            print(f"\n{'#'*70}")
            print(f"  Observation {i+1}/{n_obs}:  {obsid_str}")
            print(f"{'#'*70}")

        # Determine the config object and GUI mode for this observation
        if cfg.use_gui and cfg.gui_per_obs:
            # Independent GUI for each obs: restore original aperture each time
            cfg_obs = copy.copy(cfg)
            for k, v in _orig_aperture.items():
                setattr(cfg_obs, k, v)
            run_gui_this_obs = True
        elif cfg.use_gui and i == 0:
            # GUI for first obs only; subsequent obs inherit cfg settings
            cfg_obs = cfg
            run_gui_this_obs = True
        else:
            cfg_obs = cfg
            run_gui_this_obs = False

        per_obs_raw[obsid_str] = {}
        for module in cfg.modules:
            raw = _load_and_extract_module(
                module, obsid_str, obs_root, src_coord, cfg_obs,
                run_gui=run_gui_this_obs)
            per_obs_raw[obsid_str][module] = raw

    # -- Per-module: individual ULs + combined across obsids ------------------
    all_csv_rows        = []
    module_combined_ab  = []   # aggregated module results for the AB combine step

    for module in cfg.modules:
        print(f"\n{'='*70}")
        if n_obs > 1:
            print(f"  FPM{module} — co-added across {n_obs} observations")
        else:
            print(f"  FPM{module} — results")
        print(f"{'='*70}")

        # Individual per-obs ULs (only printed/recorded when n_obs > 1)
        if n_obs > 1:
            print(f"\n  -- Individual per-observation upper limits (FPM{module}) --")
            for obsid_str in obsids:
                raw     = per_obs_raw[obsid_str][module]
                eef_ind = raw['eef_info']['eef'] if raw['eef_info'] else None
                ul_ind  = _compute_ul_results(
                    raw['N_src'], raw['B_scaled'], raw['t_eff'],
                    raw['N_bkg_raw'], raw['area_ratio'],
                    cfg.confidence_levels, eef=eef_ind)

                print(f"\n  Obs {obsid_str}:")
                print(f"    N_src={raw['N_src']}  B={raw['B_scaled']:.2f}  "
                      f"t_eff={raw['t_eff']/1e3:.3f} ks")
                for r in ul_ind:
                    tot_str = (f"  Marg CR_tot={r['CR_marg_total']:.3e}"
                               if r['CR_marg_total'] is not None else '')
                    print(f"    CL={r['cl']:.4f}  "
                          f"Marg CR_ap={r['CR_marg_aperture']:.3e}{tot_str}")

                ind_rows = _build_csv_rows(
                    module, e_lo, e_hi,
                    raw['N_src'], raw['N_bkg_raw'], raw['B_scaled'],
                    raw['area_ratio'], raw['t_eff'], ul_ind, raw['eef_info'],
                    obsid_str, date_obs=raw['date_obs'], result_type='individual')
                all_csv_rows.extend(ind_rows)

        # Combined across obsids for this module
        N_total     = sum(per_obs_raw[oid][module]['N_src']     for oid in obsids)
        B_total     = sum(per_obs_raw[oid][module]['B_scaled']  for oid in obsids)
        N_bkg_total = sum(per_obs_raw[oid][module]['N_bkg_raw'] for oid in obsids)
        area_ratio  = per_obs_raw[obsids[0]][module]['area_ratio']
        t_total     = sum(per_obs_raw[oid][module]['t_eff']     for oid in obsids)

        # EEF: exposure-weighted average across obsids
        all_eef = [per_obs_raw[oid][module]['eef_info'] for oid in obsids]
        if all(e is not None for e in all_eef):
            t_eef_sum = sum(
                per_obs_raw[oid][module]['t_eff'] *
                per_obs_raw[oid][module]['eef_info']['eef']
                for oid in obsids)
            eef_avg = t_eef_sum / t_total if t_total > 0 else None

            all_psf_files = []
            for ei in all_eef:
                for f in ei['psf_files']:
                    if f not in all_psf_files:
                        all_psf_files.append(f)

            eef_combined_info = {
                'eef':          eef_avg,
                'theta_arcmin': None,   # individual values in per-obs rows
                'pointing_ra':  None,
                'pointing_dec': None,
                'psf_files':    all_psf_files,
                'pix_scale_arcsec': None,
                'extrapolated': any(e['extrapolated'] for e in all_eef),
                'eef_capped':   None,
                'eef_extrap':   None,
            }
            eef_val = eef_avg

            if n_obs > 1:
                print(f"\n  -- EEF across {n_obs} observations (FPM{module}) --")
                for oid in obsids:
                    raw = per_obs_raw[oid][module]
                    ei  = raw['eef_info']
                    print(f"    {oid}: theta={ei['theta_arcmin']:.3f}'  "
                          f"EEF={ei['eef']:.4f}  "
                          f"t*EEF={raw['t_eff']*ei['eef']/1e3:.3f} ks")
                print(f"    Exposure-weighted EEF = {eef_avg:.4f}")
        else:
            eef_combined_info = None
            eef_val = None

        if n_obs > 1:
            print(f"\n  -- Combined ({n_obs} obs, FPM{module}) "
                  f"N_src={N_total}  B={B_total:.2f}  t_eff={t_total/1e3:.3f} ks --")

        ul_combined = print_results_table(
            N_total, B_total, t_total, N_bkg_total, area_ratio,
            cfg.confidence_levels, eef=eef_val)

        date_obs_first = per_obs_raw[obsids[0]][module].get('date_obs', '')
        comb_obsid     = obsid_label if n_obs > 1 else obsids[0]
        comb_rtype     = 'combined'   if n_obs > 1 else 'individual'

        comb_rows = _build_csv_rows(
            module, e_lo, e_hi, N_total, N_bkg_total, B_total,
            area_ratio, t_total, ul_combined, eef_combined_info,
            comb_obsid, date_obs=date_obs_first, result_type=comb_rtype)
        all_csv_rows.extend(comb_rows)

        # Store for the A+B combining step
        module_combined_ab.append({
            'module':     module,
            'date_obs':   date_obs_first,
            'N_src':      N_total,
            'N_bkg_raw':  N_bkg_total,
            'B_scaled':   B_total,
            'area_ratio': area_ratio,
            't_eff_s':    t_total,
            'eef_info':   eef_combined_info,
            'energy':     (e_lo, e_hi),
        })

    # -- A+B combined (if both FPMs processed) --------------------------------
    if len(cfg.modules) > 1:
        ab_rows = combine_modules(module_combined_ab, cfg, obsid_label=obsid_label)
        all_csv_rows.extend(ab_rows)

    # -- Write CSV + XLSX -----------------------------------------------------
    os.makedirs(out_dir_main, exist_ok=True)
    write_results_csv(all_csv_rows, out_dir_main, obsid_label)

    print("\nDone.")
    return per_obs_raw


# =============================================================================
# CONVENIENCE ENTRY POINT
# =============================================================================

def run_uplim(base_path, obsid, ra, dec, **kwargs):
    """
    Run the full upper-limit pipeline with minimal boilerplate.

    Parameters
    ----------
    base_path : str   — root data directory
    obsid     : str or list of str  — NuSTAR observation ID(s)
                  Single obs : obsid = "80202052002"
                  Co-adding  : obsid = ["80202052002", "80202052004"]
    ra        : str or float  — source RA
    dec       : str or float  — source Dec
    **kwargs  : any Config field, e.g.
                    src_radius_arcsec=30.0,
                    energy_band='soft',
                    confidence_levels=[0.9973],
                    caldb_dir='/path/to/caldb',
                    gui_per_obs=True,
                    save_plots=False

    Returns
    -------
    per_obs_raw : dict — {obsid_str: {module: raw_data_dict}}

    Example
    -------
    >>> from nustar_uplim import run_uplim
    >>> run_uplim(
    ...     base_path = "/data/NuSTAR/2017gas/",
    ...     obsid     = ["80202052002", "80202052004"],
    ...     ra        = "20:17:11.360",
    ...     dec       = "+58:12:08.10",
    ...     energy_band       = 'soft',
    ...     confidence_levels = [0.9545, 0.9973],
    ...     caldb_dir         = "/path/to/caldb",
    ... )
    """
    cfg = Config(base_path=base_path, obsid=obsid, ra=ra, dec=dec, **kwargs)
    cfg.validate()
    return process_observations(cfg)
