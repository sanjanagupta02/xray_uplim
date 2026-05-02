"""
xray_uplim.xmm.pipeline
------------------------
Per-instrument extraction and upper-limit calculation for XMM-Newton EPIC.

Each instrument (MOS1, MOS2, PN) is processed independently.  Results are
never combined across instruments — PN and MOS have different effective areas,
response matrices, and PSF shapes.

Public API
----------
run_uplim(**kwargs)               — entry point; builds XMMConfig and loops
process_instrument(instrument, cfg)
                                  — full pipeline for one EPIC instrument
"""

import csv
import os
import warnings
import numpy as np

from .config   import XMMConfig
from .io       import locate_files, load_events, load_expmap
from .aperture import extract_src_bkg_counts, extract_exposure
from .eef      import compute_xmm_eef
from ..coords  import parse_coord
from ..statistics import net_count_rate, kraft_upper_limit, gehrels_upper_limit


# =============================================================================
# RESULTS TABLE
# =============================================================================

def _print_results_table(N_src, B_scaled, t_eff, N_bkg_raw, area_ratio,
                          confidence_levels, eef=None):
    """
    Compute and print upper limits at every confidence level.

    Columns always printed
    ----------------------
    CL              one-sided confidence level
    Net CR          (N_src - B_scaled) / t_eff  [point estimate, not a UL]
    Kraft S_ul      Kraft+91 upper limit (counts)
    Kraft CR_ap     aperture count-rate upper limit  = S_ul / t_eff
    Geh S_ul        Gehrels 1986 upper limit (counts)
    Geh CR_ap       aperture count-rate upper limit (Gehrels)

    Additional columns when eef is not None
    ----------------------------------------
    Kraft CR_tot    EEF-corrected total source rate = S_ul / (t_eff * EEF)
    Geh CR_tot      same for Gehrels

    Returns
    -------
    list of dicts — one per confidence level
    """
    CR_net, CR_sigma = net_count_rate(N_src, B_scaled, t_eff,
                                       N_bkg_raw, area_ratio)

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
        print(f"  EEF correction skipped (PSF CCF file not found).")

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
            'CR_net':             f"{r['CR_net']:.6e}",
            'CR_sigma':           f"{r['CR_sigma']:.6e}",
            'S_kraft':            f"{r['S_kraft']:.4f}",
            'CR_kraft_aperture':  f"{r['CR_kraft_aperture']:.6e}",
            'S_gehrels':          f"{r['S_gehrels']:.4f}",
            'CR_gehrels_aperture': f"{r['CR_gehrels_aperture']:.6e}",
            # EEF fields (empty when EEF not available)
            'theta_arcmin':       '',
            'eef':                '',
            'energy_ev':          '',
            'psf_file':           '',
            'eef_extrapolated':   '',
            'eef_capped':         '',
            'CR_kraft_total':     '',
            'CR_gehrels_total':   '',
        }
        if eef_info is not None:
            row['theta_arcmin']     = f"{eef_info['theta_arcmin']:.4f}"
            row['eef']              = f"{eef_info['eef']:.6f}"
            row['energy_ev']        = f"{eef_info['energy_ev']:.0f}"
            row['psf_file']         = os.path.basename(eef_info['psf_file'])
            row['eef_extrapolated'] = str(eef_info['extrapolated'])
            row['eef_capped']       = (f"{eef_info['eef_capped']:.6f}"
                                       if eef_info['eef_capped'] is not None else '')
            if r['CR_kraft_total'] is not None:
                row['CR_kraft_total']   = f"{r['CR_kraft_total']:.6e}"
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
# PER-INSTRUMENT PIPELINE
# =============================================================================

def process_instrument(instrument: str, cfg: XMMConfig):
    """
    Full extraction and upper-limit calculation for one EPIC instrument.

    Steps
    -----
    1.  Locate event file and exposure map.
    2.  Load and filter events (PATTERN, FLAG, PI).
    3.  Load exposure map.
    4.  Convert source RA/Dec to event-file and exposure-map pixel coordinates.
    5.  (Optional) Open interactive region selector GUI.
    6.  Extract source and background counts from event table.
    7.  Compute effective exposure from exposure map.
    8.  Compute EEF from XMM CCF PSF file.
    9.  Print and return results table.
    10. Save diagnostic plots.
    11. Write per-instrument CSV row.

    Parameters
    ----------
    instrument : 'MOS1', 'MOS2', or 'PN'
    cfg        : XMMConfig (validated before calling)

    Returns
    -------
    dict with keys:
        instrument, N_src, N_bkg_raw, B_scaled, area_ratio,
        net_counts, t_eff_s, exp_stats, ul, energy, eef_info, csv_rows
    """
    e_lo, e_hi = cfg.resolve_energy_band()
    out_dir    = os.path.join(cfg.data_dir, "ul_products")
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  {instrument}")
    print(f"{'='*70}")

    # -- Step 1: locate files -------------------------------------------------
    evt_file, exp_file = locate_files(cfg, instrument)
    print(f"  Event file : {os.path.basename(evt_file)}")
    print(f"  Expmap     : {os.path.basename(exp_file)}")

    # -- Step 2: load events --------------------------------------------------
    events, evt_hdr, pi_lo, pi_hi = load_events(cfg, evt_file, instrument)
    date_obs = str(evt_hdr.get('DATE-OBS', '')).strip()

    # -- Step 3: load exposure map --------------------------------------------
    exp_data, exp_hdr = load_expmap(exp_file)

    # -- Step 4: source pixel position ----------------------------------------
    from ..coords import sky_to_evt_pixel, sky_to_img_pixel
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

    if cfg.use_gui:
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

    print(f"\n  Src aperture : {cfg.src_radius_arcsec:.1f}\"")
    if cfg.bkg_mode == 'annulus':
        r_in = cfg.src_radius_arcsec * cfg.bkg_inner_factor
        print(f"  Bkg annulus  : {r_in:.1f}\" — {cfg.bkg_radius_arcsec:.1f}\"")
    else:
        print(f"  Bkg circle   : r={cfg.bkg_radius_arcsec:.1f}\"  (manual centre)")

    # -- Step 6: source and background counts ---------------------------------
    print()
    N_src, N_bkg_raw, area_ratio, cx_evt, cy_evt, pscale_evt = \
        extract_src_bkg_counts(events, evt_hdr, cfg, instrument,
                               bkg_cx_evt=bkg_cx_evt, bkg_cy_evt=bkg_cy_evt)
    B_scaled = N_bkg_raw * area_ratio

    print(f"  Area ratio   (src / bkg) : {area_ratio:.5f}")
    print(f"  Scaled bkg   B           : {B_scaled:.3f} cts")
    print(f"  Net counts   (N_src - B) : {N_src - B_scaled:.3f} cts")

    # -- Step 7: effective exposure -------------------------------------------
    print()
    exp_stats, exp_meta, cx_exp, cy_exp = extract_exposure(
        exp_data, exp_hdr, cfg, instrument)

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

    # -- Step 9: results table ------------------------------------------------
    eef_val    = eef_info['eef'] if eef_info is not None else None
    ul_results = _print_results_table(
        N_src, B_scaled, t_eff, N_bkg_raw, area_ratio,
        cfg.confidence_levels, eef=eef_val)

    # -- Step 10: diagnostic plots --------------------------------------------
    if cfg.save_plots:
        _save_plots(evt_x, evt_y, cx_evt, cy_evt, pscale_evt,
                    exp_meta, exp_stats,
                    instrument, e_lo, e_hi, cfg, out_dir,
                    src_coord, bkg_cx_evt, bkg_cy_evt)

    # -- Step 11: CSV rows ----------------------------------------------------
    csv_rows = _build_csv_rows(
        instrument, e_lo, e_hi, N_src, N_bkg_raw, B_scaled,
        area_ratio, t_eff, ul_results, eef_info, cfg.obsid,
        date_obs=date_obs)

    return {
        'instrument': instrument,
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
# DIAGNOSTIC PLOTS
# =============================================================================

def _save_plots(evt_x, evt_y, cx_evt, cy_evt, pscale_evt,
                exp_meta, exp_stats,
                instrument, e_lo, e_hi, cfg, out_dir,
                src_coord, bkg_cx_evt, bkg_cy_evt):
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
        instrument, e_lo, e_hi, cfg.obsid, cfg, out_dir)

    exposure_histogram(exp_meta, exp_stats, instrument, cfg, out_dir)

    region_image(
        evt_x, evt_y, cx_evt, cy_evt, pscale_evt,
        instrument, e_lo, e_hi, cfg.obsid, cfg, out_dir,
        src_ra_deg  = src_coord.ra.deg,
        src_dec_deg = src_coord.dec.deg,
        bkg_cx_evt  = bkg_cx_evt,
        bkg_cy_evt  = bkg_cy_evt)


# =============================================================================
# ENTRY POINT
# =============================================================================

def run_uplim(data_dir, obsid, ra, dec, **kwargs):
    """
    Run the full XMM-Newton upper-limit pipeline.

    Parameters
    ----------
    data_dir : str           — ODF working directory (epproc/emproc output)
    obsid    : str           — XMM observation ID (e.g. '0881990901')
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
    list of result dicts — one per instrument processed
    """
    cfg = XMMConfig(data_dir=data_dir, obsid=obsid, ra=ra, dec=dec, **kwargs)
    cfg.validate()

    e_lo, e_hi = cfg.resolve_energy_band()
    src_coord  = parse_coord(cfg.ra, cfg.dec)
    out_dir    = os.path.join(cfg.data_dir, "ul_products")

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
    print(f"Obs ID      :  {cfg.obsid}")
    print()

    all_results  = []
    all_csv_rows = []

    for instrument in cfg.instruments:
        try:
            result = process_instrument(instrument, cfg)
            all_results.append(result)
            all_csv_rows.extend(result['csv_rows'])
        except FileNotFoundError as exc:
            warnings.warn(
                f"\nSkipping {instrument}: {exc}",
                UserWarning, stacklevel=2)
            continue
        except Exception as exc:
            warnings.warn(
                f"\nError processing {instrument}: {exc}",
                UserWarning, stacklevel=2)
            raise

    # -- Summary --------------------------------------------------------------
    if all_results:
        print(f"\n{'='*70}")
        print("  SUMMARY")
        print(f"{'='*70}")
        print(f"  {'Instrument':<8}  {'N_src':>6}  {'B_scaled':>9}  "
              f"{'t_eff (ks)':>11}  {'EEF':>6}  "
              f"{'Kraft CR_ap (3σ)':>18}")
        print("  " + "-" * 68)
        for r in all_results:
            eef_str = (f"{r['eef_info']['eef']:.3f}"
                       if r['eef_info'] is not None else "  N/A")
            # 3σ row — first CL >= 0.997
            ul_row  = next((u for u in r['ul'] if u['cl'] >= 0.997), r['ul'][-1])
            print(f"  {r['instrument']:<8}  {r['N_src']:>6}  "
                  f"{r['B_scaled']:>9.2f}  "
                  f"{r['t_eff_s']/1e3:>11.3f}  "
                  f"{eef_str:>6}  "
                  f"{ul_row['CR_kraft_aperture']:>18.4e}")
        print()

    # -- Write CSV ------------------------------------------------------------
    if all_csv_rows:
        write_results_csv(all_csv_rows, out_dir, cfg.obsid)

    return all_results
