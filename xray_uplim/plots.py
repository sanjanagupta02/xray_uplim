"""
xray_uplim.plots
----------------
Diagnostic plots saved to <base_path>/<obsid>/ul_products/.

radial_profile()
    Log-scale radial surface-density profile of events around the source
    position.  Marks the source aperture, background annulus, and PSF
    half-FWHM.  A flat profile inside the source aperture (matching the
    background level) confirms a non-detection.

exposure_histogram()
    Distribution of exposure-map pixel values inside the source aperture,
    with vertical lines for all three summary statistics.

region_image()
    Sky image of the event field with source aperture and background annulus
    overlaid.  Saved as both PNG and PDF — suitable for inclusion in papers.
"""

import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


# ---------------------------------------------------------------------------
# Global plot style — Times New Roman throughout
# ---------------------------------------------------------------------------

matplotlib.rcParams.update({
    'font.family':        'serif',
    'font.serif':         ['Times New Roman', 'Times', 'DejaVu Serif'],
    'mathtext.fontset':   'stix',       # STIX matches Times New Roman for math
    'axes.titlesize':     12,
    'axes.labelsize':     11,
    'xtick.labelsize':    10,
    'ytick.labelsize':    10,
    'legend.fontsize':    9,
    'figure.dpi':         150,
})


# ---------------------------------------------------------------------------
# Radial surface-density profile
# ---------------------------------------------------------------------------

def radial_profile(evt_x, evt_y, cx_evt, cy_evt, pscale_evt,
                   label, e_lo, e_hi, obsid, cfg, out_dir):
    """
    Save a log-scale radial surface-density profile.

    Parameters
    ----------
    evt_x, evt_y : array  — event pixel coordinates (energy-filtered)
    cx_evt, cy_evt : float — source pixel position in event coordinates
    pscale_evt   : float  — event pixel scale in arcsec/pix
    label        : str    — instrument label, e.g. 'FPMA', 'MOS1', 'PN'
    e_lo, e_hi   : float  — energy band in keV
    obsid        : str
    cfg          : Config or XMMConfig
    out_dir      : str    — output directory
    """
    r_arcsec = (np.sqrt((evt_x - cx_evt)**2 + (evt_y - cy_evt)**2)
                * pscale_evt)
    max_r    = cfg.bkg_radius_arcsec * 1.15
    bins     = np.linspace(0, max_r, 45)
    counts, edges = np.histogram(r_arcsec, bins=bins)
    mids   = 0.5 * (edges[:-1] + edges[1:])
    areas  = np.pi * (edges[1:]**2 - edges[:-1]**2)
    surf   = np.where(areas > 0, counts / areas, 0.0)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.step(mids, np.where(surf > 0, surf, np.nan),
            where='mid', color='steelblue', lw=1.5, label='Binned events')

    r_inner = cfg.src_radius_arcsec * cfg.bkg_inner_factor
    ax.axvline(cfg.src_radius_arcsec, color='tomato', ls='--', lw=1.3,
               label=f'Src aperture ({cfg.src_radius_arcsec:.0f}")')
    ax.axvline(r_inner, color='darkorange', ls=':', lw=1.2,
               label=f'Bkg inner ({r_inner:.0f}")')
    ax.axvline(cfg.bkg_radius_arcsec, color='darkorange', ls='--', lw=1.2,
               label=f'Bkg outer ({cfg.bkg_radius_arcsec:.0f}")')
    ax.axvline(cfg.psf_fwhm_arcsec / 2.0, color='grey', ls=':', lw=1.0,
               label=f'PSF half-FWHM ({cfg.psf_fwhm_arcsec/2:.0f}")')

    ax.set_xlabel('Radius (arcsec)')
    ax.set_ylabel('Surface density (cts arcsec$^{-2}$)')
    ax.set_title(
        f'{label}  |  {e_lo:.1f}–{e_hi:.1f} keV  |  OBSID {obsid}')
    ax.legend(loc='upper right')
    ax.set_yscale('log')
    ax.set_xlim(0, max_r)
    fig.tight_layout()

    fname = os.path.join(
        out_dir, f"radial_{label}_{e_lo:.1f}-{e_hi:.1f}keV.png")
    fig.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Radial profile plot  -> {fname}")


# ---------------------------------------------------------------------------
# Exposure-map histogram
# ---------------------------------------------------------------------------

def exposure_histogram(meta, exp_stats, label, cfg, out_dir):
    """
    Save a histogram of exposure-map pixel values inside the source aperture.

    Parameters
    ----------
    meta      : dict   — from compute_exposure_stats()
    exp_stats : dict   — {'median': float, 'mean': float, 'psf_weighted': float}
    label     : str    — instrument label, e.g. 'FPMA', 'MOS1', 'PN'
    cfg       : Config or XMMConfig
    out_dir   : str
    """
    vals = meta['exp_values']
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(vals / 1e3, bins=30, color='steelblue',
            edgecolor='white', linewidth=0.5, alpha=0.8,
            label='Exposure-map pixels in aperture')

    styles = {
        'median':       ('tomato',     '--', 'Median'),
        'mean':         ('darkorange', '--', 'Mean'),
        'psf_weighted': ('purple',     ':',  'PSF-wtd mean'),
    }
    for key, (col, ls, lbl) in styles.items():
        tag = '  [PRIMARY]' if key == cfg.exp_stat else ''
        ax.axvline(exp_stats[key] / 1e3, color=col, ls=ls, lw=1.8,
                   label=f"{lbl} = {exp_stats[key]/1e3:.2f} ks{tag}")

    ax.set_xlabel('Exposure time (ks)')
    ax.set_ylabel('Number of pixels')
    ax.set_title(
        f'{label} — Exposure-map distribution in '
        f'{cfg.src_radius_arcsec:.0f}" aperture')
    ax.legend()
    fig.tight_layout()

    fname = os.path.join(out_dir, f"expmap_hist_{label}.png")
    fig.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Exposure histogram   -> {fname}")


# ---------------------------------------------------------------------------
# Region image — source aperture + background annulus on event sky map
# ---------------------------------------------------------------------------

def _fmt_ra(ra_deg):
    """Format RA in degrees as HH:MM:SS.s string."""
    from astropy.coordinates import Angle
    import astropy.units as u
    return Angle(ra_deg, u.deg).to_string(unit=u.hour, sep=':', precision=1, pad=True)


def _fmt_dec(dec_deg):
    """Format Dec in degrees as ±DD:MM:SS string."""
    from astropy.coordinates import Angle
    import astropy.units as u
    return Angle(dec_deg, u.deg).to_string(sep=':', precision=0,
                                            pad=True, alwayssign=True)


def region_image(evt_x, evt_y, cx_evt, cy_evt, pscale_evt,
                 label, e_lo, e_hi, obsid, cfg, out_dir,
                 src_ra_deg=None, src_dec_deg=None,
                 bkg_cx_evt=None, bkg_cy_evt=None):
    """
    Save a raw event count map with source and background regions overlaid.
    Saved as PNG (300 dpi) and PDF (vector, no dpi limit) for paper inclusion.

    No smoothing — individual photons visible as-is.

    Parameters
    ----------
    src_ra_deg, src_dec_deg : float, optional
        Source sky position.  When provided, axes are labelled in RA/Dec
        (J2000) instead of arcsec offsets.
    bkg_cx_evt, bkg_cy_evt : float, optional
        Background-region centre in event pixel coordinates.  When different
        from the source centre (manual bkg_mode), a separate circle is drawn
        at this position with no inner annulus.  Defaults to source centre.
    """
    # Default background centre = source centre (annulus mode)
    if bkg_cx_evt is None:
        bkg_cx_evt = cx_evt
    if bkg_cy_evt is None:
        bkg_cy_evt = cy_evt
    bkg_separate = (abs(bkg_cx_evt - cx_evt) > 1.0 or
                    abs(bkg_cy_evt - cy_evt) > 1.0)

    # ---- Build raw event image ----------------------------------------------
    # Centre the image on the source; expand view to include bkg if separate
    all_cx = [cx_evt, bkg_cx_evt]
    all_cy = [cy_evt, bkg_cy_evt]
    pad_arcsec = cfg.bkg_radius_arcsec * 1.4
    if bkg_separate:
        max_sep = max(abs(bkg_cx_evt - cx_evt), abs(bkg_cy_evt - cy_evt))
        pad_arcsec = max(pad_arcsec,
                         (max_sep + cfg.bkg_radius_arcsec / pscale_evt)
                         * pscale_evt * 1.2)
    pad_pix = pad_arcsec / pscale_evt

    x_lo = cx_evt - pad_pix;  x_hi = cx_evt + pad_pix
    y_lo = cy_evt - pad_pix;  y_hi = cy_evt + pad_pix

    n_bins = 300
    img, _, _ = np.histogram2d(
        evt_x, evt_y,
        bins=[np.linspace(x_lo, x_hi, n_bins + 1),
              np.linspace(y_lo, y_hi, n_bins + 1)])
    img = img.T   # rows = Y, cols = X

    # Arcsec offsets from source centre (used for circle positions + RA/Dec)
    as_lo_x = (x_lo - cx_evt) * pscale_evt
    as_hi_x = (x_hi - cx_evt) * pscale_evt
    as_lo_y = (y_lo - cy_evt) * pscale_evt
    as_hi_y = (y_hi - cy_evt) * pscale_evt
    extent  = [as_lo_x, as_hi_x, as_lo_y, as_hi_y]

    # Background centre in arcsec offset from source centre
    bkg_dx_as = (bkg_cx_evt - cx_evt) * pscale_evt
    bkg_dy_as = (bkg_cy_evt - cy_evt) * pscale_evt

    # ---- Figure -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6.5, 6.0))

    vmax = max(1, np.percentile(img[img > 0], 99)) if img.any() else 1
    ax.imshow(img, origin='lower', extent=extent,
              cmap='viridis', aspect='equal',
              interpolation='nearest', vmin=0, vmax=vmax)

    # Source circle + crosshair
    r_src = cfg.src_radius_arcsec
    ax.add_patch(Circle((0, 0), r_src,
                         edgecolor='tomato', facecolor='none',
                         linewidth=1.8, linestyle='--',
                         label=f'Source ({r_src:.0f}")'))
    ch = r_src * 0.15
    ax.plot([-ch, ch], [0, 0], color='tomato', lw=0.8)
    ax.plot([0, 0],    [-ch, ch], color='tomato', lw=0.8)

    # Background region
    r_out = cfg.bkg_radius_arcsec
    if bkg_separate:
        # Manual mode: single circle at background centre, no inner annulus
        ax.add_patch(Circle((bkg_dx_as, bkg_dy_as), r_out,
                             edgecolor='orange', facecolor='none',
                             linewidth=1.5, linestyle='-',
                             label=f'Background ({r_out:.0f}")'))
        ax.plot([bkg_dx_as], [bkg_dy_as], '+', color='orange',
                markersize=8, markeredgewidth=1.2)
    else:
        # Annulus mode: inner + outer circles around source
        r_in = r_src * cfg.bkg_inner_factor
        ax.add_patch(Circle((0, 0), r_out,
                             edgecolor='orange', facecolor='none',
                             linewidth=1.5, linestyle='-',
                             label=f'Bkg outer ({r_out:.0f}")'))
        ax.add_patch(Circle((0, 0), r_in,
                             edgecolor='orange', facecolor='none',
                             linewidth=1.2, linestyle=':',
                             label=f'Bkg inner ({r_in:.0f}")'))

    ax.legend(loc='upper right', framealpha=0.8, fontsize=8)
    ax.set_title(
        f'{label}  |  {e_lo:.1f}–{e_hi:.1f} keV  |  OBSID {obsid}',
        fontsize=10)

    # ---- Axes: RA/Dec labels when source position is known ------------------
    if src_ra_deg is not None and src_dec_deg is not None:
        cos_dec = np.cos(np.radians(src_dec_deg))

        # Let matplotlib choose tick positions in arcsec, then relabel as RA/Dec
        ax.set_xlabel('RA (J2000)')
        ax.set_ylabel('Dec (J2000)')
        fig.tight_layout()          # fix layout before reading tick positions
        fig.canvas.draw()           # force tick computation

        x_ticks = ax.get_xticks()
        y_ticks = ax.get_yticks()

        # Only label ticks inside the image extent
        x_ticks = x_ticks[(x_ticks >= as_lo_x) & (x_ticks <= as_hi_x)]
        y_ticks = y_ticks[(y_ticks >= as_lo_y) & (y_ticks <= as_hi_y)]

        ax.set_xticks(x_ticks)
        ax.set_xticklabels(
            [_fmt_ra(src_ra_deg - dx / (3600.0 * cos_dec)) for dx in x_ticks],
            rotation=30, ha='right', fontsize=8)

        ax.set_yticks(y_ticks)
        ax.set_yticklabels(
            [_fmt_dec(src_dec_deg + dy / 3600.0) for dy in y_ticks],
            fontsize=8)
    else:
        ax.set_xlabel('$\\Delta X$ (arcsec)')
        ax.set_ylabel('$\\Delta Y$ (arcsec)')

    fig.tight_layout()

    stem = f"regions_{label}_{e_lo:.1f}-{e_hi:.1f}keV"

    png_path = os.path.join(out_dir, f"{stem}.png")
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    print(f"  Region image (png)   -> {png_path}")

    pdf_path = os.path.join(out_dir, f"{stem}.pdf")
    fig.savefig(pdf_path, bbox_inches='tight')       # vector — no dpi
    print(f"  Region image (pdf)   -> {pdf_path}")

    plt.close(fig)
