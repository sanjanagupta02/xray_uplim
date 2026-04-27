#!/usr/bin/env python
"""
run_uplim.py
------------
Command-line entry point for xray_uplim.

Edit the CONFIG block below and run:
    python run_uplim.py

Or import and call from a notebook:
    from xray_uplim.nustar import run_uplim
    run_uplim(base_path=..., obsid=..., ra=..., dec=...)
"""

# =============================================================================
# CONFIG  — edit this block
# =============================================================================

OBSERVATORY = 'nustar'    # 'nustar' | 'xmm'  ← set this first

# ---------------------------------------------------------------------------
# NuSTAR settings  (used when OBSERVATORY = 'nustar')
# ---------------------------------------------------------------------------

NUSTAR = dict(
    base_path         = "/Users/sanjanagupta/Documents/data/NuSTAR/2012ap/",
    obsid             = "80802504004",
    caldb_dir         = "/Users/sanjanagupta/Documents/software/caldb",

    ra                = "05:00:13.721",     # "HH:MM:SS.ss" or decimal degrees
    dec               = "-03:20:51.22",     # "±DD:MM:SS.ss" or decimal degrees

    src_radius_arcsec = 60.0,   # NuSTAR EEF: ~50% at 20", ~60% at 30", ~80% at 60"
    bkg_radius_arcsec = 200.0,  # outer radius of background annulus
    bkg_inner_factor  = 1.2,    # inner radius = src_radius * this

    psf_fwhm_arcsec   = 18.0,   # Harrison+13; increase for off-axis sources

    energy_band       = '(8.0, 24.5)',  # 'full' (3–79 keV) | 'soft' (4.5–6) |
                                        # 'iron' (6–8) | 'medium' (8–12) |
                                        # 'hard' (12–20) | 'ultra-hard' (20–79) |
                                        # or custom tuple e.g. (8.0, 24.0)

    modules           = ['A', 'B'],

    bkg_mode          = 'annulus',  # 'annulus' or 'manual'
    bkg_ra            = "",         # only used if bkg_mode = 'manual'
    bkg_dec           = "",

    exp_stat          = 'median',   # 'median' | 'mean' | 'psf_weighted'

    psf_gamma         = 2.0,        # photon index for PSF spectral weighting
                                    # 2.0 = soft source prior; 1.7 = harder; 0.0 = flat

    confidence_levels = [0.9545, 0.9973],   # ~2σ and ~3σ

    use_gui           = True,   # True: interactive region selector (requires display)
    save_plots        = True,
)

# ---------------------------------------------------------------------------
# XMM-Newton settings  (used when OBSERVATORY = 'xmm')
# ---------------------------------------------------------------------------

XMM = dict(
    data_dir          = "/Users/sanjanagupta/Documents/data/XMM/2012ap_/ODF",             # ODF working directory (epproc/emproc output)
    obsid             = "0881990901",             # e.g. '0881990901'

    ra                = "05:00:13.721",             # "HH:MM:SS.ss" or decimal degrees
    dec               = "-03:20:51.22",             # "±DD:MM:SS.ss" or decimal degrees

    src_radius_arcsec = 20.0,   # MOS on-axis FWHM ~4.5", PN ~6"; typical 15–20"
    bkg_radius_arcsec = 60.0,
    bkg_inner_factor  = 1.5,
    psf_fwhm_arcsec   = 5.0,   # used for PSF-weighted exposure diagnostic only
                                # MOS on-axis ~4.5", PN on-axis ~6"

    energy_band       = 'full', # 'full' (0.5–10) | 'soft' (0.5–2) | 'hard' (2–10) |
                                # 'medium' (1–2) | 'ultrasoft' (0.2–0.5) |
                                # or custom tuple e.g. (0.5, 7.0)

    instruments       = ['MOS1', 'MOS2', 'PN'],  # any subset

    bkg_mode          = 'annulus',  # 'annulus' or 'manual'
    bkg_ra            = "",
    bkg_dec           = "",

    exp_stat          = 'median',

    confidence_levels = [0.9545, 0.9973],

    psf_dir           = "",     # path to directory containing XRT[1-3]_XPSF_*.CCF files
                                # leave empty if you have copied them to xray_uplim/data/xmm/psf/
                                # download from: https://www.cosmos.esa.int/web/xmm-newton/current-calibration-files

    use_gui           = True,
    save_plots        = True,
)

# =============================================================================

if __name__ == "__main__":
    if OBSERVATORY == 'nustar':
        from xray_uplim.nustar import run_uplim
        run_uplim(**NUSTAR)

    elif OBSERVATORY == 'xmm':
        from xray_uplim.xmm import run_uplim
        run_uplim(**XMM)

    else:
        raise ValueError(
            f"Unknown OBSERVATORY '{OBSERVATORY}'. "
            "Choose 'nustar' or 'xmm'.")
