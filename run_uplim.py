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

OBSERVATORY = 'swift'    # 'nustar' | 'xmm' | 'swift'  ← set this first

# ---------------------------------------------------------------------------
# NuSTAR settings  (used when OBSERVATORY = 'nustar')
# ---------------------------------------------------------------------------

NUSTAR = dict(
    base_path         = "/Users/sanjanagupta/Documents/data/NuSTAR/2012ap/",
    obsid             = "80802504004",
    caldb_dir         = "/Users/sanjanagupta/Documents/software/caldb",  # or "" to use $CALDB

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

    psf_dir           = "",     # leave empty — $SAS_CCFPATH is checked automatically.
                                # if SAS is initialised in your shell, no change needed here.
                                # or set explicitly: psf_dir = "/path/to/dir/with/XRT?_XPSF_*.CCF"
                                # download: https://www.cosmos.esa.int/web/xmm-newton/current-calibration-files

    use_gui           = True,
    save_plots        = True,
)

# ---------------------------------------------------------------------------
# Swift XRT settings  (used when OBSERVATORY = 'swift')
# ---------------------------------------------------------------------------

SWIFT = dict(
    data_dir          = "/Users/sanjanagupta/Documents/data/Swift/raw_data/2021bmf/",
                        # Parent directory containing one sub-folder per obsid.
                        # The pipeline constructs: <data_dir>/<obsid>/xrt/event/
                        # (same convention as NuSTAR base_path / obsid.)

    obsid             = "03000397004",
                        # Single observation:  obsid = "03000397004"
                        # Multiple co-added:   obsid = ["03000397001",
                        #                               "03000397002",
                        #                               "03000397004"]
                        # Co-adding sums counts + exposures, then gives one
                        # combined upper limit — ideal for short Swift exposures.

    ra                = "16:33:29.416",   # "HH:MM:SS.ss" or decimal degrees
    dec               = "-06:22:49.51",   # "±DD:MM:SS.ss" or decimal degrees

    src_radius_arcsec = 20.0,   # XRT on-axis FWHM ~5–6"; typical aperture 20–30"
    bkg_radius_arcsec = 80.0,   # outer radius of background annulus
    bkg_inner_factor  = 1.5,    # inner radius = src_radius * this

    psf_fwhm_arcsec   = 6.0,    # on-axis XRT FWHM; used for PSF-weighted exposure only

    energy_band       = 'full', # 'full' (0.3–10) | 'soft' (0.3–1.5) | 'hard' (1.5–10) |
                                # 'ultrasoft' (0.3–1.0) | or custom tuple e.g. (0.5, 7.0)

    bkg_mode          = 'annulus',  # 'annulus' or 'manual'
    bkg_ra            = "",         # only used if bkg_mode = 'manual'
    bkg_dec           = "",

    exp_stat          = 'median',   # 'median' | 'mean' | 'psf_weighted'

    caldb_dir         = "",   # leave empty — $CALDB is checked automatically, then
                              # bundled psfconst_xrt.fits is used as fallback.
                              # or set explicitly: caldb_dir = "/path/to/caldb"
                              # (file expected at <caldb_dir>/data/swift/xrt/cpf/psf/)

    confidence_levels = [0.9545, 0.9973],   # ~2σ and ~3σ

    use_gui           = True,   # True: interactive region selector (requires display)
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

    elif OBSERVATORY == 'swift':
        from xray_uplim.swift import run_uplim
        run_uplim(**SWIFT)

    else:
        raise ValueError(
            f"Unknown OBSERVATORY '{OBSERVATORY}'. "
            "Choose 'nustar', 'xmm', or 'swift'.")
