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

OBSERVATORY = 'chandra'    # 'nustar' | 'xmm' | 'swift' | 'chandra'  ← set this first

# ---------------------------------------------------------------------------
# NuSTAR settings  (used when OBSERVATORY = 'nustar')
# ---------------------------------------------------------------------------

NUSTAR = dict(
    base_path         = "/Users/sanjanagupta/Documents/data/NuSTAR/2012ap/",
    obsid             = ["80802504004", "80802504002"],
                        # or: obsid = ["80802504004", "80802504006"]
                        # Co-adding sums counts + exposures and gives one
                        # combined upper limit with individual per-obs rows too.
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
    gui_per_obs       = True,  # True: independent GUI for EACH observation —
                                #   use if pointings differ or source falls
                                #   at the edge / off-chip in some obs.
                                # False (default): GUI for first obs only,
                                #   regions carried to all subsequent obs.
    save_plots        = True,
)

# ---------------------------------------------------------------------------
# XMM-Newton settings  (used when OBSERVATORY = 'xmm')
# ---------------------------------------------------------------------------

XMM = dict(
    data_dir          = "/Users/sanjanagupta/Documents/data/XMM/2012ap_/ODF",             # ODF working directory (epproc/emproc output)
    obsid             = "0881990901",
                        # or: obsid = ["0881990901", "0881990902"]

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
    gui_per_obs       = False,  # True: independent GUI for EACH observation —
                                #   use if pointings differ or source falls
                                #   at the edge / off-chip in some obs.
                                # False (default): GUI for first obs only,
                                #   regions carried to all subsequent obs.
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

    obsid             = ["03000397004","03000397002"],
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

    psf_file          = "",   # leave empty — bundled psfconst_xrt.fits is used
                              # (identical to HEASoft 6.36 / XIMAGE calibration).
                              # Set to an absolute path only if you have a newer
                              # or custom PSF coefficient file to use instead.

    confidence_levels = [0.9545, 0.9973],   # ~2σ and ~3σ

    use_gui           = True,   # True: interactive region selector (requires display)
    gui_per_obs       = True,  # True: independent GUI for EACH observation —
                                #   use if pointings differ or source falls
                                #   at the edge / off-chip in some obs.
                                # False (default): GUI for first obs only,
                                #   regions shared with all others.
    save_plots        = True,
)

# ---------------------------------------------------------------------------
# Chandra ACIS settings  (used when OBSERVATORY = 'chandra')
# ---------------------------------------------------------------------------
# Requires CIAO to be initialised in the shell:
#   conda activate ciao-4.XX
#   source $ASCDS_INSTALL/bin/ciao.sh   (if needed)

CHANDRA = dict(
    base_path         = "/Users/sanjanagupta/Documents/data/Chandra/2022xxf",
                        # Directory containing one sub-folder per obsid.
                        # Each obsid folder should hold the standard CDA layout
                        # (primary/ and secondary/).  chandra_repro is run
                        # automatically if obsid/repro/ does not exist yet.

    obsid             = "26631",
                        # or: obsid = ["26631", "26632"]  — co-adds both obs

    ra                = "11:30:05.94",   # "HH:MM:SS.ss" or decimal degrees
    dec               = "+09:16:57.37",  # "±DD:MM:SS.ss" or decimal degrees

    src_radius_arcsec = 5.0,    # Chandra on-axis FWHM ≈ 0.5–1"; typical 2–10"
    bkg_radius_arcsec = 15.0,   # outer radius of background annulus
    bkg_inner_factor  = 1.0,    # inner radius = src_radius * this

    psf_fwhm_arcsec   = 0.9,    # on-axis ACIS; increase for off-axis sources

    energy_band       = (0.3, 10.0),  # 'broad' (0.5–7) | 'soft' (0.5–2) |
                                  # 'medium' (2–4) | 'hard' (4–7) |
                                  # 'full' (0.5–10) | 'ultrasoft' (0.3–1) |
                                  # or custom tuple e.g. (1.0, 6.0)

    bkg_mode          = 'annulus',  # 'annulus' or 'manual'
    bkg_ra            = "",         # only used if bkg_mode = 'manual'
    bkg_dec           = "",

    confidence_levels = [0.68, 0.9545, 0.9973],   # ~2σ and ~3σ

    ciao_prefix       = "/Applications/ciao-4.18",     # Leave empty — CIAO conda env is auto-detected.
                                # Set explicitly only if auto-detection fails, e.g.
                                #   ciao_prefix = "~/opt/miniconda3/envs/ciao-4.16"

    run_repro         = True,   # auto-run chandra_repro if repro/ not found

    use_aprates       = True,   # True: CIAO aprates (primary, recommended)
                                # False: pure-Python Kraft/Gehrels only

    use_gui           = True,   # True: interactive region selector
    gui_per_obs       = False,  # True: independent GUI for each obs
                                # False (default): GUI for first obs only
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

    elif OBSERVATORY == 'chandra':
        from xray_uplim.chandra import run_uplim
        run_uplim(**CHANDRA)

    else:
        raise ValueError(
            f"Unknown OBSERVATORY '{OBSERVATORY}'. "
            "Choose 'nustar', 'xmm', 'swift', or 'chandra'.")
