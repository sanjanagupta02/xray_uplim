"""
xray_uplim.xmm.io
-----------------
File discovery and loading utilities for XMM-Newton EPIC data.

Expected directory layout (ODF working directory after SAS processing)
-----------------------------------------------------------------------
Events (from epproc / emproc):
    *{obsid}*EMOS1*ImagingEvts.ds   — MOS1
    *{obsid}*EMOS2*ImagingEvts.ds   — MOS2
    *{obsid}*EPN*ImagingEvts.ds     — PN

Exposure maps (from eexpmap; see pipeline guide in config.py):
    mos1_expmap.fits  /  mos2_expmap.fits  /  pn_expmap.fits

The locate_files() function searches for both naming patterns and falls
back gracefully.  When multiple matches exist (e.g. S+U exposures),
scheduled (_S_) exposures are preferred over unscheduled (_U_).

Public API
----------
locate_files(data_dir, obsid_str, instrument, cfg)   → (evt_path, exp_path)
load_events(cfg, evt_path, instrument)
    → (events_table, header, pi_lo, pi_hi)
load_expmap(exp_path)           → (exp_array, exp_header)
"""

import os
import glob
import warnings
import numpy as np
from astropy.io import fits
from astropy.table import Table

from .config import XMMConfig


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _prefer_scheduled(paths):
    """
    Given a list of file paths, return those containing '_S_' (scheduled
    exposure) if any exist; otherwise return the full list.
    Raises FileNotFoundError if the input list is empty.
    """
    if not paths:
        raise FileNotFoundError("No matching files found.")
    scheduled = [p for p in paths if '_S_' in os.path.basename(p)]
    return sorted(scheduled if scheduled else paths)


def _glob_first(pattern, label):
    """
    Return the first match for *pattern* (preferring scheduled exposures),
    or raise FileNotFoundError with a helpful message.
    """
    matches = glob.glob(pattern)
    if not matches:
        raise FileNotFoundError(
            f"No {label} file found matching pattern:\n  {pattern}\n"
            "Check that data_dir is set to the correct ODF working directory "
            "and that SAS processing has been run (epproc/emproc, eexpmap).")
    return _prefer_scheduled(matches)[0]


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def locate_files(data_dir: str, obsid_str: str, instrument: str, cfg: XMMConfig):
    """
    Locate the cleaned event file and exposure map for one EPIC instrument.

    Search strategy
    ---------------
    Event file  : glob for *{instrume_key}*ImagingEvts.ds in data_dir,
                  with optional obsid in the pattern for narrower matching.
    Exposure map: first looks for the canonical names written by the SAS
                  guide (mos1_expmap.fits, etc.); then tries broader globs.

    Parameters
    ----------
    data_dir   : str   — directory to search in (obs-specific for multi-obsid)
    obsid_str  : str   — single obsid string for file globbing
    instrument : 'MOS1', 'MOS2', or 'PN'
    cfg        : XMMConfig

    Returns
    -------
    evt_path : str   — absolute path to cleaned event file
    exp_path : str   — absolute path to exposure map
    """
    instrume_key = cfg.INSTRUME_KEYS[instrument]   # 'EMOS1', 'EMOS2', 'EPN'

    # -- Event file -----------------------------------------------------------
    # Try obsid-scoped pattern first, then fall back to broader pattern
    if obsid_str:
        evt_pattern_narrow = os.path.join(
            data_dir, f'*{obsid_str}*{instrume_key}*ImagingEvts.ds')
        matches = glob.glob(evt_pattern_narrow)
    else:
        matches = []

    if not matches:
        # Broader: any event file for this instrument
        evt_pattern_broad = os.path.join(
            data_dir, f'*{instrume_key}*ImagingEvts.ds')
        matches = glob.glob(evt_pattern_broad)

    if not matches:
        raise FileNotFoundError(
            f"No ImagingEvts.ds file found for {instrument} "
            f"(INSTRUME={instrume_key}) in:\n  {data_dir}\n"
            "Run epproc (PN) or emproc (MOS) in SAS first.\n"
            "Expected filename contains: "
            f"'{instrume_key}' and 'ImagingEvts.ds'.")

    evt_path = _prefer_scheduled(matches)[0]

    # -- Exposure map ---------------------------------------------------------
    # Canonical names from the SAS preprocessing guide
    canonical = {
        'MOS1': 'mos1_expmap.fits',
        'MOS2': 'mos2_expmap.fits',
        'PN'  : 'pn_expmap.fits',
    }
    exp_canonical = os.path.join(data_dir, canonical[instrument])
    if os.path.isfile(exp_canonical):
        exp_path = exp_canonical
    else:
        # Fall back: look for any exposure map file matching the instrument key
        inst_lower = instrument.lower().replace('mos', 'mos')
        fallback_patterns = [
            os.path.join(data_dir, f'*{instrume_key}*expmap*.fits'),
            os.path.join(data_dir, f'*{instrume_key}*expmap*.fits.gz'),
            os.path.join(data_dir, f'*expmap*{instrume_key}*.fits'),
            os.path.join(data_dir, f'expmap_{inst_lower.lower()}.fits'),
        ]
        found = []
        for pat in fallback_patterns:
            found.extend(glob.glob(pat))
        if not found:
            raise FileNotFoundError(
                f"No exposure map found for {instrument} in:\n  {data_dir}\n"
                f"Looked for '{canonical[instrument]}' and fallback patterns.\n"
                "Run eexpmap in SAS first.  See the SAS guide in "
                "xray_uplim/xmm/config.py for the recommended command.")
        exp_path = sorted(found)[0]

    return evt_path, exp_path


# ---------------------------------------------------------------------------
# Event file loading
# ---------------------------------------------------------------------------

def load_events(cfg: XMMConfig, evt_path: str, instrument: str):
    """
    Load and filter a cleaned XMM EPIC event file.

    Filters applied
    ---------------
    - PATTERN  <= PATTERN_LIMITS[instrument]  (12 for MOS, 4 for PN)
    - FLAG     == 0                           (no bad pixels / CCD edges)
    - PI       in [pi_lo, pi_hi]              (energy band)

    Parameters
    ----------
    cfg        : XMMConfig
    evt_path   : str   — path returned by locate_files()
    instrument : 'MOS1', 'MOS2', or 'PN'

    Returns
    -------
    events  : astropy.table.Table  — filtered events (rows: photons)
    hdr     : astropy.io.fits.Header  — EVENTS extension header
    pi_lo   : int   — lower PI channel bound (inclusive)
    pi_hi   : int   — upper PI channel bound (inclusive)

    Notes
    -----
    The EVENTS extension is usually extension 1 ('EVENTS').  If that
    fails, the function tries the primary HDU and then walks all
    extensions looking for an EVENTS table.
    """
    # Resolve energy band → PI channels
    e_lo_kev, e_hi_kev = cfg.resolve_energy_band()
    pi_lo, pi_hi = cfg.energy_to_pi(e_lo_kev, e_hi_kev)

    pat_limit = cfg.PATTERN_LIMITS[instrument]

    # -- Open event file and find EVENTS extension ----------------------------
    with fits.open(evt_path, memmap=True) as hdul:
        hdr = _find_events_header(hdul, evt_path)
        ext_name = hdr.get('EXTNAME', 'EVENTS').strip()

        # Verify this is the right instrument
        instrume = hdr.get('INSTRUME', '').strip().upper()
        expected = cfg.INSTRUME_KEYS[instrument].upper()
        if instrume and instrume != expected:
            warnings.warn(
                f"Event file INSTRUME='{instrume}' does not match expected "
                f"'{expected}' for instrument '{instrument}'.  "
                "Proceeding anyway — check your file paths.",
                RuntimeWarning, stacklevel=2)

        # Load the table from the correct extension
        tbl = Table.read(evt_path, hdu=ext_name)

    # -- Sanity-check required columns ----------------------------------------
    for col in ('PATTERN', 'FLAG', 'PI', 'X', 'Y'):
        if col not in tbl.colnames:
            raise RuntimeError(
                f"Required column '{col}' not found in {evt_path}.\n"
                f"Available columns: {tbl.colnames}")

    # -- Apply filters ---------------------------------------------------------
    n_raw = len(tbl)

    mask = (
        (tbl['PATTERN'] <= pat_limit) &
        (tbl['FLAG']    == 0)         &
        (tbl['PI']      >= pi_lo)     &
        (tbl['PI']      <= pi_hi)
    )
    events = tbl[mask]
    n_filt = len(events)

    if n_filt == 0:
        warnings.warn(
            f"{instrument}: 0 events remain after filtering "
            f"(PATTERN<={pat_limit}, FLAG==0, PI=[{pi_lo},{pi_hi}]) "
            f"from {n_raw} raw events.  "
            "Check energy band, event file quality, and PI range.",
            RuntimeWarning, stacklevel=2)
    else:
        print(f"  {instrument}: {n_raw:,d} raw events → "
              f"{n_filt:,d} after filtering "
              f"(PATTERN<={pat_limit}, FLAG==0, "
              f"PI=[{pi_lo}:{pi_hi}] = "
              f"{e_lo_kev:.2f}–{e_hi_kev:.2f} keV)")

    return events, hdr, pi_lo, pi_hi


def _find_events_header(hdul, evt_path):
    """
    Return the header of the EVENTS binary table extension.

    Try in order: extension named 'EVENTS', extension 1, then all
    extensions looking for XTENSION='BINTABLE' with NAXIS2 > 0.
    """
    # Preferred: named extension
    for name in ('EVENTS', 'STDGTI'):
        try:
            return hdul['EVENTS'].header
        except KeyError:
            pass

    # Extension 1 is standard for XMM event files
    if len(hdul) > 1 and hasattr(hdul[1], 'header'):
        ext1_type = hdul[1].header.get('XTENSION', '').strip()
        if ext1_type == 'BINTABLE':
            return hdul[1].header

    # Walk all extensions
    for hdu in hdul:
        hdr = hdu.header
        if (hdr.get('XTENSION', '').strip() == 'BINTABLE' and
                hdr.get('NAXIS2', 0) > 0):
            return hdr

    raise RuntimeError(
        f"Cannot find EVENTS binary table extension in: {evt_path}")


# ---------------------------------------------------------------------------
# Exposure map loading
# ---------------------------------------------------------------------------

def load_expmap(exp_path: str):
    """
    Load an XMM EPIC exposure map produced by SAS eexpmap.

    The exposure map is a standard FITS image with units of seconds.
    Pixels with zero exposure are retained as-is; callers should mask
    them when computing background rates.

    Parameters
    ----------
    exp_path : str   — path returned by locate_files()

    Returns
    -------
    exp_data : numpy.ndarray  — 2-D float64 array, units = seconds
    exp_hdr  : astropy.io.fits.Header  — primary image header (WCS etc.)

    Notes
    -----
    eexpmap always writes the map as the primary HDU (IMAGE extension).
    If a multi-extension file is encountered the function tries the
    first IMAGE extension before falling back to the primary HDU.
    """
    with fits.open(exp_path) as hdul:
        # Primary HDU
        primary = hdul[0]
        if primary.data is not None and primary.data.ndim == 2:
            exp_data = primary.data.astype(np.float64)
            exp_hdr  = primary.header
        else:
            # Walk extensions looking for a 2-D image
            exp_data = exp_hdr = None
            for hdu in hdul[1:]:
                if (hasattr(hdu, 'data') and hdu.data is not None and
                        hdu.data.ndim == 2):
                    exp_data = hdu.data.astype(np.float64)
                    exp_hdr  = hdu.header
                    break
            if exp_data is None:
                raise RuntimeError(
                    f"No 2-D image found in exposure map: {exp_path}")

    # Sanity check
    if exp_data.max() <= 0:
        warnings.warn(
            f"Exposure map appears to be all zeros: {exp_path}.\n"
            "Check that eexpmap ran successfully and wrote the correct file.",
            RuntimeWarning, stacklevel=2)

    print(f"  Exposure map loaded: {exp_data.shape[1]}×{exp_data.shape[0]} pix, "
          f"max={exp_data.max():.0f} s, "
          f"median (nonzero)={np.median(exp_data[exp_data > 0]):.0f} s")

    return exp_data, exp_hdr
