"""
xray_uplim.swift.io
-------------------
File location and loading for Swift XRT data.

locate_files(cfg)
    Finds the cleaned event file and exposure map for the given obsid.
    Searches the standard archive subdirectory layout automatically:
        <data_dir>/xrt/event/    ← cleaned event files
        <data_dir>/xrt/products/ ← exposure maps and sky images
    Mode (PC / WT) is auto-detected from what files are present;
    PC is preferred when both exist.

load_events(cfg, evt_file, mode)
    Opens a Swift XRT event file, applies energy (PI) and grade filters,
    and returns the filtered event table and header.

load_expmap(exp_path)
    Loads a Swift XRT exposure map and returns the data array and header.

Swift XRT archive layout (standard HEASARC download)
-----------------------------------------------------
<obsid>/
  xrt/
    event/
      sw{obsid}xpcw3po_cl.evt.gz   ← PC cleaned events
      sw{obsid}xwtw2st_cl.evt.gz   ← WT steady-tracking cleaned events
      sw{obsid}xwtw2sl_cl.evt.gz   ← WT settling-mode cleaned events
    products/
      sw{obsid}xpc_ex.img.gz       ← PC exposure map
      sw{obsid}xwt_ex.img.gz       ← WT exposure map (if present)
      sw{obsid}xpc_sk.img.gz       ← PC sky image (not used by pipeline)

All files may be gzipped (.gz) or plain — both are tried.
astropy.io.fits reads .gz files transparently.

WT mode notes
-------------
Two WT cleaned files can be present:
  xwtw2st_cl  — steady-tracking (preferred; longer exposure, better attitude)
  xwtw2sl_cl  — settling mode   (spacecraft is still slewing; use as fallback)
We prefer 'st' over 'sl'. For very bright sources both may be combined by
the standard pipeline, but for upper-limit work 'st' alone is sufficient.
"""

import os
import glob
import warnings
import numpy as np
from astropy.io import fits
from astropy.table import Table


# ---------------------------------------------------------------------------
# Subdirectory layout within the observation root
# ---------------------------------------------------------------------------
_EVENT_SUBDIR   = os.path.join('xrt', 'event')
_PRODUCT_SUBDIR = os.path.join('xrt', 'products')

# Glob patterns for cleaned event files — tried in order, first match wins.
# Each entry is tried with and without the .gz suffix.
_EVT_GLOBS = {
    'PC': [
        'sw{obsid}xpcw3po_cl.evt',
        'sw{obsid}xpc*po_cl.evt',
        'sw{obsid}xpc*cl.evt',
    ],
    'WT': [
        # Prefer steady-tracking ('st') over settling ('sl')
        'sw{obsid}xwtw2st_cl.evt',
        'sw{obsid}xwt*st_cl.evt',
        'sw{obsid}xwtw2sl_cl.evt',
        'sw{obsid}xwt*sl_cl.evt',
        'sw{obsid}xwt*cl.evt',
    ],
}

# Glob patterns for exposure maps
_EXP_GLOBS = {
    'PC': [
        'sw{obsid}xpc_ex.img',
        'sw{obsid}xpcw3po_ex.img',
        'sw{obsid}xpc*ex.img',
    ],
    'WT': [
        'sw{obsid}xwt_ex.img',
        'sw{obsid}xwtw2st_ex.img',
        'sw{obsid}xwt*ex.img',
    ],
}


def _glob_first(directory, patterns, obsid):
    """
    Return the first file found matching any pattern (with or without .gz).
    Returns None if nothing found.
    """
    for pat in patterns:
        pat = pat.format(obsid=obsid)
        # Try plain, then gzipped
        for suffix in ('', '.gz'):
            matches = sorted(glob.glob(os.path.join(directory, pat + suffix)))
            if matches:
                return matches[-1]   # newest if multiple
    return None


def locate_files(obs_root, obsid, cfg):
    """
    Locate the Swift XRT cleaned event file and exposure map.

    Searches <obs_root>/xrt/event/ for event files and
    <obs_root>/xrt/products/ for exposure maps.

    Mode (PC / WT) is auto-detected from what is present;
    PC is preferred (typical for faint sources / upper limits).

    Parameters
    ----------
    obs_root : str   — observation root directory (e.g. cfg.data_dir / obsid)
    obsid    : str   — observation ID, used for file-name glob patterns
    cfg      : SwiftConfig

    Returns
    -------
    evt_path : str
    exp_path : str or None   (None if exposure map not found — pipeline warns)
    mode     : str           'PC' or 'WT'
    """
    evt_dir = os.path.join(obs_root, _EVENT_SUBDIR)
    exp_dir = os.path.join(obs_root, _PRODUCT_SUBDIR)

    if not os.path.isdir(evt_dir):
        # Fallback: obs_root already points directly at the data
        evt_dir = obs_root
        exp_dir = obs_root

    for mode in ('PC', 'WT'):
        evt_path = _glob_first(evt_dir, _EVT_GLOBS[mode], obsid)
        if evt_path is None:
            continue

        exp_path = _glob_first(exp_dir, _EXP_GLOBS[mode], obsid)
        if exp_path is None:
            warnings.warn(
                f"Found {mode} event file but no exposure map in {exp_dir!r}.\n"
                "Run xrtexpomap to generate it.",
                UserWarning, stacklevel=2)

        print(f"  Event file   : {os.path.relpath(evt_path, obs_root)}")
        if exp_path:
            print(f"  Exposure map : {os.path.relpath(exp_path, obs_root)}")
        print(f"  Mode         : {mode}")
        return evt_path, exp_path, mode

    # Nothing found — helpful error listing what was tried
    tried_evt = [p.format(obsid=obsid)
                 for mode in ('PC', 'WT')
                 for p in _EVT_GLOBS[mode]]
    raise FileNotFoundError(
        f"No Swift XRT cleaned event file found.\n"
        f"Searched in : {evt_dir}\n"
        f"Patterns tried (first 6 shown):\n"
        + "\n".join(f"  {p}[.gz]" for p in tried_evt[:6])
        + "\nRun xrtpipeline to generate cleaned event files.")


def load_events(cfg, evt_file, mode):
    """
    Load and filter a Swift XRT event file.

    Applies:
    - Energy (PI) filter for the configured energy band
    - Grade filter (0–12 for PC, 0–2 for WT)

    Parameters
    ----------
    cfg      : SwiftConfig
    evt_file : str    path to cleaned event file (plain or .gz)
    mode     : str    'PC' or 'WT'

    Returns
    -------
    events  : astropy.table.Table   filtered event table
    hdr     : astropy.io.fits.Header  merged primary + extension header
    pi_lo   : int
    pi_hi   : int
    """
    e_lo, e_hi = cfg.resolve_energy_band()
    pi_lo, pi_hi = cfg.energy_to_pi(e_lo, e_hi)
    grade_max = cfg.GRADE_LIMITS[mode]

    with fits.open(evt_file, memmap=True) as hdul:
        hdr = hdul[0].header.copy()
        hdr.update(hdul[1].header)

        telescop = hdr.get('TELESCOP', '').strip().upper()
        instrume = hdr.get('INSTRUME', '').strip().upper()
        if 'SWIFT' not in telescop or 'XRT' not in instrume:
            warnings.warn(
                f"Expected Swift XRT event file; got "
                f"TELESCOP={telescop!r}, INSTRUME={instrume!r}.",
                UserWarning, stacklevel=2)

        data = Table(hdul[1].data)

    # Grade column name varies slightly between pipeline versions
    grade_col = next((c for c in ('GRADE', 'PATTERN') if c in data.colnames),
                     None)
    if grade_col is None:
        warnings.warn(
            "No GRADE or PATTERN column found — skipping grade filter.",
            UserWarning, stacklevel=2)

    mask = (data['PI'] >= pi_lo) & (data['PI'] <= pi_hi)
    if grade_col is not None:
        mask &= (data[grade_col] >= 0) & (data[grade_col] <= grade_max)

    events  = data[mask]
    n_total = len(data)
    n_filt  = len(events)
    print(f"  Events       : {n_total:,} total  →  {n_filt:,} after "
          f"PI [{pi_lo}–{pi_hi}] + grade [0–{grade_max}]  "
          f"({e_lo:.2f}–{e_hi:.2f} keV, {mode} mode)")

    return events, hdr, pi_lo, pi_hi


def load_expmap(exp_path):
    """
    Load a Swift XRT exposure map (plain or gzipped).

    Parameters
    ----------
    exp_path : str

    Returns
    -------
    exp_data : np.ndarray   2-D exposure map in seconds
    exp_hdr  : fits.Header
    """
    with fits.open(exp_path) as hdul:
        if hdul[0].data is not None and hdul[0].data.ndim == 2:
            exp_data = hdul[0].data.astype(float)
            exp_hdr  = hdul[0].header.copy()
        else:
            exp_data = hdul[1].data.astype(float)
            exp_hdr  = hdul[1].header.copy()

    print(f"  Exposure map : {os.path.basename(exp_path)}  "
          f"({exp_data.shape[1]}×{exp_data.shape[0]} pix,  "
          f"max={exp_data.max():.0f} s)")

    return exp_data, exp_hdr
