"""
xray_uplim.chandra.io
----------------------
CIAO tool wrappers and file-finding utilities for the Chandra pipeline.

CIAO environment
----------------
CIAO does NOT need to be pre-activated in the shell.  The pipeline
auto-detects the CIAO conda environment by searching standard conda
prefix locations for an env whose bin/ contains 'dmlist'.

If auto-detection fails (non-standard install location), set
    ciao_prefix = "/path/to/ciao-4.XX"   in ChandraConfig,
or activate CIAO manually before running:
    conda activate ciao-4.XX

Public functions
----------------
find_ciao_prefix(hint)           — locate CIAO conda env root
check_ciao(prefix)               — verify tools exist, return env dict
find_repro_dir(base_path, oid)
find_evt2(repro_dir)
run_chandra_repro(base_path, oid, env)
find_fluximage_dir(repro_dir)
run_fluximage(evt2, fdir, ..., env)
find_expmap(fdir)
find_counts_img(fdir)
load_evt2_xy(evt2, e_lo_ev, e_hi_ev)
dmkeypar(evt2, keyword, env)
dmlist_counts(..., env)
dmstat_mean(..., env)
run_aprates(..., env)
"""

import glob
import os
import subprocess

import numpy as np
from astropy.io import fits

REQUIRED_TOOLS = [
    'chandra_repro', 'fluximage', 'dmlist',
    'dmstat', 'dmkeypar', 'aprates', 'pget',
]


# =============================================================================
# CIAO environment discovery
# =============================================================================

def find_ciao_prefix(hint: str = '') -> str | None:
    """
    Locate the root directory of the CIAO conda installation.

    Search order
    ------------
    1. hint          — explicit path supplied by the user (ciao_prefix config param)
    2. Current shell — CIAO already activated ($ASCDS_INSTALL or dmlist in PATH)
    3. Conda envs    — searches base prefixes from CONDA_EXE / CONDA_PREFIX_1
                       and common install paths for an env dir containing dmlist

    Returns the CIAO prefix path, or None if not found.
    """
    import shutil

    # 1. User hint
    if hint:
        dmlist_path = os.path.join(hint, 'bin', 'dmlist')
        if os.path.isfile(dmlist_path):
            return hint
        raise FileNotFoundError(
            f"ciao_prefix='{hint}' was set but dmlist not found at:\n"
            f"  {dmlist_path}\n"
            "Check that this is the root of the CIAO conda environment.")

    # 2. Already in PATH (CIAO activated in shell)
    if shutil.which('dmlist') is not None:
        ascds = os.environ.get('ASCDS_INSTALL')
        if ascds and os.path.isdir(ascds):
            return ascds
        # CIAO is in PATH but ASCDS_INSTALL not set — infer from dmlist location
        dmlist_bin = shutil.which('dmlist')
        inferred = os.path.dirname(os.path.dirname(dmlist_bin))
        if os.path.isfile(os.path.join(inferred, 'bin', 'dmlist')):
            return inferred

    # 3. macOS standalone installer  (/Applications/ciao-4.XX)
    for standalone in sorted(glob.glob('/Applications/ciao-*')):
        if os.path.isfile(os.path.join(standalone, 'bin', 'dmlist')):
            return standalone

    # 4. Linux standalone install in common locations
    home = os.path.expanduser('~')
    for pattern in ('/usr/local/ciao-*', f'{home}/ciao-*', '/opt/ciao-*'):
        for standalone in sorted(glob.glob(pattern)):
            if os.path.isfile(os.path.join(standalone, 'bin', 'dmlist')):
                return standalone

    # 5. Search conda environments
    conda_bases = _candidate_conda_bases()
    for base in conda_bases:
        envs_dir = os.path.join(base, 'envs')
        if not os.path.isdir(envs_dir):
            continue
        # Sort to prefer newer CIAO versions (higher version number last)
        for env_dir in sorted(glob.glob(os.path.join(envs_dir, 'ciao*'))):
            if os.path.isfile(os.path.join(env_dir, 'bin', 'dmlist')):
                return env_dir

    return None


def _candidate_conda_bases() -> list:
    """Return a list of conda base prefix directories to search."""
    candidates = []

    # From CONDA_EXE: e.g. ~/opt/miniconda3/bin/conda → ~/opt/miniconda3
    conda_exe = os.environ.get('CONDA_EXE', '')
    if conda_exe:
        candidates.append(os.path.dirname(os.path.dirname(conda_exe)))

    # CONDA_PREFIX_1 is set when a non-base env is active
    prefix1 = os.environ.get('CONDA_PREFIX_1', '')
    if prefix1:
        candidates.append(prefix1)

    # CONDA_PREFIX itself (base env or active env)
    prefix = os.environ.get('CONDA_PREFIX', '')
    if prefix:
        # If an env is active, base is one level up
        candidates.append(prefix)
        candidates.append(os.path.dirname(prefix))

    # Common install locations (macOS and Linux)
    home = os.path.expanduser('~')
    for name in ('opt/miniconda3', 'miniconda3', 'opt/anaconda3', 'anaconda3',
                 'opt/miniforge3', 'miniforge3'):
        candidates.append(os.path.join(home, name))
    candidates.append('/opt/miniconda3')
    candidates.append('/opt/anaconda3')

    # Deduplicate while preserving order
    seen = set()
    result = []
    for c in candidates:
        if c and c not in seen:
            seen.add(c)
            result.append(c)
    return result


def build_ciao_env(prefix: str) -> dict:
    """
    Build a complete os.environ dict with all CIAO environment variables set.

    Strategy
    --------
    Standalone CIAO (macOS .pkg / Linux tar installer, e.g. /Applications/ciao-4.18):
        Sources $prefix/bin/ciao.sh in a bash subprocess and captures the
        resulting environment.  This is the only reliable way to pick up all
        ~20 internal variables that CIAO tools expect (ASCDS_CALIB,
        ASCDS_WORK_PATH, ASCDS_UPARM, PFILES, etc.).

    Conda CIAO (no ciao.sh present):
        Falls back to manually constructing the essential variables
        (PATH, ASCDS_INSTALL, CALDB, PFILES).  Conda activation scripts
        handle the rest when the env is active; if the env is not active
        we set the variables we know about.
    """
    ciao_sh = os.path.join(prefix, 'bin', 'ciao.sh')

    if os.path.isfile(ciao_sh):
        # Source ciao.sh and capture every resulting env var.
        # We use 'env -0' (null-delimited output) to safely handle vars
        # whose values contain newlines or '=' signs.
        cmd = f'source "{ciao_sh}" > /dev/null 2>&1 && env -0'
        result = subprocess.run(
            ['bash', '-c', cmd],
            capture_output=True, text=True
        )
        if result.returncode == 0 and result.stdout:
            env = {}
            for entry in result.stdout.split('\0'):
                if '=' in entry:
                    key, _, val = entry.partition('=')
                    if key:
                        env[key] = val
            if env:
                # Make sure the user-writable param dir exists
                pfiles_user = os.path.expanduser('~/cxcds_param4')
                os.makedirs(pfiles_user, exist_ok=True)
                # Prepend user param dir so user-modified pfiles take priority
                existing_pfiles = env.get('PFILES', '')
                if pfiles_user not in existing_pfiles:
                    env['PFILES'] = (f"{pfiles_user};"
                                     + existing_pfiles.lstrip(';'))
                return env

    # Fallback: conda-style installation without ciao.sh
    env = os.environ.copy()
    bin_dir = os.path.join(prefix, 'bin')
    env['PATH'] = bin_dir + os.pathsep + env.get('PATH', '')
    env['ASCDS_INSTALL'] = prefix

    caldb_conda = os.path.join(prefix, 'CALDB')
    if os.path.isdir(caldb_conda) and 'CALDB' not in os.environ:
        env['CALDB'] = caldb_conda

    pfiles_user = os.path.expanduser('~/cxcds_param4')
    os.makedirs(pfiles_user, exist_ok=True)
    param_dir = os.path.join(prefix, 'param')
    env['PFILES'] = f"{pfiles_user};{param_dir}"

    return env


def check_ciao(ciao_prefix: str = '') -> dict:
    """
    Locate CIAO and verify all required tools exist.

    Parameters
    ----------
    ciao_prefix : optional explicit path to the CIAO conda env root.
                  If empty, auto-detection is attempted.

    Returns
    -------
    env : dict  — environment dict to pass to _run() calls; contains the
                  PATH / ASCDS_INSTALL / CALDB / PFILES needed for CIAO.

    Raises RuntimeError if CIAO cannot be found.
    """
    import shutil

    prefix = find_ciao_prefix(ciao_prefix)
    if prefix is None:
        raise RuntimeError(
            "CIAO installation not found.\n\n"
            "Options:\n"
            "  1. Set ciao_prefix='/path/to/ciao-4.XX' in the CHANDRA config.\n"
            "  2. Activate CIAO before running:  conda activate ciao-4.XX\n"
            "  3. Install CIAO via conda:  "
            "https://cxc.cfa.harvard.edu/ciao/download/\n"
        )

    env = build_ciao_env(prefix)

    # Verify all required tools are present in the env
    missing = []
    for tool in REQUIRED_TOOLS:
        # Check in the env's bin first, then the whole env PATH
        if not os.path.isfile(os.path.join(prefix, 'bin', tool)):
            # Fall back to shutil.which with modified PATH
            path_save = os.environ.get('PATH', '')
            os.environ['PATH'] = env['PATH']
            found = shutil.which(tool) is not None
            os.environ['PATH'] = path_save
            if not found:
                missing.append(tool)

    if missing:
        raise RuntimeError(
            f"CIAO prefix found at:\n  {prefix}\n"
            f"But these tools are missing: {', '.join(missing)}\n"
            "Your CIAO installation may be incomplete.")

    print(f"  CIAO found at: {prefix}")
    return env


# =============================================================================
# Internal subprocess helper
# =============================================================================

def _run(cmd: list, env: dict | None = None, cwd: str | None = None) -> str:
    """Run a command (list of strings), return stdout.  Raises on failure."""
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env=env,
        cwd=cwd,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed (exit {result.returncode}):\n"
            f"  {' '.join(str(x) for x in cmd)}\n"
            f"stderr:\n{result.stderr.strip()}"
        )
    return result.stdout.strip()


# =============================================================================
# File finding
# =============================================================================

def find_repro_dir(base_path: str, obsid_str: str) -> str | None:
    """Return path to obsid/repro/ if it exists, else None."""
    repro = os.path.join(base_path, obsid_str, 'repro')
    return repro if os.path.isdir(repro) else None


def find_evt2(repro_dir: str) -> str | None:
    """Find the level-2 reprocessed event file in repro_dir."""
    for pat in [
        os.path.join(repro_dir, '*repro_evt2.fits'),
        os.path.join(repro_dir, '*_evt2.fits'),
        os.path.join(repro_dir, '*.fits'),
    ]:
        hits = sorted(glob.glob(pat))
        hits = [h for h in hits if not any(
            x in os.path.basename(h)
            for x in ('asol', 'msk', 'bpix', 'fov', 'pbk', 'bias'))]
        if hits:
            return hits[0]
    return None


def find_fluximage_dir(repro_dir: str) -> str:
    """Return the fluximage output directory (repro_dir/fluximage/)."""
    return os.path.join(repro_dir, 'fluximage')


def find_expmap(fluximage_dir: str,
                e_lo_kev: float | None = None,
                e_hi_kev: float | None = None) -> str | None:
    """
    Find the *_thresh.expmap for the requested energy band inside
    the managed fluximage/ subdirectory.

    When e_lo_kev / e_hi_kev are given (the normal case), only the
    band-specific file is accepted — e.g. ``0.5-7.0_thresh.expmap``.
    If the file exists but is for a *different* band, None is returned
    so that the caller re-runs fluximage for the correct band.
    """
    if e_lo_kev is not None and e_hi_kev is not None:
        fname = f'{e_lo_kev}-{e_hi_kev}_thresh.expmap'
        full  = os.path.join(fluximage_dir, fname)
        return full if os.path.isfile(full) else None
    # Fallback (no band info): return whatever is there
    hits = sorted(glob.glob(os.path.join(fluximage_dir, '*_thresh.expmap')))
    return hits[0] if hits else None


def find_counts_img(fluximage_dir: str,
                    e_lo_kev: float | None = None,
                    e_hi_kev: float | None = None) -> str | None:
    """Find the *_thresh.img for the requested energy band."""
    if e_lo_kev is not None and e_hi_kev is not None:
        fname = f'{e_lo_kev}-{e_hi_kev}_thresh.img'
        full  = os.path.join(fluximage_dir, fname)
        return full if os.path.isfile(full) else None
    hits = sorted(glob.glob(os.path.join(fluximage_dir, '*_thresh.img')))
    return hits[0] if hits else None


# =============================================================================
# CIAO tool calls  (all accept env= dict from check_ciao())
# =============================================================================

def run_chandra_repro(base_path: str, obsid_str: str, env: dict):
    """Run chandra_repro for obsid_str in base_path."""
    obs_dir   = os.path.join(base_path, obsid_str)
    repro_dir = os.path.join(obs_dir, 'repro')
    if not os.path.isdir(obs_dir):
        raise FileNotFoundError(
            f"Observation directory not found: {obs_dir}\n"
            "Download the Chandra data first (e.g. via the CDA web portal "
            "or chandra_downloader).")
    print(f"  Running chandra_repro on {obsid_str}…")
    _run(['chandra_repro', obs_dir, f'outdir={repro_dir}', 'verbose=0'],
         env=env, cwd=base_path)
    print(f"  chandra_repro complete → {repro_dir}")


def run_fluximage(evt2: str, fluximage_dir: str,
                  e_lo_kev: float, e_hi_kev: float, ref_kev: float,
                  env: dict, clobber: bool = True):
    """
    Run CIAO fluximage to create the counts image and exposure map.

    Parameters
    ----------
    evt2          : path to the level-2 event file
    fluximage_dir : output directory
    e_lo_kev      : band low energy in keV
    e_hi_kev      : band high energy in keV
    ref_kev       : spectral reference energy in keV (for ARF weighting)
    env           : CIAO environment dict from check_ciao()
    """
    os.makedirs(fluximage_dir, exist_ok=True)
    band_str = f"{e_lo_kev}:{e_hi_kev}:{ref_kev}"
    clob     = 'yes' if clobber else 'no'
    # fluximage's second arg is an outROOT prefix, not a directory.
    # Appending '/' tells it to treat the path as a directory and write
    # files inside it as  {dir}/{band}_thresh.expmap  (no extra prefix).
    outroot  = fluximage_dir.rstrip('/') + '/'
    print(f"  Running fluximage  (band = {band_str} keV)…")
    _run([
        'fluximage',
        evt2,
        outroot,
        f'bands={band_str}',
        f'clobber={clob}',
    ], env=env, cwd=os.path.dirname(evt2))
    print(f"  fluximage complete → {fluximage_dir}")


def load_evt2_xy(evt2: str, e_lo_ev: int, e_hi_ev: int):
    """
    Load X, Y sky-pixel coordinates from the event file, filtered by energy.
    This is pure Python (no CIAO tools needed).

    Returns
    -------
    evt_x, evt_y : np.ndarray  — sky pixel coordinates
    evt_hdr      : fits.Header — EVENTS extension header (for WCS)
    """
    with fits.open(evt2) as hdul:
        evtext = None
        for hdu in hdul:
            if hdu.name.upper() in ('EVENTS', 'UNFILTERED_EVENTS'):
                evtext = hdu
                break
        if evtext is None:
            evtext = hdul[1]

        data  = evtext.data
        hdr   = evtext.header
        pi    = np.asarray(data['energy'], dtype=float)
        mask  = (pi >= e_lo_ev) & (pi <= e_hi_ev)
        evt_x = np.asarray(data['x'][mask], dtype=float)
        evt_y = np.asarray(data['y'][mask], dtype=float)

    return evt_x, evt_y, hdr


def dmkeypar(evt2: str, keyword: str, env: dict) -> str:
    """Read a header keyword from an event file via CIAO dmkeypar."""
    return _run(['dmkeypar', evt2, keyword, 'echo+'], env=env).strip()


def _fmt_coord(ra_deg: float, dec_deg: float, radius_arcsec: float,
               shape: str = 'circle',
               inner_arcsec: float | None = None) -> str:
    """Build a CIAO (ra,dec)=circle/annulus filter expression."""
    if shape == 'circle':
        return f'(ra,dec)=circle({ra_deg:.6f},{dec_deg:+.6f},{radius_arcsec}")'
    elif shape == 'annulus':
        if inner_arcsec is None:
            raise ValueError("inner_arcsec required for annulus shape")
        return (f'(ra,dec)=annulus({ra_deg:.6f},{dec_deg:+.6f},'
                f'{inner_arcsec}",{radius_arcsec}")')
    raise ValueError(f"Unknown shape '{shape}'")


def dmlist_counts(evt2: str,
                  ra_deg: float, dec_deg: float,
                  radius_arcsec: float,
                  e_lo_ev: int, e_hi_ev: int,
                  env: dict,
                  inner_arcsec: float | None = None) -> int:
    """Count photons in a circle or annulus via CIAO dmlist."""
    shape  = 'annulus' if inner_arcsec is not None else 'circle'
    region = _fmt_coord(ra_deg, dec_deg, radius_arcsec, shape, inner_arcsec)
    filt   = f"{evt2}[energy={e_lo_ev}:{e_hi_ev}][{region}]"
    out    = _run(['dmlist', filt, 'counts'], env=env)
    return int(out.strip())


def dmstat_mean(fitsfile: str,
                ra_deg: float, dec_deg: float,
                radius_arcsec: float,
                env: dict,
                inner_arcsec: float | None = None) -> float:
    """Return mean pixel value in a circle or annulus via CIAO dmstat."""
    shape  = 'annulus' if inner_arcsec is not None else 'circle'
    region = _fmt_coord(ra_deg, dec_deg, radius_arcsec, shape, inner_arcsec)
    filt   = f"{fitsfile}[{region}]"
    out    = _run(['dmstat', filt, 'centroid=no', 'sigma=no', 'median=no'],
                  env=env)
    for line in out.splitlines():
        if line.strip().lower().startswith('mean:'):
            return float(line.split(':')[1].strip())
    raise RuntimeError(f"Could not parse dmstat mean from:\n{out}")


def expmap_aperture_mean(expmap_path: str,
                         ra_deg: float, dec_deg: float,
                         radius_arcsec: float,
                         inner_arcsec: float | None = None) -> float:
    """
    Compute the mean exposure-map value inside a circular aperture or
    annulus using astropy WCS — no CIAO filtering needed.

    The ``(ra,dec)=circle(...)`` dmstat syntax works for event files (which
    have actual RA/Dec columns) but not for image FITS files.  For image
    files we convert the celestial coordinates to pixel coordinates via
    the WCS header and build a pixel mask directly.

    Parameters
    ----------
    expmap_path    : path to the *_thresh.expmap FITS image
    ra_deg, dec_deg : source (or bkg centre) in decimal degrees
    radius_arcsec  : outer aperture radius in arcseconds
    inner_arcsec   : annulus inner radius; None → filled circle

    Returns
    -------
    float  — mean exposure map value (cm²·s) over the aperture/annulus.
             Returns 0.0 if no valid pixels found.
    """
    from astropy.wcs import WCS

    with fits.open(expmap_path) as hdul:
        # Chandra fluximage expmaps are in the primary HDU
        img = hdul[0].data.astype(float)
        hdr = hdul[0].header

    wcs = WCS(hdr, naxis=2)

    # World → pixel  (astropy uses 0-indexed pixel convention)
    # all_world2pix returns (x_pix, y_pix) with origin=0
    px, py = wcs.all_world2pix(ra_deg, dec_deg, 0)

    # Pixel scale: derive from the CD/CDELT matrix (degrees → arcsec)
    try:
        pscale_deg = float(abs(wcs.proj_plane_pixel_scales()[1].value))
    except Exception:
        pscale_deg = abs(hdr.get('CDELT2',
                        hdr.get('CD2_2', 0.000136667)))   # fallback: ACIS ~0.492"/pix
    pscale_arcsec = pscale_deg * 3600.0

    r_pix  = radius_arcsec / pscale_arcsec
    ny, nx = img.shape
    yy, xx = np.ogrid[0:ny, 0:nx]
    dist2  = (xx - px) ** 2 + (yy - py) ** 2

    if inner_arcsec is not None:
        r_in_pix = inner_arcsec / pscale_arcsec
        mask = (dist2 >= r_in_pix ** 2) & (dist2 <= r_pix ** 2)
    else:
        mask = dist2 <= r_pix ** 2

    pixels = img[mask]
    if len(pixels) == 0:
        return 0.0
    return float(np.mean(pixels))


def run_aprates(n: int, m: int,
                A_s: float, A_b: float,
                T_s: float, T_b: float,
                E_s: float, E_b: float,
                conf: float,
                outfile: str,
                env: dict,
                clobber: bool = True) -> tuple:
    """
    Run CIAO aprates and return (src_rate, err_lo, err_up) in cts/s.

    Parameters
    ----------
    n, m     : source and background counts
    A_s, A_b : source and background areas in arcsec²
    T_s, T_b : LIVETIME in seconds (source and background; equal for same obs)
    E_s, E_b : mean exposure map (cm²·s) in src and bkg apertures
    conf     : one-sided confidence level (e.g. 0.9545 for 2σ)
    outfile  : path for the aprates output parameter file
    env      : CIAO environment dict from check_ciao()

    Returns INDEF values as 0.0.
    """
    clob = 'yes' if clobber else 'no'

    _run([
        'aprates',
        f'n={n}', f'm={m}',
        f'A_s={A_s:.6f}', f'A_b={A_b:.6f}',
        f'T_s={T_s:.6f}', f'T_b={T_b:.6f}',
        f'E_s={E_s:.6f}', f'E_b={E_b:.6f}',
        'eng_s=1', 'flux_s=1', 'eng_b=1', 'flux_b=1',
        f'conf={conf:.6f}',
        f'outfile={outfile}',
        f'clobber={clob}',
        'mode=h',
    ], env=env)

    out   = _run(['pget', outfile,
                  'src_rate', 'src_rate_err_lo', 'src_rate_err_up'], env=env)
    lines = out.strip().splitlines()

    def _parse(val: str) -> float:
        return 0.0 if val.strip().upper() in ('INDEF', '', 'NONE') else float(val)

    src_rate = _parse(lines[0]) if len(lines) > 0 else 0.0
    err_lo   = _parse(lines[1]) if len(lines) > 1 else 0.0
    err_up   = _parse(lines[2]) if len(lines) > 2 else 0.0
    return src_rate, err_lo, err_up
