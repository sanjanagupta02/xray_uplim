"""
xray_uplim.flux_conversion
--------------------------
Optional flux / luminosity conversion for X-ray count-rate upper limits.

Workflow
--------
1.  Auto-fetch Galactic N_H from the HEASARC HI4PI tool (or use a
    user-supplied value).
2.  Call WebPIMMS (NASA HEASARC) to convert a count-rate upper limit to
    an absorbed and unabsorbed flux (erg/cm^2/s) given a spectral model.
3.  Optionally convert flux → luminosity using astropy.cosmology and a
    source redshift obtained from NED or supplied by the user.

All network calls are wrapped in try/except so that pipeline failure does
not prevent the count-rate results from being written.

Usage (called from the individual telescope pipelines)
------------------------------------------------------
    from ..flux_conversion import compute_flux_for_rows
    nh_used = compute_flux_for_rows(
        all_csv_rows, 'NuSTAR', cfg,
        pimms_from_map={'FPMA': 'NUSTAR Count Rate',
                        'FPMB': 'NUSTAR Count Rate',
                        'AB':   'NUSTAR Count Rate'},
        e_lo_kev=3.0, e_hi_kev=79.0,
        nh_cm2=cfg.nh_cm2,
        ra_deg=src_coord.ra.deg,
        dec_deg=src_coord.dec.deg)
"""

import re
import warnings

# ---------------------------------------------------------------------------
# APEC log-temperature lookup table
# (logt, kT_kev) — from the WebPIMMS temperature grid
# ---------------------------------------------------------------------------

APEC_LOGT_OPTIONS = [
    (5.60, 0.0343), (5.65, 0.0385), (5.70, 0.0432), (5.75, 0.0485),
    (5.80, 0.0544), (5.85, 0.0610), (5.90, 0.0684), (5.95, 0.0768),
    (6.00, 0.0862), (6.05, 0.0967), (6.10, 0.1085), (6.15, 0.1217),
    (6.20, 0.1366), (6.25, 0.1532), (6.30, 0.1719), (6.35, 0.1929),
    (6.40, 0.2165), (6.45, 0.2429), (6.50, 0.2725), (6.55, 0.3058),
    (6.60, 0.3431), (6.65, 0.3849), (6.70, 0.4319), (6.75, 0.4846),
    (6.80, 0.5437), (6.85, 0.6101), (6.90, 0.6845), (6.95, 0.7680),
    (7.00, 0.8617), (7.05, 0.9669), (7.10, 1.0849), (7.15, 1.2172),
    (7.20, 1.3658), (7.25, 1.5324), (7.30, 1.7194), (7.35, 1.9292),
    (7.40, 2.1646), (7.45, 2.4287), (7.50, 2.7250), (7.55, 3.0575),
    (7.60, 3.4306), (7.65, 3.8492), (7.70, 4.3189),
]


def _apec_logt_string(temperature_kev):
    """
    Find the closest APEC log-temperature entry and return the formatted
    string expected by WebPIMMS: "6.50 | 0.2725".
    """
    best_logt, best_kev = min(
        APEC_LOGT_OPTIONS,
        key=lambda t: abs(t[1] - temperature_kev))
    return f"{best_logt:.2f} | {best_kev:.4f}"


def _solar_abundance_string(abundance):
    """Map a float abundance to the WebPIMMS solar abundance dropdown string."""
    options = [1.0, 0.8, 0.6, 0.4, 0.2]
    closest = min(options, key=lambda x: abs(x - abundance))
    return f"{closest:.1f} Solar Abundance"


# ---------------------------------------------------------------------------
# Network helpers
# ---------------------------------------------------------------------------

def fetch_nh(ra_deg, dec_deg):
    """
    Query the HEASARC HI4PI N_H tool for the Galactic hydrogen column density.

    Parameters
    ----------
    ra_deg  : float — source right ascension (decimal degrees, J2000)
    dec_deg : float — source declination     (decimal degrees, J2000)

    Returns
    -------
    float or None
        Weighted-average N_H in cm^-2, or None if the query fails.
    """
    import urllib.request
    import urllib.parse

    # Space-separated RA Dec (positive or negative Dec both work)
    entry = f"{ra_deg:.6f} {dec_deg:.6f}"
    url = (
        'https://heasarc.gsfc.nasa.gov/cgi-bin/Tools/w3nh/w3nh.pl'
        f'?Entry={urllib.parse.quote(entry)}'
        '&NR=GRB%2FSIMBAD%2BSesame%2FNED'
        '&CoordSys=Equatorial&equinox=2000&radius=0.1&usemap=0'
    )
    try:
        with urllib.request.urlopen(url, timeout=20) as resp:
            html = resp.read().decode('utf-8', errors='replace')
    except Exception as exc:
        warnings.warn(f"NH fetch failed (network): {exc}", UserWarning)
        return None

    # Try weighted average first, then plain average
    for pattern in (
        r'Weighted average nH \(cm\*\*-2\)\s+([\d.E+\-]+)',
        r'Average nH \(cm\*\*-2\)\s+([\d.E+\-]+)',
    ):
        m = re.search(pattern, html)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                pass

    warnings.warn("NH fetch: could not parse N_H from HEASARC response.",
                  UserWarning)
    return None


def fetch_redshift_ned(src_name):
    """
    Query the NED REST API for the redshift of a named source.

    Parameters
    ----------
    src_name : str — source name (e.g. 'SN 2012ap')

    Returns
    -------
    float or None
    """
    import urllib.request
    import urllib.parse
    import json

    url = ('https://ned.ipac.caltech.edu/srs/ObjectLookup?'
           f'name={urllib.parse.quote(src_name)}&mode=json')
    try:
        with urllib.request.urlopen(url, timeout=15) as resp:
            data = json.loads(resp.read())
        z = data['Preferred']['Redshift']['Value']
        if z is not None:
            return float(z)
    except (KeyError, TypeError):
        pass
    except Exception as exc:
        warnings.warn(f"NED redshift fetch failed: {exc}", UserWarning)
    return None


# ---------------------------------------------------------------------------
# FITS header keyword detectors
# ---------------------------------------------------------------------------

def detect_xmm_filter(evt_file, instrument):
    """
    Read the FILTER keyword from an XMM event file FITS header.

    Parameters
    ----------
    evt_file   : str — path to the event file
    instrument : str — 'MOS1', 'MOS2', or 'PN' (unused but kept for clarity)

    Returns
    -------
    str — 'Thin', 'Medium', or 'Thick'
    """
    try:
        from astropy.io import fits
        hdr = fits.getheader(evt_file, ext=1)
        filt = str(hdr.get('FILTER', '')).strip().title()
        # Normalise common variants
        if 'Thin' in filt:
            return 'Thin'
        if 'Med' in filt:
            return 'Medium'
        if 'Thick' in filt:
            return 'Thick'
    except Exception as exc:
        warnings.warn(f"Could not read XMM FILTER keyword from {evt_file}: {exc}",
                      UserWarning)
    return 'Thin'   # safe default


def detect_chandra_acis_type(evt_file):
    """
    Read DETNAM (or INSTRUME) from a Chandra event file to determine
    whether this is an ACIS-S or ACIS-I observation.

    Returns
    -------
    str — 'ACIS-S' or 'ACIS-I'
    """
    try:
        from astropy.io import fits
        hdr = fits.getheader(evt_file, ext=1)
        detnam = str(hdr.get('DETNAM', '')).strip().upper()
        if not detnam:
            detnam = str(hdr.get('INSTRUME', '')).strip().upper()
        # DETNAM is typically e.g. "ACIS-0123456789" or "ACIS-S3"
        # ACIS-I chips: 0-3; ACIS-S chips: 4-9 (primarily S3=chip6/ACIS-S)
        if 'ACIS-S' in detnam:
            return 'ACIS-S'
        if 'ACIS-I' in detnam:
            return 'ACIS-I'
        # Parse chip numbers: presence of 7 (S3) → ACIS-S; 0-3 → ACIS-I
        chips = re.sub(r'[^0-9]', '', detnam)
        if any(c in chips for c in ('4', '5', '6', '7', '8', '9')):
            return 'ACIS-S'
        if any(c in chips for c in ('0', '1', '2', '3')):
            return 'ACIS-I'
    except Exception as exc:
        warnings.warn(
            f"Could not read Chandra DETNAM keyword from {evt_file}: {exc}",
            UserWarning)
    return 'ACIS-S'   # safe default


def detect_swift_xrt_mode(evt_file):
    """
    Read DATAMODE keyword from a Swift XRT event file.

    Returns
    -------
    str — 'PC', 'WT', or 'PD'
    """
    try:
        from astropy.io import fits
        hdr = fits.getheader(evt_file, ext=1)
        mode = str(hdr.get('DATAMODE', '')).strip().upper()
        if 'PHOTON' in mode or mode == 'PC':
            return 'PC'
        if 'WINDOW' in mode or mode == 'WT':
            return 'WT'
        if 'PILEDUP' in mode or mode == 'PD':
            return 'PD'
    except Exception as exc:
        warnings.warn(
            f"Could not read Swift DATAMODE keyword from {evt_file}: {exc}",
            UserWarning)
    return 'PC'   # safe default


# ---------------------------------------------------------------------------
# PIMMS instrument code mapping
# ---------------------------------------------------------------------------

def pimms_instrument_code(telescope, instrument, filter_or_mode=None):
    """
    Map a telescope / instrument / filter combination to the WebPIMMS
    'from' form field string.

    Parameters
    ----------
    telescope      : str — 'NuSTAR', 'XMM', 'Swift', 'Chandra'
    instrument     : str — 'FPMA'/'FPMB', 'MOS1'/'MOS2'/'PN', 'XRT',
                          'ACIS-S'/'ACIS-I'
    filter_or_mode : str or None — XMM filter ('Thin','Medium','Thick')
                     or Swift mode ('PC','WT')

    Returns
    -------
    str — WebPIMMS instrument string
    """
    tele = telescope.upper()

    if tele == 'NUSTAR':
        return 'NUSTAR Count Rate'

    if tele == 'XMM':
        filt = (filter_or_mode or 'Thin').strip().title()
        if filt.startswith('Med'):
            filt = 'Med'
        elif filt.startswith('Thick'):
            filt = 'Thick'
        else:
            filt = 'Thin'
        inst = instrument.upper()
        if inst in ('MOS1', 'MOS2'):
            return f"XMM/MOS {filt} Count Rate 5' region"
        if inst == 'PN':
            return f"XMM/PN {filt} Count Rate 5' region"

    if tele in ('SWIFT', 'XRT'):
        mode = (filter_or_mode or 'PC').strip().upper()
        if mode == 'WT':
            return 'SWIFT/XRT/WT Count Rate'
        return 'SWIFT/XRT/PC Count Rate'

    if tele == 'CHANDRA':
        inst = instrument.upper()
        if 'ACIS-I' in inst:
            return 'CHANDRA/ACIS-I Count Rate'
        return 'CHANDRA/ACIS-S Count Rate'

    warnings.warn(f"Unknown telescope '{telescope}' — cannot map to PIMMS code.",
                  UserWarning)
    return ''


# ---------------------------------------------------------------------------
# WebPIMMS count-rate → flux conversion
# ---------------------------------------------------------------------------

def webpimms_cr_to_flux(pimms_from, count_rate, e_lo_kev, e_hi_kev,
                         nh_cm2, model='powerlaw', photon_index=2.0,
                         temperature_kev=1.0, abundance=1.0):
    """
    Convert a count-rate upper limit to flux using NASA HEASARC WebPIMMS.

    Parameters
    ----------
    pimms_from     : str   — WebPIMMS 'from' instrument string
    count_rate     : float — count rate (cts/s)
    e_lo_kev       : float — lower energy bound (keV)
    e_hi_kev       : float — upper energy bound (keV)
    nh_cm2         : float — Galactic N_H (cm^-2)
    model          : str   — 'powerlaw', 'blackbody', 'bremsstrahlung', 'apec'
    photon_index   : float — photon index Γ (power law only)
    temperature_kev: float — kT in keV (blackbody / bremsstrahlung / apec)
    abundance      : float — solar abundance (apec only)

    Returns
    -------
    dict with keys 'flux_abs' and 'flux_unabs' (erg/cm^2/s)

    Raises
    ------
    RuntimeError if PIMMS returns an error message.
    """
    import urllib.request
    import urllib.parse

    # Map model name to PIMMS form field values
    model_lower = model.lower()
    if model_lower == 'powerlaw':
        pimms_model = 'Power Law'
        gama = str(photon_index)
        gamb = ''
        gamc = ''
        logt = ''
    elif model_lower == 'blackbody':
        pimms_model = 'Black Body'
        gama = ''
        gamb = str(temperature_kev)
        gamc = ''
        logt = ''
    elif model_lower in ('bremsstrahlung', 'bremss'):
        pimms_model = 'Therm. Bremss.'
        gama = ''
        gamb = ''
        gamc = str(temperature_kev)
        logt = ''
    elif model_lower == 'apec':
        pimms_model = 'APEC'
        gama = ''
        gamb = ''
        gamc = ''
        logt = _apec_logt_string(temperature_kev)
    else:
        raise ValueError(f"Unknown spectral model: '{model}'. "
                         "Use 'powerlaw', 'blackbody', 'bremsstrahlung', or 'apec'.")

    solar_str   = _solar_abundance_string(abundance)
    energy_range = f"{e_lo_kev}-{e_hi_kev}"

    form_data = urllib.parse.urlencode({
        'from':   pimms_from,
        'sat':    'FLUX',
        'model':  pimms_model,
        'gama':   gama,
        'gamb':   gamb,
        'gamc':   gamc,
        'solar':  solar_str,
        'logt':   logt,
        'nh':     str(nh_cm2),
        'nhi':    'none',
        'red':    'none',
        'flusso': str(count_rate),
        'range':  energy_range,
        'orange': energy_range,
        'etype':  'kev',
        'otype':  'kev',
    }).encode('ascii')

    url = 'https://heasarc.gsfc.nasa.gov/cgi-bin/Tools/w3pimms/w3pimms.pl'
    req = urllib.request.Request(url, data=form_data, method='POST')

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            html = resp.read().decode('utf-8', errors='replace')
    except Exception as exc:
        raise RuntimeError(f"WebPIMMS network error: {exc}") from exc

    # Check for PIMMS error
    err_m = re.search(r'<div[^>]*pimmserror[^>]*>(.*?)</div>',
                      html, re.IGNORECASE | re.DOTALL)
    if err_m:
        raise RuntimeError(f"WebPIMMS error: {err_m.group(1).strip()}")

    # Parse absorbed flux
    abs_m = re.search(
        r'PIMMS predicts a flux\s*\([^)]+\)\s*of\s*([\d.E+\-]+)\s*ergs',
        html, re.IGNORECASE)
    # Parse unabsorbed flux
    unabs_m = re.search(
        r'PIMMS predicts an unabsorbed flux\s*\([^)]+\)\s*of\s*([\d.E+\-]+)\s*ergs',
        html, re.IGNORECASE)

    if not abs_m:
        raise RuntimeError(
            "WebPIMMS: could not parse flux from response. "
            "Check instrument name and energy range.")

    flux_abs   = float(abs_m.group(1))
    flux_unabs = float(unabs_m.group(1)) if unabs_m else flux_abs

    return {'flux_abs': flux_abs, 'flux_unabs': flux_unabs}


# ---------------------------------------------------------------------------
# Flux → luminosity
# ---------------------------------------------------------------------------

def flux_to_luminosity(flux_cgs, redshift, cosmology='Planck18',
                        h0=67.4, omega_m=0.315):
    """
    Convert flux (erg/cm^2/s) to luminosity (erg/s) using luminosity distance.

    Parameters
    ----------
    flux_cgs   : float — absorbed or unabsorbed flux (erg/cm^2/s)
    redshift   : float — source redshift
    cosmology  : str   — 'Planck18', 'WMAP9', or 'custom'
    h0         : float — H0 in km/s/Mpc (only used when cosmology='custom')
    omega_m    : float — Omega_M (only used when cosmology='custom')

    Returns
    -------
    float — luminosity (erg/s)
    """
    from astropy.cosmology import Planck18, WMAP9, FlatLambdaCDM
    import astropy.units as u

    cosmo_key = cosmology.strip().lower()
    if cosmo_key == 'planck18':
        cosmo = Planck18
    elif cosmo_key == 'wmap9':
        cosmo = WMAP9
    else:
        cosmo = FlatLambdaCDM(H0=h0, Om0=omega_m)

    d_L_cm = cosmo.luminosity_distance(redshift).to(u.cm).value
    luminosity = flux_cgs * 4.0 * 3.141592653589793 * d_L_cm ** 2
    return luminosity


# ---------------------------------------------------------------------------
# Model parameter string helper
# ---------------------------------------------------------------------------

def _model_params_str(model, photon_index, temperature_kev, abundance):
    """Format a short human-readable parameter string for the CSV."""
    m = model.lower()
    if m == 'powerlaw':
        return f"Gamma={photon_index:.2f}"
    if m == 'blackbody':
        return f"kT={temperature_kev:.3f}keV"
    if m in ('bremsstrahlung', 'bremss'):
        return f"kT={temperature_kev:.3f}keV"
    if m == 'apec':
        return f"kT={temperature_kev:.3f}keV Z={abundance:.1f}solar"
    return ''


# ---------------------------------------------------------------------------
# High-level per-row flux computation
# ---------------------------------------------------------------------------

def compute_flux_for_rows(csv_rows, telescope, cfg,
                           pimms_from_map,
                           e_lo_kev, e_hi_kev,
                           nh_cm2=None, ra_deg=None, dec_deg=None):
    """
    For each CSV row that has a count-rate upper limit, compute flux
    (and optionally luminosity) using WebPIMMS and add the results as
    new keys in the row dict.

    New keys added to each row
    --------------------------
    'nh_cm2_used'        : N_H used (cm^-2)
    'pimms_instrument'   : WebPIMMS instrument string
    'spectral_model'     : model name
    'model_params_str'   : e.g. "Gamma=2.00"
    'flux_ul_cgs'        : absorbed flux UL (erg/cm^2/s)
    'flux_ul_unabs_cgs'  : unabsorbed flux UL (erg/cm^2/s)
    'lum_ul_cgs'         : luminosity UL (erg/s) — only if redshift is set

    Parameters
    ----------
    csv_rows       : list of dicts (modified in place)
    telescope      : str   — 'NuSTAR', 'XMM', 'Swift', 'Chandra'
    cfg            : config dataclass with flux conversion fields
    pimms_from_map : dict mapping instrument/module field → PIMMS string
                     e.g. {'FPMA': 'NUSTAR Count Rate',
                            'FPMB': 'NUSTAR Count Rate',
                            'AB':   'NUSTAR Count Rate'}
    e_lo_kev       : float
    e_hi_kev       : float
    nh_cm2         : float or None — manual N_H; auto-fetched if None
    ra_deg         : float or None — source RA for auto N_H fetch
    dec_deg        : float or None — source Dec for auto N_H fetch

    Returns
    -------
    float or None — the actual N_H used (cm^-2)
    """
    if not csv_rows:
        return nh_cm2

    # --- Resolve N_H ---------------------------------------------------------
    nh_used = nh_cm2
    if nh_used is None:
        if ra_deg is not None and dec_deg is not None:
            print(f"\n  [Flux] Auto-fetching Galactic N_H from HEASARC HI4PI…")
            nh_used = fetch_nh(ra_deg, dec_deg)
            if nh_used is not None:
                print(f"  [Flux] N_H = {nh_used:.3e} cm^-2  (HI4PI weighted average)")
            else:
                print("  [Flux] WARNING: N_H auto-fetch failed — skipping flux conversion.")
                return None
        else:
            print("  [Flux] WARNING: nh_cm2 not set and no RA/Dec provided — "
                  "skipping flux conversion.")
            return None

    # --- Spectral model settings from cfg ------------------------------------
    model          = getattr(cfg, 'spectral_model',  'powerlaw')
    photon_index   = getattr(cfg, 'photon_index',    2.0)
    temperature_kev= getattr(cfg, 'temperature_kev', 1.0)
    abundance      = getattr(cfg, 'abundance',       1.0)
    redshift       = getattr(cfg, 'redshift',        None)
    cosmology      = getattr(cfg, 'cosmology',       'Planck18')
    h0             = getattr(cfg, 'h0',              67.4)
    omega_m        = getattr(cfg, 'omega_m',         0.315)

    params_str = _model_params_str(model, photon_index, temperature_kev, abundance)

    # Determine which field holds the module/instrument name in each row
    # NuSTAR rows have 'module'; XMM rows have 'instrument'; Swift rows have 'mode'
    inst_field = None
    for candidate in ('module', 'instrument', 'mode'):
        if csv_rows[0].get(candidate) is not None:
            inst_field = candidate
            break

    print(f"\n  [Flux] Converting count-rate ULs → flux  "
          f"(model: {model}, {params_str}, N_H={nh_used:.3e} cm^-2)")

    # Track per-instrument PIMMS codes to avoid redundant lookups
    _pimms_cache = {}
    _flux_cache  = {}   # (pimms_code, cr_str) → flux dict

    for row in csv_rows:
        # Determine which count rate to use
        cr = None
        cr_field_used = None
        for field in ('CR_marg_total', 'CR_marg_aperture'):
            val = row.get(field, '')
            if val not in ('', None):
                try:
                    cr = float(val)
                    cr_field_used = field
                    break
                except (ValueError, TypeError):
                    pass

        if cr is None or cr <= 0:
            # No usable count rate — fill empty fields and continue
            for k in ('nh_cm2_used', 'pimms_instrument', 'spectral_model',
                      'model_params_str', 'flux_ul_cgs', 'flux_ul_unabs_cgs',
                      'lum_ul_cgs'):
                row.setdefault(k, '')
            continue

        # Determine PIMMS instrument code
        inst_val = row.get(inst_field, '') if inst_field else ''
        if inst_val not in _pimms_cache:
            _pimms_cache[inst_val] = pimms_from_map.get(inst_val, '')
        pimms_code = _pimms_cache[inst_val]

        if not pimms_code:
            warnings.warn(
                f"No PIMMS instrument code found for '{inst_val}' — "
                f"skipping flux for this row.", UserWarning)
            for k in ('nh_cm2_used', 'pimms_instrument', 'spectral_model',
                      'model_params_str', 'flux_ul_cgs', 'flux_ul_unabs_cgs',
                      'lum_ul_cgs'):
                row.setdefault(k, '')
            continue

        # Always set metadata fields
        row['nh_cm2_used']      = f"{nh_used:.4e}"
        row['pimms_instrument'] = pimms_code
        row['spectral_model']   = model
        row['model_params_str'] = params_str

        # WebPIMMS (cache by instrument + CR to avoid duplicate POSTs for
        # identical values across confidence-level rows that differ only in CL)
        cache_key = (pimms_code, f"{cr:.6e}")
        if cache_key not in _flux_cache:
            try:
                flux_dict = webpimms_cr_to_flux(
                    pimms_code, cr, e_lo_kev, e_hi_kev, nh_used,
                    model=model,
                    photon_index=photon_index,
                    temperature_kev=temperature_kev,
                    abundance=abundance)
                _flux_cache[cache_key] = flux_dict
            except RuntimeError as exc:
                warnings.warn(f"WebPIMMS conversion failed: {exc}", UserWarning)
                _flux_cache[cache_key] = None

        flux_dict = _flux_cache[cache_key]
        if flux_dict is None:
            row['flux_ul_cgs']       = ''
            row['flux_ul_unabs_cgs'] = ''
            row['lum_ul_cgs']        = ''
            continue

        row['flux_ul_cgs']       = f"{flux_dict['flux_abs']:.4e}"
        row['flux_ul_unabs_cgs'] = f"{flux_dict['flux_unabs']:.4e}"

        # Luminosity
        if redshift is not None and redshift > 0:
            try:
                lum = flux_to_luminosity(
                    flux_dict['flux_unabs'], redshift,
                    cosmology=cosmology, h0=h0, omega_m=omega_m)
                row['lum_ul_cgs'] = f"{lum:.4e}"
            except Exception as exc:
                warnings.warn(f"Luminosity calculation failed: {exc}", UserWarning)
                row['lum_ul_cgs'] = ''
        else:
            row['lum_ul_cgs'] = ''

    # ------------------------------------------------------------------
    # Print per-confidence-level results grouped by instrument/module
    # ------------------------------------------------------------------
    converted = [r for r in csv_rows if r.get('flux_ul_cgs', '')]
    if not converted:
        print("  [Flux] WARNING: No rows were successfully converted.")
        return nh_used

    has_lum = redshift is not None and redshift > 0

    # Collect unique (instrument, result_type) pairs preserving order
    seen_keys = []
    for row in converted:
        inst_key = row.get(inst_field, '') if inst_field else ''
        key = (inst_key, row.get('result_type', ''))
        if key not in seen_keys:
            seen_keys.append(key)

    for (inst_key, rtype) in seen_keys:
        group = [r for r in converted
                 if (r.get(inst_field, '') if inst_field else '') == inst_key
                 and r.get('result_type', '') == rtype]
        if not group:
            continue

        label = inst_key if inst_key else telescope
        print(f"\n  {label}  [{rtype}]  —  flux upper limits:")
        print(f"  {'CL':>8}   {'CR (cts/s)':>13}   "
              f"{'F_abs (erg/cm²/s)':>20}   {'F_unabs (erg/cm²/s)':>21}"
              + (f"   {'L (erg/s)':>14}" if has_lum else ''))
        print(f"  {'-'*8}   {'-'*13}   {'-'*20}   {'-'*21}"
              + (f"   {'-'*14}" if has_lum else ''))

        for row in group:
            try:
                cl_val = float(row.get('confidence_level', 0))
            except (TypeError, ValueError):
                cl_val = 0.0
            cr_str = row.get('CR_marg_total') or row.get('CR_marg_aperture', '')
            try:
                cr_val = float(cr_str)
                cr_fmt = f"{cr_val:.4e}"
            except (TypeError, ValueError):
                cr_fmt = str(cr_str)

            f_abs   = row.get('flux_ul_cgs', '')
            f_unabs = row.get('flux_ul_unabs_cgs', '')
            lum     = row.get('lum_ul_cgs', '')

            line = (f"  {cl_val*100:>7.2f}%   {cr_fmt:>13}   "
                    f"{f_abs:>20}   {f_unabs:>21}")
            if has_lum:
                line += f"   {lum:>14}"
            print(line)

    return nh_used
