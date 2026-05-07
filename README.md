# xray_uplim

**Unified X-ray non-detection upper limit calculator** for NuSTAR, XMM-Newton, Swift XRT, and Chandra ACIS.

Given a source position and an X-ray observation in which the source was not detected, `xray_uplim` computes a Bayesian upper limit on the source count rate and flux at one or more confidence levels. It handles multi-observation co-adding, aperture photometry, encircled energy fraction (EEF) correction, and diagnostic plot generation automatically.

---

## Platform support

| Platform | NuSTAR | Swift | XMM | Chandra |
|----------|--------|-------|-----|---------|
| macOS    | ✓ | ✓ | ✓ | ✓ |
| Linux    | ✓ | ✓ | ✓ | ✓ |
| Windows  | ✓ | ✓ | Limited (SAS not officially supported) | ✗ (CIAO not available) |

> **Windows users**: NuSTAR and Swift pipelines work natively. XMM requires SAS which does not have an official Windows build. Chandra requires CIAO which is Linux/macOS only.

---

## Requirements

### Python
- Python ≥ 3.8

### Python packages

These are installed automatically when you run `pip install .` from the cloned repository (see [Installation](#installation) below).

| Package | Version | 
|---------|---------|
| numpy | ≥ 1.21 |
| scipy | ≥ 1.7 | 
| astropy | ≥ 5.0 | 
| matplotlib | ≥ 3.4 | 
| openpyxl | ≥ 3.0 | 
| pyyaml | ≥ 6.0 | 

PySide6 ≥ 6.4 is an optional dependency installed only by `pip install ".[gui]"`.

### External astronomy software (telescope-specific)

| Telescope | Software | Required for |
|-----------|----------|-------------|
| NuSTAR | [HEASoft](https://heasarc.gsfc.nasa.gov/docs/software/heasoft/) + NuSTAR CALDB | EEF via 2D PSF images (CALDB `bcf/psf/`) |
| Swift | None required | Bundled PSF coefficient file included |
| XMM | [SAS](https://www.cosmos.esa.int/web/xmm-newton/sas) ≥ 20 + CCF files | Event file processing, exposure maps, PSF calibration |
| Chandra | [CIAO](https://cxc.cfa.harvard.edu/ciao/) ≥ 4.15 | `chandra_repro`, `aprates`, `fluximage` |

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/sanjana207298/xray_uplim.git
cd xray_uplim
```

> **Note**: `xray_uplim` is not on PyPI. Installation is from the cloned repository using `pip install .`, which reads `setup.py` and installs all Python dependencies automatically into your active environment.

### 2. Create a dedicated environment (recommended)

Choose one of the following:

**conda:**
```bash
conda create -n xray_uplim python=3.11
conda activate xray_uplim
```

**venv (macOS / Linux):**
```bash
python3 -m venv xray_uplim_env
source xray_uplim_env/bin/activate
```

**venv (Windows):**
```bat
python -m venv xray_uplim_env
xray_uplim_env\Scripts\activate.bat
```

### 3. Install the package

**Core package (CLI only, no GUI):**
```bash
pip install .
```

**With desktop GUI (adds PySide6):**
```bash
pip install ".[gui]"
```

**Editable / development install:**
```bash
pip install -e ".[gui]"
```

Use the editable install if you plan to modify the source — changes take effect immediately without reinstalling.

---

## Environment variables

These must be set **before** running any pipeline. The recommended approach is to add them to your shell profile (`~/.zshrc`, `~/.bashrc`, or `~/.profile`) so they are set automatically in every session.

### NuSTAR — CALDB

NuSTAR calibration files are distributed through the NASA CALDB system (shared with HEASoft).

**Install / update CALDB** (one-time, ~500 MB including NuSTAR):
```bash
# Full instructions: https://heasarc.gsfc.nasa.gov/docs/heasarc/caldb/caldb_install.html
# Quick start after downloading:
tar xzf caldb.tar.gz -C /path/to/caldb
```

**Set environment variables** (add to `~/.zshrc` or `~/.bashrc`):
```bash
export CALDB=/path/to/caldb
export CALDBCONFIG=$CALDB/software/tools/caldb.config
export CALDBALIAS=$CALDB/software/tools/alias_config.fits
```

**Verify:**
```bash
quzcif NUSTAR FPM - - ARF 2020-01-01 0:0:0 -
# should print a list of ARF files
```

The CALDB directory tree that `xray_uplim` reads:
```
$CALDB/
└── data/
    └── nustar/
        └── fpm/
            └── bcf/
                └── psf/        ← 2D PSF image FITS files (one per energy sub-band,
                                   tabulated at discrete off-axis angles 0–8.5')
```

The GUI has a **CALDB directory** field — leave it empty if `$CALDB` is already set in your shell.

### XMM-Newton — SAS and Current Calibration Files (CCF)

**Install SAS** following the official guide:
https://www.cosmos.esa.int/web/xmm-newton/sas-installation

**Initialise SAS** (add to your shell profile, or run before each session):
```bash
source /path/to/xmm/sas/setsas.sh      # adjust path to your SAS installation
# or, if installed via conda:
conda activate sas
```

**Download CCF files** for your observation (one-time per ODF):
```bash
cd /path/to/ODF
cifbuild                  # builds ccf.cif — requires SAS and internet access
export SAS_CCF=$(pwd)/ccf.cif
```

**Set the CCF path** (add to shell profile):
```bash
export SAS_CCFPATH=/path/to/ccf         # directory containing *.CCF files
```

CCF files used by `xray_uplim` for EEF correction:
```
$SAS_CCFPATH/
├── XRT1_XPSF_*.CCF     ← MOS1 PSF calibration
├── XRT2_XPSF_*.CCF     ← MOS2 PSF calibration
└── XRT3_XPSF_*.CCF     ← pn PSF calibration
```

The GUI has a **SAS CCF/PSF directory** field. If SAS is initialised in your shell (`$SAS_CCFPATH` is set), leave it empty.

### Chandra — CIAO

CIAO ships its own CALDB and is best installed via conda:

```bash
conda create -n ciao -c https://cxc.cfa.harvard.edu/conda/ciao \
    -c conda-forge ciao sherpa ds9 ciao-contrib caldb_main
conda activate ciao
```

Or download the standalone installer:
https://cxc.cfa.harvard.edu/ciao/download/

**Verify the installation:**
```bash
ciaover
check_ciao_caldb
echo $ASCDS_INSTALL      # should print the CIAO root directory
```

CIAO sets `$CALDB` automatically when activated. The files used internally by `aprates` are:
```
$CALDB/
└── data/
    └── chandra/
        └── acis/
            ├── eff2evt/     ← Effective area calibration
            ├── psf/         ← PSF maps (used by aprates)
            └── ardlib/      ← Detector response
```

The GUI has a **CIAO prefix** field. If you activated CIAO via conda or the standalone initialiser, leave it empty.

---

## Usage

### Desktop GUI

```bash
xray_uplim
```

A graphical window opens. Select your observatory from the dropdown, fill in the required fields (data directory, ObsID, RA, Dec), and click **Run Pipeline**. Before processing begins, an interactive matplotlib window opens so you can visually confirm the source and background aperture positions on the event image.

Results and diagnostic plots appear in the **Results** tab after the run completes.

### Command-line interface

The primary way to run from the command line is to edit the `CONFIG` block in `run_uplim.py` and run:

```bash
python run_uplim.py
```

Alternatively, use the YAML-based CLI:

```bash
# Print a template config file
xray_uplim-cli --template > config.yaml

# Edit the file, then run
xray_uplim-cli config.yaml

# JSON format is also accepted
xray_uplim-cli config.json
```

Both interfaces accept identical parameter names. See [Configuration Reference](Configuration-Reference) for all options.

---

## Data directory structure

`xray_uplim` expects the standard HEASArc/ESA archive directory layout for each telescope.

### NuSTAR
```
base_path/
└── {obsid}/
    └── event_cl/
        ├── nu{obsid}A01_cl.evt     ← cleaned FPMA event file
        └── nu{obsid}B01_cl.evt     ← cleaned FPMB event file
```
Output → `{base_path}/{obsid}/ul_products/`

### Swift XRT
```
data_dir/
└── {obsid}/
    └── xrt/
        ├── event/
        │   └── sw{obsid}x*_cl.evt      ← cleaned event file (PC or WT mode)
        └── expmap/
            └── sw{obsid}x*_ex.img      ← exposure map
```
Output → `{data_dir}/{obsid}/ul_products/`

### XMM-Newton EPIC
```
data_dir/                 ← ODF working directory (after running emproc / epproc)
├── *EMOS1*ImagingEvts.ds     ← MOS1 event file
├── *EMOS2*ImagingEvts.ds     ← MOS2 event file
├── *EPN*ImagingEvts.ds       ← pn event file
├── mos1_expmap.fits           ← MOS1 exposure map (from eexpmap)
├── mos2_expmap.fits           ← MOS2 exposure map
└── pn_expmap.fits             ← pn exposure map
```
Output → `{data_dir}/ul_products/`

### Chandra ACIS
```
base_path/
└── {obsid}/
    ├── primary/
    │   └── acisf{obsid}N???_evt2.fits.gz   ← Level-2 event file (from archive)
    └── repro/                               ← created automatically by chandra_repro
        └── acisf{obsid}_repro_evt2.fits
```
Output → `{base_path}/{obsid}/ul_products/`

---

## Output files

All output is written to `ul_products/` inside the observation directory:

| File | Description |
|------|-------------|
| `{tel}_uplim_{obsid}.csv` | Results table: counts, exposure, EEF, upper limits at each CL |
| `{tel}_uplim_{obsid}.xlsx` | Same in Excel format |
| `radial_{label}_{band}keV.pdf` | Log-scale radial surface-density profile |
| `expmap_hist_{label}.pdf` | Exposure-map pixel distribution in aperture |
| `regions_{label}_{band}keV.pdf` | Sky image with source and background apertures (vector, for papers) |
| `chandra_regions_{obsid}_{band}keV.pdf` | Sky image for Chandra (PDF) |

---

## License

MIT License — see `LICENSE` for details.
