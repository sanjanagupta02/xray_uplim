"""
xray_uplim.gui.app
------------------
PySide6 desktop GUI for the xray_uplim pipeline.

The pipeline runs as a QProcess (subprocess), so the matplotlib region
selector appears as a normal separate window with no Qt event-loop conflicts.
Stdout/stderr streams live into the log panel.
"""

import glob
import json
import os
import sys
import tempfile
from collections import OrderedDict

from PySide6.QtCore    import Qt, QProcess, QSize, Signal, QEvent, QThread, QTimer
from PySide6.QtGui     import QFont, QColor, QTextCursor, QPixmap, \
                               QStandardItemModel, QStandardItem

try:
    from PySide6.QtPdf import QPdfDocument
    _PDF_SUPPORT = True
except ImportError:
    _PDF_SUPPORT = False
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QFormLayout, QGroupBox, QLabel, QLineEdit, QDoubleSpinBox,
    QComboBox, QCheckBox, QPushButton, QScrollArea, QSplitter,
    QTextEdit, QFileDialog, QStackedWidget, QStatusBar,
    QMessageBox, QTabWidget, QSizePolicy,
)


# =============================================================================
# Energy band definitions  {key → display label}
# Telescope energy limits (keV) for custom band spin boxes
# =============================================================================

NUSTAR_BANDS = OrderedDict([
    ('full',       'full  ·  3–79 keV'),
    ('extra-soft', 'extra-soft  ·  3–4.5 keV'),
    ('soft',       'soft  ·  4.5–6 keV'),
    ('iron',       'iron  ·  6–8 keV'),
    ('medium',     'medium  ·  8–12 keV'),
    ('hard',       'hard  ·  12–20 keV'),
    ('ultra-hard', 'ultra-hard  ·  20–79 keV'),
])
NUSTAR_ELIM = (3.0, 79.0)

XMM_BANDS = OrderedDict([
    ('full',      'full  ·  0.2–12 keV'),
    ('soft',      'soft  ·  0.5–2 keV'),
    ('hard',      'hard  ·  2–10 keV'),
    ('medium',    'medium  ·  1–2 keV'),
    ('ultrasoft', 'ultrasoft  ·  0.2–0.5 keV'),
])
XMM_ELIM = (0.2, 12.0)

SWIFT_BANDS = OrderedDict([
    ('full',      'full  ·  0.3–10 keV'),
    ('soft',      'soft  ·  0.3–1.5 keV'),
    ('hard',      'hard  ·  1.5–10 keV'),
    ('ultrasoft', 'ultrasoft  ·  0.3–1 keV'),
])
SWIFT_ELIM = (0.3, 10.0)

CHANDRA_BANDS = OrderedDict([
    ('broad',     'broad  ·  0.5–7 keV'),
    ('soft',      'soft  ·  0.5–2 keV'),
    ('medium',    'medium  ·  2–4 keV'),
    ('hard',      'hard  ·  4–7 keV'),
    ('full',      'full  ·  0.5–10 keV'),
    ('ultrasoft', 'ultrasoft  ·  0.3–1 keV'),
])
CHANDRA_ELIM = (0.1, 10.0)


# =============================================================================
# ObsID auto-detection helpers
# =============================================================================

def _scan_obsids(data_dir, validator):
    """Return sorted list of subdirectory names in data_dir that pass validator."""
    if not os.path.isdir(data_dir):
        return []
    return sorted(
        name for name in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, name))
        and validator(os.path.join(data_dir, name))
    )

def _valid_nustar(p):  return os.path.isdir(os.path.join(p, 'event_cl'))
def _valid_swift(p):   return os.path.isdir(os.path.join(p, 'xrt', 'event'))
def _valid_xmm(p):
    odf = os.path.join(p, 'ODF')
    return os.path.isdir(odf) and bool(glob.glob(os.path.join(odf, '*ImagingEvts.ds')))
def _valid_chandra(p): return os.path.isdir(os.path.join(p, 'primary'))


# =============================================================================
# Widget helpers
# =============================================================================

def _group(title: str) -> tuple:
    box = QGroupBox(title)
    box.setStyleSheet("QGroupBox { font-weight: bold; }")
    lay = QFormLayout()
    lay.setHorizontalSpacing(12)
    lay.setVerticalSpacing(6)
    box.setLayout(lay)
    return box, lay


def _line(default: str = '', placeholder: str = '', tooltip: str = '') -> QLineEdit:
    w = QLineEdit(default)
    if placeholder: w.setPlaceholderText(placeholder)
    if tooltip:     w.setToolTip(tooltip)
    return w


def _spin(default: float, lo: float, hi: float, step: float = 1.0,
          decimals: int = 1, tooltip: str = '') -> QDoubleSpinBox:
    w = QDoubleSpinBox()
    w.setRange(lo, hi)
    w.setSingleStep(step)
    w.setDecimals(decimals)
    w.setValue(default)
    if tooltip: w.setToolTip(tooltip)
    return w


def _combo(options: list, default: str = '', tooltip: str = '') -> QComboBox:
    w = QComboBox()
    w.addItems(options)
    if default in options: w.setCurrentText(default)
    if tooltip: w.setToolTip(tooltip)
    return w


def _path_row(form: QFormLayout, label: str, default: str = '',
              placeholder: str = '', file_mode: bool = False,
              tooltip: str = '') -> QLineEdit:
    edit = _line(default, placeholder, tooltip)
    btn  = QPushButton('Browse…')
    btn.setMaximumWidth(80)

    def _open():
        path = (QFileDialog.getOpenFileName(caption=f'Select {label}')[0]
                if file_mode else
                QFileDialog.getExistingDirectory(caption=f'Select {label}'))
        if path: edit.setText(path)

    btn.clicked.connect(_open)
    row = QWidget()
    h   = QHBoxLayout(row)
    h.setContentsMargins(0, 0, 0, 0)
    h.addWidget(edit)
    h.addWidget(btn)
    form.addRow(label, row)
    return edit


class _CheckableComboBox(QComboBox):
    """
    QComboBox where every item has a checkbox.
    The popup stays open while the user checks / unchecks items;
    it closes only when the user clicks outside or presses Escape.
    """
    selectionChanged = Signal(list)   # emits list[str] of checked texts

    def __init__(self, placeholder='No ObsIDs detected — set data directory first',
                 parent=None):
        super().__init__(parent)
        self.setEditable(True)
        self.lineEdit().setReadOnly(True)
        self.lineEdit().setPlaceholderText(placeholder)
        self._placeholder = placeholder

        self._model = QStandardItemModel(self)
        self.setModel(self._model)
        self._model.itemChanged.connect(self._on_item_changed)
        self._updating = False

        # Match the width of the path-row browse fields
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setMinimumWidth(260)

        # Intercept mouse clicks so popup stays open when checking items
        self.view().viewport().installEventFilter(self)

    def populate(self, items):
        """Replace all items; all checked by default."""
        self._updating = True
        self._model.clear()
        for text in items:                          # already sorted by caller
            item = QStandardItem(text)
            item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked)
            self._model.appendRow(item)
        self._updating = False
        self._refresh_text()
        self.selectionChanged.emit(self.checked_items())

    def clear_items(self):
        self._model.clear()
        self.lineEdit().clear()

    # ------------------------------------------------------------------
    def eventFilter(self, obj, event):
        """Swallow mouse-release on checkbox items so popup does not close."""
        if (obj is self.view().viewport() and
                event.type() == QEvent.MouseButtonRelease):
            idx = self.view().indexAt(event.pos())
            if idx.isValid():
                item = self._model.itemFromIndex(idx)
                if item and item.isCheckable():
                    new = (Qt.Unchecked if item.checkState() == Qt.Checked
                           else Qt.Checked)
                    item.setCheckState(new)
                    return True     # swallow — keep popup open
        return super().eventFilter(obj, event)

    # ------------------------------------------------------------------
    def _on_item_changed(self, item):
        if not self._updating:
            self._refresh_text()
            self.selectionChanged.emit(self.checked_items())

    def _refresh_text(self):
        checked = self.checked_items()
        self.lineEdit().setText(', '.join(checked) if checked else '')

    def checked_items(self):
        return [self._model.item(i).text()
                for i in range(self._model.rowCount())
                if self._model.item(i).checkState() == Qt.Checked]

    def has_selection(self):
        return bool(self.checked_items())


def _energy_band_widget(bands_dict: OrderedDict, default: str,
                        e_min: float, e_max: float) -> tuple:
    """
    Combo of named bands + Custom option.
    Custom spin boxes are clamped to [e_min, e_max] for the telescope.
    get_value() returns the pipeline key string, or [lo, hi] for custom.
    """
    outer  = QWidget()
    layout = QHBoxLayout(outer)
    layout.setContentsMargins(0, 0, 0, 0)

    key_for = {v: k for k, v in bands_dict.items()}
    combo   = QComboBox()
    combo.addItems(list(bands_dict.values()) + ['Custom…'])
    if default in bands_dict:
        combo.setCurrentText(bands_dict[default])

    lo_spin = _spin(e_min, e_min, e_max, 0.1, 2)
    hi_spin = _spin(e_max, e_min, e_max, 0.1, 2)
    lo_spin.setPrefix('lo: ')
    hi_spin.setPrefix('hi: ')
    lo_spin.setSuffix(' keV')
    hi_spin.setSuffix(' keV')
    lo_spin.setToolTip(f'Low energy (min {e_min} keV for this telescope)')
    hi_spin.setToolTip(f'High energy (max {e_max} keV for this telescope)')
    lo_spin.setMaximumWidth(120)
    hi_spin.setMaximumWidth(120)
    dash = QLabel('–')
    for w in (lo_spin, hi_spin, dash):
        w.hide()

    def _on_change(text):
        custom = (text == 'Custom…')
        lo_spin.setVisible(custom)
        dash.setVisible(custom)
        hi_spin.setVisible(custom)

    combo.currentTextChanged.connect(_on_change)
    layout.addWidget(combo)
    layout.addWidget(lo_spin)
    layout.addWidget(dash)
    layout.addWidget(hi_spin)
    layout.addStretch()

    def get_value():
        txt = combo.currentText()
        if txt == 'Custom…':
            return [lo_spin.value(), hi_spin.value()]
        return key_for.get(txt, txt)

    return outer, get_value


def _cl_widget() -> tuple:
    outer = QWidget()
    vlay  = QVBoxLayout(outer)
    vlay.setContentsMargins(0, 0, 0, 0)
    vlay.setSpacing(4)

    preset = QWidget()
    play   = QHBoxLayout(preset)
    play.setContentsMargins(0, 0, 0, 0)
    cb1 = QCheckBox('1σ  (68.3%)')
    cb2 = QCheckBox('2σ  (95.5%)')
    cb3 = QCheckBox('3σ  (99.7%)')
    for cb in (cb1, cb2, cb3):
        cb.setChecked(True)
        play.addWidget(cb)
    play.addStretch()
    vlay.addWidget(preset)

    custom_row  = QWidget()
    clay        = QHBoxLayout(custom_row)
    clay.setContentsMargins(0, 0, 0, 0)
    cb_custom   = QCheckBox('Custom:')
    custom_edit = QLineEdit()
    custom_edit.setPlaceholderText('e.g.  0.90, 0.95')
    custom_edit.setEnabled(False)
    custom_edit.setToolTip('Comma-separated CL values in (0, 1)')
    cb_custom.toggled.connect(custom_edit.setEnabled)
    clay.addWidget(cb_custom)
    clay.addWidget(custom_edit)
    vlay.addWidget(custom_row)

    def get_value():
        out = []
        if cb1.isChecked(): out.append(0.6827)
        if cb2.isChecked(): out.append(0.9545)
        if cb3.isChecked(): out.append(0.9973)
        if cb_custom.isChecked():
            for tok in custom_edit.text().split(','):
                try:
                    v = float(tok.strip())
                    if 0.0 < v < 1.0:
                        out.append(v)
                except ValueError:
                    pass
        return sorted(set(out)) or [0.9973]

    return outer, get_value


# =============================================================================
# =============================================================================
# SIMBAD / NED name resolver — runs in a background thread
# =============================================================================

class _NameResolverThread(QThread):
    """
    Resolves a source name to RA/Dec using the CDS Sesame service
    (https://cdsweb.u-strasbg.fr/cgi-bin/nph-sesame).

    Sesame is the same lightweight endpoint used by Aladin, ESASky, and most
    other astronomy tools — typically responds in under one second.  It queries
    SIMBAD, NED, and VizieR in sequence and returns the first match.
    No extra dependencies: uses only Python's built-in urllib and xml modules.
    """
    resolved = Signal(float, float, str)   # ra_deg, dec_deg, catalogue_name
    failed   = Signal(str)                 # error message

    def __init__(self, name, parent=None):
        super().__init__(parent)
        self._name = name

    def run(self):
        import urllib.request
        import urllib.parse
        import xml.etree.ElementTree as ET

        url = ('https://cdsweb.u-strasbg.fr/cgi-bin/nph-sesame/-ox?'
               + urllib.parse.quote(self._name))
        try:
            with urllib.request.urlopen(url, timeout=10) as resp:
                xml_data = resp.read()
        except Exception as exc:
            self.failed.emit(f'Network error: {exc}')
            return

        try:
            root = ET.fromstring(xml_data)
            for resolver in root.iter('Resolver'):
                ra_el  = resolver.find('jradeg')
                dec_el = resolver.find('jdedeg')
                if ra_el is not None and dec_el is not None:
                    src = resolver.get('name', 'Sesame')
                    if   'imbad' in src: src = 'SIMBAD'
                    elif 'NED'   in src: src = 'NED'
                    else:                src = 'Sesame'
                    self.resolved.emit(float(ra_el.text),
                                       float(dec_el.text), src)
                    return
        except Exception as exc:
            self.failed.emit(f'Failed to parse Sesame response: {exc}')
            return

        self.failed.emit(f"'{self._name}' not found in SIMBAD or NED.")


# =============================================================================
# Per-observatory config forms
# =============================================================================

class _BaseForm(QWidget):

    # Subclasses populate these in __init__ so MainWindow can watch them
    _ra:    QLineEdit
    _dec:   QLineEdit
    _obsid: QLineEdit

    def __init__(self):
        super().__init__()
        self._main_layout = QVBoxLayout(self)
        self._main_layout.setSpacing(8)
        self._main_layout.setContentsMargins(6, 6, 6, 6)

    # ---- shared section builders --------------------------------------------

    def _add_source_position(self):
        box, lay = _group('Source position')

        # --- Name resolver row -------------------------------------------
        name_row = QWidget()
        name_lay = QHBoxLayout(name_row)
        name_lay.setContentsMargins(0, 0, 0, 0)
        name_lay.setSpacing(6)
        self._src_name    = _line('', 'e.g.  SN 2012ap')
        self._resolve_btn = QPushButton('Resolve →')
        self._resolve_btn.setMaximumWidth(90)
        self._resolve_btn.setToolTip(
            'Query SIMBAD (then NED as fallback) and fill RA / Dec automatically.')
        self._resolve_btn.clicked.connect(self._resolve_name)
        name_lay.addWidget(self._src_name)
        name_lay.addWidget(self._resolve_btn)
        lay.addRow('Source name', name_row)

        self._ra  = _line('', 'HH:MM:SS.ss  or decimal deg')
        self._dec = _line('', '±DD:MM:SS.ss or decimal deg')
        lay.addRow('RA',  self._ra)
        lay.addRow('Dec', self._dec)
        self._main_layout.addWidget(box)
        self._resolver_thread = None

    # --- Name resolver slots -------------------------------------------------

    def _resolve_name(self):
        name = self._src_name.text().strip()
        if not name:
            return
        self._resolve_btn.setText('Resolving…')
        self._resolve_btn.setEnabled(False)
        self._resolver_thread = _NameResolverThread(name, self)
        self._resolver_thread.resolved.connect(self._on_resolved)
        self._resolver_thread.failed.connect(self._on_resolve_failed)
        self._resolver_thread.start()

    def _on_resolved(self, ra_deg, dec_deg, catalogue):
        self._ra.setText(f'{ra_deg:.6f}')
        self._dec.setText(f'{dec_deg:.6f}')
        self._resolve_btn.setText(f'✓ {catalogue}')
        self._resolve_btn.setEnabled(True)
        QTimer.singleShot(2500, lambda: self._resolve_btn.setText('Resolve →'))

    def _on_resolve_failed(self, msg):
        QMessageBox.warning(self, 'Name not resolved', msg)
        self._resolve_btn.setText('Resolve →')
        self._resolve_btn.setEnabled(True)

    def _add_aperture(self, src_def=60.0, bkg_def=200.0, factor_def=1.2,
                      psf_def=18.0, src_min=1.0, src_max=500.0,
                      bkg_min=10.0, bkg_max=1000.0, psf_note=''):
        # Store for use by _add_options() which places the widget inline
        self._psf_def  = psf_def
        self._psf_note = psf_note

        box, lay = _group('Aperture')
        self._src_r   = _spin(src_def,    src_min, src_max,  1.0, 1,
                              'Source extraction radius (arcsec)')
        self._bkg_r   = _spin(bkg_def,    bkg_min, bkg_max,  5.0, 1,
                              'Background annulus outer radius (arcsec)')
        self._bkg_fac = _spin(factor_def, 1.0,     5.0,      0.05, 2,
                              'Annulus inner radius = src_radius × this factor')
        lay.addRow('Source radius (")',    self._src_r)
        lay.addRow('Bkg outer radius (")', self._bkg_r)
        lay.addRow('Bkg inner factor',     self._bkg_fac)
        self._main_layout.addWidget(box)

    def _add_energy_band(self, bands_dict, default, e_min, e_max):
        box, lay = _group('Energy band')
        widget, self._get_band = _energy_band_widget(
            bands_dict, default, e_min, e_max)
        lay.addRow('Band', widget)
        self._main_layout.addWidget(box)

    def _add_background_mode(self):
        box, lay = _group('Background region')
        self._bkg_mode = _combo(
            ['annulus', 'manual'], 'annulus',
            '"annulus": background taken from an annulus around the source.\n'
            '"manual": place a separate background circle at a custom RA/Dec.')
        lay.addRow('Mode', self._bkg_mode)

        self._bkg_ra        = _line('', 'e.g. 05:00:20.0')
        self._bkg_dec       = _line('', 'e.g. -03:21:00.0')
        self._bkg_ra_label  = QLabel('Bkg RA')
        self._bkg_dec_label = QLabel('Bkg Dec')
        lay.addRow(self._bkg_ra_label,  self._bkg_ra)
        lay.addRow(self._bkg_dec_label, self._bkg_dec)

        def _toggle(text):
            manual = (text == 'manual')
            for w in (self._bkg_ra, self._bkg_dec,
                      self._bkg_ra_label, self._bkg_dec_label):
                w.setVisible(manual)

        self._bkg_mode.currentTextChanged.connect(_toggle)
        _toggle('annulus')
        self._main_layout.addWidget(box)

    def _add_confidence(self):
        box, lay = _group('Confidence levels')
        cl_widget, self._get_cl = _cl_widget()
        lay.addRow('', cl_widget)
        self._main_layout.addWidget(box)

    def _add_options(self, exp_stat_opts=None):
        box, lay = _group('Options')
        if exp_stat_opts is None:
            exp_stat_opts = ['median', 'mean', 'psf_weighted']
        self._exp_stat = _combo(exp_stat_opts, 'median',
            'How to summarise the exposure map across the aperture.\n\n'
            'median       — robust against outliers; recommended for most cases\n'
            'mean         — simple arithmetic average\n'
            'psf_weighted — weights each pixel by its PSF fraction, giving more\n'
            '               weight to central pixels where the source is brightest.\n'
            '               Slightly more accurate in theory, but sensitive to\n'
            '               PSF model errors.  Stick with median unless you have\n'
            '               a specific reason.')
        self._use_gui     = QCheckBox()
        self._use_gui.setChecked(True)
        self._use_gui.setToolTip(
            'Open the interactive region selector before processing.\n'
            'Lets you visually place source and background apertures on the image.')
        self._gui_per_obs = QCheckBox()
        self._gui_per_obs.setChecked(False)
        self._gui_per_obs.setToolTip(
            'Show a separate region selector for each observation.\n'
            'Use when pointings differ or the source is near a chip edge in some obs.')
        self._save_plots  = QCheckBox()
        self._save_plots.setChecked(True)
        self._save_plots.setToolTip(
            'Save diagnostic images to <data_dir>/ul_products/')
        # PSF FWHM — inline, right below Exposure stat, hidden until psf_weighted chosen
        psf_note = getattr(self, '_psf_note', '')
        psf_def  = getattr(self, '_psf_def',  18.0)
        psf_tip  = (
            'Only used by the "PSF-weighted" exposure statistic.\n'
            'Weights exposure-map pixels by a Gaussian centred on the source\n'
            '(σ = FWHM / 2.355), so central pixels near the source core count more.\n\n'
            + (psf_note or
               'Use the on-axis value for on-axis sources;\n'
               'increase it for off-axis sources where the PSF broadens.'))
        self._psf       = _spin(psf_def, 0.1, 120.0, 0.5, 2, psf_tip)
        self._psf_label = QLabel('PSF FWHM (")')
        self._psf_label.setToolTip(psf_tip)

        lay.addRow('Exposure stat',   self._exp_stat)
        lay.addRow(self._psf_label,   self._psf)
        lay.addRow('Interactive GUI', self._use_gui)
        lay.addRow('GUI per obs',     self._gui_per_obs)
        lay.addRow('Save plots',      self._save_plots)
        self._main_layout.addWidget(box)

        # Show/hide PSF FWHM based on selected exposure stat
        def _toggle_psf(text):
            vis = (text == 'psf_weighted')
            self._psf.setVisible(vis)
            self._psf_label.setVisible(vis)

        self._exp_stat.currentTextChanged.connect(_toggle_psf)
        _toggle_psf(self._exp_stat.currentText())   # correct initial state

    # ---- config extraction --------------------------------------------------

    def _base_config(self) -> dict:
        return {
            'ra':                self._ra.text().strip(),
            'dec':               self._dec.text().strip(),
            'src_radius_arcsec': self._src_r.value(),
            'bkg_radius_arcsec': self._bkg_r.value(),
            'bkg_inner_factor':  self._bkg_fac.value(),
            'psf_fwhm_arcsec':   self._psf.value(),
            'energy_band':       self._get_band(),
            'bkg_mode':          self._bkg_mode.currentText(),
            'bkg_ra':            self._bkg_ra.text().strip(),
            'bkg_dec':           self._bkg_dec.text().strip(),
            'confidence_levels': self._get_cl(),
            'exp_stat':          self._exp_stat.currentText(),
            'use_gui':           self._use_gui.isChecked(),
            'gui_per_obs':       self._gui_per_obs.isChecked(),
            'save_plots':        True,
            'src_name':          self._src_name.text().strip(),
        }

    def get_config(self) -> dict:
        raise NotImplementedError

    def data_path(self) -> str:
        for attr in ('_base_path', '_data_dir'):
            w = getattr(self, attr, None)
            if w is not None:
                return w.text().strip()
        return ''

    # ---- validation helper --------------------------------------------------

    def is_ready(self) -> bool:
        """True when the minimum required fields are filled."""
        obsid_w = getattr(self, '_obsid', None)
        if isinstance(obsid_w, _CheckableComboBox):
            obsid_ok = obsid_w.has_selection()
        else:
            obsid_ok = bool(obsid_w.text().strip()) if obsid_w else True
        return bool(self.data_path() and obsid_ok and
                    self._ra.text().strip() and self._dec.text().strip())

    def connect_change(self, callback):
        """Connect all key fields to callback so run-button can be toggled."""
        obsid_w = getattr(self, '_obsid', None)
        if isinstance(obsid_w, _CheckableComboBox):
            obsid_w.selectionChanged.connect(lambda _: callback())
        elif obsid_w:
            obsid_w.textChanged.connect(callback)
        for attr in ('_base_path', '_data_dir'):
            w = getattr(self, attr, None)
            if w: w.textChanged.connect(callback)
        self._ra.textChanged.connect(callback)
        self._dec.textChanged.connect(callback)

    def _setup_obsid_scan(self, dir_widget, validator):
        """Auto-populate self._obsid when dir_widget path changes."""
        def _scan(path):
            obsids = _scan_obsids(path.strip(), validator)
            if obsids:
                self._obsid.populate(obsids)
            else:
                self._obsid.clear_items()
        dir_widget.textChanged.connect(_scan)


# ---------------------------------------------------------------------------
# NuSTAR
# ---------------------------------------------------------------------------

class NuSTARForm(_BaseForm):

    def __init__(self):
        super().__init__()

        box, lay = _group('Observation')
        self._base_path = _path_row(lay, 'Data directory', '',
                                    '/path/to/NuSTAR/data/')
        self._obsid     = _CheckableComboBox()
        self._caldb     = _path_row(lay, 'CALDB directory', '',
                                    'Leave empty to use $CALDB env variable')
        lay.addRow('ObsID(s)', self._obsid)
        self._setup_obsid_scan(self._base_path, _valid_nustar)
        self._main_layout.addWidget(box)

        self._add_source_position()
        self._add_aperture(src_def=60.0, bkg_def=200.0, factor_def=1.2,
                           psf_def=18.0,
                           psf_note='NuSTAR on-axis FWHM ≈ 18".  '
                                    'Increase for off-axis sources.')
        self._add_energy_band(NUSTAR_BANDS, 'full', *NUSTAR_ELIM)

        box2, lay2 = _group('Focal-plane modules')
        self._mod_a = QCheckBox('Module A')
        self._mod_b = QCheckBox('Module B')
        self._mod_a.setChecked(True)
        self._mod_b.setChecked(True)
        row = QWidget()
        row.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        rh  = QHBoxLayout(row)
        rh.setContentsMargins(0, 0, 0, 0)
        rh.setSpacing(24)
        rh.addWidget(self._mod_a)
        rh.addWidget(self._mod_b)
        rh.addStretch()
        lay2.addRow(row)          
        self._main_layout.addWidget(box2)

        self._add_background_mode()
        self._add_confidence()
        self._add_options()

        box3, lay3 = _group('PSF spectral weighting')
        self._psf_gamma = _spin(2.0, 0.0, 5.0, 0.1, 2,
            'Photon index Γ used to weight the multi-band exposure map.\n'
            '2.0 = soft source prior,  1.7 = harder,  0.0 = flat spectrum.')
        lay3.addRow('Photon index Γ', self._psf_gamma)
        self._main_layout.addWidget(box3)
        self._main_layout.addStretch()

    def get_config(self) -> dict:
        obsid = self._obsid.checked_items()
        if len(obsid) == 1: obsid = obsid[0]
        modules = (['A'] if self._mod_a.isChecked() else []) + \
                  (['B'] if self._mod_b.isChecked() else [])
        cfg = self._base_config()
        cfg.update({
            'base_path': self._base_path.text().strip(),
            'obsid':     obsid,
            'caldb_dir': self._caldb.text().strip(),
            'modules':   modules,
            'psf_gamma': self._psf_gamma.value(),
        })
        return cfg


# ---------------------------------------------------------------------------
# Swift
# ---------------------------------------------------------------------------

class SwiftForm(_BaseForm):

    def __init__(self):
        super().__init__()

        box, lay = _group('Observation')
        self._data_dir = _path_row(lay, 'Data directory', '',
                                   '/path/to/Swift/data/')
        self._obsid    = _CheckableComboBox()
        lay.addRow('ObsID(s)', self._obsid)
        self._setup_obsid_scan(self._data_dir, _valid_swift)
        self._main_layout.addWidget(box)

        self._add_source_position()
        self._add_aperture(src_def=20.0, bkg_def=80.0, factor_def=1.5,
                           psf_def=6.0, src_min=2.0, src_max=200.0,
                           bkg_min=10.0, bkg_max=500.0,
                           psf_note='Swift XRT on-axis FWHM ≈ 6".  '
                                    'Increase for off-axis sources.')
        self._add_energy_band(SWIFT_BANDS, 'full', *SWIFT_ELIM)
        self._add_background_mode()
        self._add_confidence()
        self._add_options()

        # PSF override — optional, at the bottom
        box2, lay2 = _group('Advanced')
        self._psf_file = _path_row(
            lay2, 'Override PSF file', '',
            'Leave empty — bundled calibration used by default',
            file_mode=True,
            tooltip='The bundled psfconst_xrt.fits (identical to HEASoft 6.36 / XIMAGE)\n'
                    'is used automatically.  Only set this if you have a custom or\n'
                    'newer PSF coefficient file you want to use instead.')
        self._main_layout.addWidget(box2)
        self._main_layout.addStretch()

    def get_config(self) -> dict:
        obsid = self._obsid.checked_items()
        if len(obsid) == 1: obsid = obsid[0]
        cfg = self._base_config()
        cfg.update({
            'data_dir': self._data_dir.text().strip(),
            'obsid':    obsid,
            'psf_file': self._psf_file.text().strip(),
        })
        return cfg


# ---------------------------------------------------------------------------
# XMM-Newton
# ---------------------------------------------------------------------------

class XMMForm(_BaseForm):

    def __init__(self):
        super().__init__()

        box, lay = _group('Observation')
        self._data_dir = _path_row(lay, 'Data directory', '',
                                   '/path/to/XMM/object/')
        self._obsid    = _CheckableComboBox()
        self._psf_dir  = _path_row(lay, 'SAS CCF/PSF directory', '',
                                   'Leave empty — $SAS_CCFPATH used automatically',
                                   tooltip='Directory containing XRT?_XPSF_*.CCF files.\n'
                                           'Leave empty if SAS is initialised in your shell\n'
                                           '(sasinit / conda activate sas).')
        lay.addRow('ObsID(s)', self._obsid)
        self._setup_obsid_scan(self._data_dir, _valid_xmm)
        self._main_layout.addWidget(box)

        self._add_source_position()
        self._add_aperture(src_def=20.0, bkg_def=60.0, factor_def=1.5,
                           psf_def=5.0, src_min=2.0, src_max=200.0,
                           bkg_min=10.0, bkg_max=400.0,
                           psf_note='XMM MOS on-axis FWHM ≈ 4.5",  pn ≈ 6".\n'
                                    'Increase for off-axis sources.')
        self._add_energy_band(XMM_BANDS, 'full', *XMM_ELIM)

        box2, lay2 = _group('Instruments')
        self._mos1 = QCheckBox('MOS1')
        self._mos2 = QCheckBox('MOS2')
        self._pn   = QCheckBox('pn')
        for cb in (self._mos1, self._mos2, self._pn):
            cb.setChecked(True)
        row = QWidget()
        row.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        rh  = QHBoxLayout(row)
        rh.setContentsMargins(0, 0, 0, 0)
        rh.setSpacing(24)
        for cb in (self._mos1, self._mos2, self._pn):
            rh.addWidget(cb)
        rh.addStretch()
        lay2.addRow(row)          # no label — group title already says "Instruments"
        self._main_layout.addWidget(box2)

        self._add_background_mode()
        self._add_confidence()
        self._add_options()
        self._main_layout.addStretch()

    def get_config(self) -> dict:
        obsid = self._obsid.checked_items()
        if len(obsid) == 1: obsid = obsid[0]
        instruments = ((['MOS1'] if self._mos1.isChecked() else []) +
                       (['MOS2'] if self._mos2.isChecked() else []) +
                       (['PN']   if self._pn.isChecked()   else []))
        cfg = self._base_config()
        cfg.update({
            'data_dir':    self._data_dir.text().strip(),
            'obsid':       obsid,
            'instruments': instruments,
            'psf_dir':     self._psf_dir.text().strip(),
        })
        return cfg


# ---------------------------------------------------------------------------
# Chandra
# ---------------------------------------------------------------------------

class ChandraForm(_BaseForm):

    def __init__(self):
        super().__init__()

        info = QLabel(
            '⚠  Requires CIAO.  If already activated in your shell '
            '(conda activate ciao-4.XX) the prefix below can be left empty.')
        info.setWordWrap(True)
        info.setStyleSheet(
            'background:#fff3cd; color:#856404; padding:6px; border-radius:4px;')
        self._main_layout.addWidget(info)

        box, lay = _group('Observation')
        self._base_path   = _path_row(lay, 'Data directory', '',
                                      '/path/to/Chandra/data/')
        self._obsid       = _CheckableComboBox()
        self._ciao_prefix = _path_row(
            lay, 'CIAO prefix', '',
            'e.g. /Applications/ciao-4.18  (leave empty if CIAO is active in shell)',
            tooltip='Root directory of the CIAO installation.\n'
                    'Auto-detected from: $ASCDS_INSTALL, PATH,\n'
                    'and /Applications/ciao-4.XX (macOS standalone installer).\n'
                    'Only set this if auto-detection fails.')
        lay.addRow('ObsID(s)', self._obsid)
        self._setup_obsid_scan(self._base_path, _valid_chandra)
        self._main_layout.addWidget(box)

        self._add_source_position()
        self._add_aperture(src_def=5.0, bkg_def=15.0, factor_def=1.0,
                           psf_def=0.9, src_min=0.5, src_max=60.0,
                           bkg_min=2.0, bkg_max=120.0,
                           psf_note='Chandra on-axis ACIS FWHM ≈ 0.5–1".\n'
                                    'Increases strongly off-axis; use the actual\n'
                                    'PSF size at your source position if off-axis.')
        self._add_energy_band(CHANDRA_BANDS, 'broad', *CHANDRA_ELIM)
        self._add_background_mode()
        self._add_confidence()

        box2, lay2 = _group('CIAO pipeline options')
        self._run_repro   = QCheckBox()
        self._run_repro.setChecked(True)
        self._run_repro.setToolTip(
            'Run chandra_repro automatically if the repro/ directory is missing.\n\n'
            'chandra_repro converts raw Level-1 data into a calibrated Level-2\n'
            'event file — a one-time step.  Subsequent runs reuse repro/ and skip it.')
        self._use_aprates = QCheckBox()
        self._use_aprates.setChecked(True)
        self._use_aprates.setToolTip(
            'Use CIAO aprates for the Bayesian upper limit (recommended).\n\n'
            'aprates marginalises over the unknown background rate and accounts\n'
            'for the Chandra exposure map.  The pure-Python marginalized UL\n'
            'is always computed as a cross-check.')
        lay2.addRow('Run chandra_repro', self._run_repro)
        lay2.addRow('Use aprates',       self._use_aprates)
        self._main_layout.addWidget(box2)

        box3, lay3 = _group('Options')
        self._use_gui     = QCheckBox()
        self._use_gui.setChecked(True)
        self._use_gui.setToolTip('Open the interactive region selector')
        self._gui_per_obs = QCheckBox()
        self._gui_per_obs.setChecked(False)
        self._save_plots  = QCheckBox()
        self._save_plots.setChecked(True)

        # PSF FWHM — always shown for Chandra because it feeds the Gaussian EEF model
        psf_tip = (
            'Gaussian FWHM used to compute the Encircled Energy Fraction (EEF).\n'
            'The EEF corrects aperture counts → total source counts.\n\n'
            'Chandra on-axis ACIS FWHM ≈ 0.5–1".\n'
            'Increases strongly off-axis; enter the actual PSF size at your\n'
            'source position if it is significantly off-axis.')
        self._psf       = _spin(getattr(self, '_psf_def', 0.9), 0.1, 30.0, 0.1, 2, psf_tip)
        self._psf_label = QLabel('PSF FWHM (")')
        self._psf_label.setToolTip(psf_tip)

        lay3.addRow('Interactive GUI', self._use_gui)
        lay3.addRow('GUI per obs',     self._gui_per_obs)
        lay3.addRow('Save plots',      self._save_plots)
        lay3.addRow(self._psf_label,   self._psf)
        self._main_layout.addWidget(box3)
        self._main_layout.addStretch()

    def get_config(self) -> dict:
        obsid = self._obsid.checked_items()
        if len(obsid) == 1: obsid = obsid[0]
        return {
            'base_path':         self._base_path.text().strip(),
            'obsid':             obsid,
            'ra':                self._ra.text().strip(),
            'dec':               self._dec.text().strip(),
            'src_radius_arcsec': self._src_r.value(),
            'bkg_radius_arcsec': self._bkg_r.value(),
            'bkg_inner_factor':  self._bkg_fac.value(),
            'psf_fwhm_arcsec':   self._psf.value(),
            'energy_band':       self._get_band(),
            'bkg_mode':          self._bkg_mode.currentText(),
            'bkg_ra':            self._bkg_ra.text().strip(),
            'bkg_dec':           self._bkg_dec.text().strip(),
            'confidence_levels': self._get_cl(),
            'ciao_prefix':       self._ciao_prefix.text().strip(),
            'run_repro':         self._run_repro.isChecked(),
            'use_aprates':       self._use_aprates.isChecked(),
            'use_gui':           self._use_gui.isChecked(),
            'gui_per_obs':       self._gui_per_obs.isChecked(),
            'save_plots':        True,
        }


# =============================================================================
# Main window
# =============================================================================

class MainWindow(QMainWindow):

    OBS_NAMES = ['NuSTAR', 'Swift', 'XMM', 'Chandra']
    OBS_KEYS  = ['nustar', 'swift', 'xmm', 'chandra']

    def __init__(self):
        super().__init__()
        self.setWindowTitle('xray_uplim — X-ray Upper Limit Calculator')
        self.resize(1200, 820)
        self._process  = None
        self._run_path = ''
        self._build_ui()
        self._apply_style()
        self._check_run_ready()   # start with button greyed out

    # ---- UI -----------------------------------------------------------------

    def _build_ui(self):
        splitter = QSplitter(Qt.Horizontal)
        self.setCentralWidget(splitter)

        # LEFT panel
        left     = QWidget()
        left_lay = QVBoxLayout(left)
        left_lay.setContentsMargins(6, 6, 0, 6)
        left_lay.setSpacing(6)

        obs_row = QWidget()
        obs_h   = QHBoxLayout(obs_row)
        obs_h.setContentsMargins(0, 0, 0, 0)
        obs_h.addWidget(QLabel('<b>Observatory:</b>'))
        self._obs_combo = QComboBox()
        self._obs_combo.addItems(self.OBS_NAMES)
        self._obs_combo.setMinimumWidth(130)
        obs_h.addWidget(self._obs_combo)
        obs_h.addStretch()
        left_lay.addWidget(obs_row)

        self._forms = {
            'nustar':  NuSTARForm(),
            'swift':   SwiftForm(),
            'xmm':     XMMForm(),
            'chandra': ChandraForm(),
        }
        self._stack = QStackedWidget()
        for key in self.OBS_KEYS:
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setWidget(self._forms[key])
            self._stack.addWidget(scroll)

        # Connect observatory switch
        self._obs_combo.currentIndexChanged.connect(self._stack.setCurrentIndex)
        self._obs_combo.currentIndexChanged.connect(
            lambda _: self._check_run_ready())

        # Connect validation to each form's key fields
        for form in self._forms.values():
            form.connect_change(self._check_run_ready)

        left_lay.addWidget(self._stack)

        self._run_btn = QPushButton('▶  Run Pipeline')
        self._run_btn.setMinimumHeight(42)
        self._run_btn.setEnabled(False)
        self._run_btn.setStyleSheet(
            'QPushButton { background:#2d7d46; color:white; font-size:14px;'
            '  font-weight:bold; border-radius:5px; }'
            'QPushButton:hover { background:#3a9d59; }'
            'QPushButton:disabled { background:#aaa; color:#eee; }')
        self._run_btn.clicked.connect(self._run_pipeline)
        left_lay.addWidget(self._run_btn)

        left.setMinimumWidth(400)
        left.setMaximumWidth(500)
        splitter.addWidget(left)

        # RIGHT panel: log + results tabs
        right     = QWidget()
        right_lay = QVBoxLayout(right)
        right_lay.setContentsMargins(0, 6, 6, 6)

        self._tabs = QTabWidget()
        if sys.platform == 'darwin':
            mono = QFont('Menlo', 11)
        elif sys.platform == 'win32':
            mono = QFont('Consolas', 10)
        else:                              # Linux / FreeBSD
            mono = QFont('Liberation Mono', 10)
            if not mono.exactMatch():
                mono = QFont('DejaVu Sans Mono', 10)

        # Log tab
        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setFont(mono)
        self._tabs.addTab(self._log, 'Log')

        # Results tab: scrollable image gallery
        self._results_scroll  = QScrollArea()
        self._results_scroll.setWidgetResizable(True)
        self._results_content = QWidget()
        self._results_layout  = QVBoxLayout(self._results_content)
        self._results_layout.setSpacing(12)
        self._results_scroll.setWidget(self._results_content)
        self._results_scroll.setStyleSheet('background:#2b2b2b;')
        self._tabs.addTab(self._results_scroll, 'Results')

        right_lay.addWidget(self._tabs)
        splitter.addWidget(right)
        splitter.setSizes([440, 760])

        self._status = QStatusBar()
        self.setStatusBar(self._status)
        self._status.showMessage(
            'Fill in data directory, ObsID, RA and Dec to enable Run.')

    # ---- Validation ---------------------------------------------------------

    def _check_run_ready(self):
        form  = self._forms[self._current_key()]
        ready = form.is_ready()
        self._run_btn.setEnabled(
            ready and (self._process is None or
                       self._process.state() == QProcess.NotRunning))
        if ready:
            self._status.showMessage('Ready — press Run Pipeline.')
        else:
            self._status.showMessage(
                'Fill in data directory, ObsID, RA and Dec to enable Run.')

    # ---- Style --------------------------------------------------------------

    def _apply_style(self):
        self.setStyleSheet("""
            QMainWindow, QWidget { background: #f5f5f5; }
            QGroupBox {
                border: 1px solid #ccc; border-radius: 4px;
                margin-top: 8px; padding-top: 4px;
            }
            QGroupBox::title { subcontrol-origin: margin; left: 8px; color: #333; }
            QLineEdit, QDoubleSpinBox, QSpinBox, QComboBox {
                background: white; border: 1px solid #bbb;
                border-radius: 3px; padding: 3px 5px;
            }
            QTextEdit { background: #1e1e1e; color: #d4d4d4; border: none; }
            QTabWidget::pane { border: 1px solid #ccc; }
            QScrollArea { border: none; }
        """)

    # ---- Pipeline execution -------------------------------------------------

    def _current_key(self) -> str:
        return self.OBS_KEYS[self._obs_combo.currentIndex()]

    def _run_pipeline(self):
        if self._process and self._process.state() != QProcess.NotRunning:
            QMessageBox.warning(self, 'Already running',
                                'A pipeline run is already in progress.')
            return

        cfg = self._forms[self._current_key()].get_config()
        cfg['_observatory'] = self._current_key()
        self._run_path = self._forms[self._current_key()].data_path()

        fd, tmp_path = tempfile.mkstemp(suffix='.json', prefix='xray_uplim_')
        with os.fdopen(fd, 'w') as f:
            json.dump(cfg, f, indent=2)

        self._log.clear()
        self._tabs.setCurrentIndex(0)
        self._append_log(
            f'=== xray_uplim  |  {self._obs_combo.currentText()}  '
            f'|  RA={cfg.get("ra")}  Dec={cfg.get("dec")} ===\n\n',
            color='#4ec9b0')

        self._process = QProcess(self)
        self._process.setProcessChannelMode(QProcess.MergedChannels)
        self._process.readyReadStandardOutput.connect(self._on_stdout)
        self._process.finished.connect(
            lambda code, _: self._on_finished(code, tmp_path))

        self._run_btn.setEnabled(False)
        self._status.showMessage('Pipeline running…')
        self._process.start(sys.executable,
                            ['-m', 'xray_uplim._runner', '--config', tmp_path])

    def _on_stdout(self):
        data = self._process.readAllStandardOutput().data().decode(
            'utf-8', errors='replace')
        self._append_log(data)

    def _on_finished(self, exit_code: int, tmp_path: str):
        if exit_code == 0:
            self._append_log('\n✓ Pipeline completed successfully.\n',
                             color='#4ec9b0')
            self._status.showMessage('Done.')
            self._load_results()
        else:
            self._append_log(f'\n✗ Pipeline exited with code {exit_code}.\n',
                             color='#f44747')
            self._status.showMessage(f'Failed (exit code {exit_code}).')

        self._check_run_ready()
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    # ---- Results gallery ----------------------------------------------------

    def _load_results(self):
        """Show PDF plots from ul_products/ in the Results tab."""
        # Clear previous content
        while self._results_layout.count():
            item = self._results_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        # Find ul_products/ in both base_path/ and base_path/*/
        candidates = (
            glob.glob(os.path.join(self._run_path, 'ul_products')) +
            glob.glob(os.path.join(self._run_path, '*', 'ul_products'))
        )
        candidates = [d for d in candidates if os.path.isdir(d)]

        if not candidates:
            lbl = QLabel(
                f'No ul_products/ directory found under:\n  {self._run_path}')
            lbl.setStyleSheet('color:#aaa; padding:20px;')
            self._results_layout.addWidget(lbl)
            self._results_layout.addStretch()
            self._tabs.setCurrentIndex(1)
            return

        pdfs = sorted(
            p for d in candidates
            for p in glob.glob(os.path.join(d, '*.pdf'))
        )

        if not pdfs:
            lbl = QLabel('Pipeline ran but no plot files found in ul_products/.')
            lbl.setStyleSheet('color:#aaa; padding:20px;')
            self._results_layout.addWidget(lbl)
            self._results_layout.addStretch()
            self._tabs.setCurrentIndex(1)
            return

        if not _PDF_SUPPORT:
            lbl = QLabel(
                f'Found {len(pdfs)} plot(s) in ul_products/ but PDF rendering '
                f'requires PySide6.QtPdf (Qt ≥ 6.4).\n'
                f'Open the files directly from:\n  {candidates[0]}')
            lbl.setStyleSheet('color:#aaa; padding:20px;')
            lbl.setWordWrap(True)
            self._results_layout.addWidget(lbl)
            self._results_layout.addStretch()
            self._tabs.setCurrentIndex(1)
            return

        # Display each PDF (first page only — all diagnostic plots are single-page)
        right_width = self.width() - 460   # approximate available width
        img_width   = max(500, right_width - 40)

        for pdf_path in pdfs:
            # Filename label
            name_lbl = QLabel(f'<b style="color:#9cdcfe;">'
                               f'{os.path.basename(pdf_path)}</b>')
            name_lbl.setStyleSheet('background:#2b2b2b; padding:4px 8px;')
            self._results_layout.addWidget(name_lbl)

            # Render first page of PDF to QImage via QPdfDocument
            doc = QPdfDocument(self)
            doc.load(pdf_path)
            if doc.pageCount() > 0:
                page_size = doc.pagePointSize(0)   # QSizeF in points
                if page_size.width() > 0:
                    scale = img_width / page_size.width()
                    render_h = int(page_size.height() * scale)
                else:
                    render_h = img_width
                image = doc.render(0, QSize(img_width, render_h))
                pixmap = QPixmap.fromImage(image)
                if not pixmap.isNull():
                    img_lbl = QLabel()
                    img_lbl.setPixmap(pixmap)
                    img_lbl.setAlignment(Qt.AlignHCenter)
                    img_lbl.setStyleSheet('background:#2b2b2b; padding:4px;')
                    self._results_layout.addWidget(img_lbl)
            doc.close()

        self._results_layout.addStretch()
        self._tabs.setCurrentIndex(1)

    # ---- Log ----------------------------------------------------------------

    def _append_log(self, text: str, color: str = '#d4d4d4'):
        cursor = self._log.textCursor()
        cursor.movePosition(QTextCursor.End)
        fmt = cursor.charFormat()
        fmt.setForeground(QColor(color))
        cursor.setCharFormat(fmt)
        cursor.insertText(text)
        self._log.setTextCursor(cursor)
        self._log.ensureCursorVisible()


# =============================================================================

def launch():
    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName('xray_uplim')
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    launch()
