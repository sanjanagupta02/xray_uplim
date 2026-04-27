"""
nustar_uplim.region_selector
-----------------------------
Interactive matplotlib widget for selecting source and background regions.

Layout
------
  Left  (60%) : large event count map
  Right (38%) : info panel  →  mode toggle  →  sliders  →  buttons

Controls
--------
  Radio "Move: Source / Background" — switches which region a click repositions
  Click image                       — places the active region centre
  Sliders                           — source radius, bkg outer radius, inner factor
  Confirm                           — accept and return values to the pipeline
  Reset                             — restore original cfg values
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.widgets import Slider, Button, RadioButtons


def select_regions_interactive(evt_x, evt_y, cx_evt, cy_evt, pscale_evt,
                                cfg, label):
    """
    Open an interactive region-selection window and return confirmed values.

    Returns
    -------
    dict with keys: cx, cy, bkg_cx, bkg_cy,
                    src_radius_arcsec, bkg_radius_arcsec, bkg_inner_factor
    """
    # ---- Override DPI so the window fits on screen --------------------------
    _saved_dpi = matplotlib.rcParams.get('figure.dpi', 100)
    matplotlib.rcParams['figure.dpi'] = 96

    # ---- Mutable state ------------------------------------------------------
    state = {
        'cx':                float(cx_evt),
        'cy':                float(cy_evt),
        'bkg_cx':            float(cx_evt),
        'bkg_cy':            float(cy_evt),
        'src_radius_arcsec': float(cfg.src_radius_arcsec),
        'bkg_radius_arcsec': float(cfg.bkg_radius_arcsec),
        'bkg_inner_factor':  float(cfg.bkg_inner_factor),
        'confirmed':         False,
    }
    defaults = dict(state)

    # ---- Raw event image ----------------------------------------------------
    pad_arcsec = cfg.bkg_radius_arcsec * 1.5
    pad_pix    = pad_arcsec / pscale_evt
    x_lo, x_hi = cx_evt - pad_pix, cx_evt + pad_pix
    y_lo, y_hi = cy_evt - pad_pix, cy_evt + pad_pix

    img, _, _ = np.histogram2d(
        evt_x, evt_y,
        bins=[np.linspace(x_lo, x_hi, 301),
              np.linspace(y_lo, y_hi, 301)])
    img = img.T

    ext = [(x_lo - cx_evt) * pscale_evt, (x_hi - cx_evt) * pscale_evt,
           (y_lo - cy_evt) * pscale_evt, (y_hi - cy_evt) * pscale_evt]

    def _pix_to_as(px, py):
        return (px - cx_evt) * pscale_evt, (py - cy_evt) * pscale_evt

    # =========================================================================
    # Layout  (all in figure fraction, origin = bottom-left)
    #
    # Sliders are INSET from the right panel edges so the label (drawn just
    # LEFT of the axes) and value text (drawn just RIGHT of the axes) both
    # have space and don't clip against the figure boundary or the image.
    #   left=0.73 leaves 0.09 fig-width (~117px) for the label text
    #   right=0.94 leaves 0.04 fig-width (~52px) for the value text
    #
    #   Image  : left=0.03  bot=0.04  w=0.58  h=0.90
    #   Info   : left=0.64  bot=0.66  w=0.34  h=0.26
    #   Radio  : left=0.64  bot=0.52  w=0.34  h=0.12
    #   Sl src : left=0.73  bot=0.42  w=0.21  h=0.025
    #   Sl bkg : left=0.73  bot=0.34  w=0.21  h=0.025
    #   Sl fac : left=0.73  bot=0.26  w=0.21  h=0.025
    #   RstView: left=0.64  bot=0.21  w=0.32  h=0.04
    #   Confirm: left=0.64  bot=0.12  w=0.15  h=0.08
    #   Reset  : left=0.81  bot=0.12  w=0.15  h=0.08
    # =========================================================================
    fig = plt.figure(figsize=(13, 9))

    ax_img   = fig.add_axes([0.03, 0.04, 0.58, 0.90])
    ax_info  = fig.add_axes([0.64, 0.66, 0.34, 0.26])
    ax_radio = fig.add_axes([0.64, 0.52, 0.34, 0.12])
    ax_sl_s  = fig.add_axes([0.73, 0.42, 0.21, 0.025])
    ax_sl_b  = fig.add_axes([0.73, 0.34, 0.21, 0.025])
    ax_sl_f  = fig.add_axes([0.73, 0.26, 0.21, 0.025])
    ax_rstv  = fig.add_axes([0.64, 0.21, 0.32, 0.04])
    ax_ok    = fig.add_axes([0.64, 0.12, 0.15, 0.08])
    ax_rst   = fig.add_axes([0.81, 0.12, 0.15, 0.08])

    # ---- Image --------------------------------------------------------------
    vmax = float(np.percentile(img[img > 0], 99)) if img.any() else 1.0
    ax_img.imshow(img, origin='lower', extent=ext,
                  cmap='viridis', aspect='equal',
                  interpolation='nearest', vmin=0, vmax=max(1, vmax))
    ax_img.set_xlabel('$\\Delta X$ (arcsec)', fontsize=14)
    ax_img.set_ylabel('$\\Delta Y$ (arcsec)', fontsize=14)
    ax_img.tick_params(axis='both', labelsize=12)
    ax_img.set_title(f'{label} — click to place region  |  scroll wheel to zoom',
                     fontsize=14, pad=8)

    # Circles -----------------------------------------------------------------
    src_circ = Circle((0, 0), state['src_radius_arcsec'],
                       edgecolor='tomato', facecolor='none',
                       lw=2.0, ls='--', zorder=5, label='Source')
    out_circ = Circle((0, 0), state['bkg_radius_arcsec'],
                       edgecolor='orange', facecolor='none',
                       lw=1.8, ls='-',  zorder=5, label='Bkg outer')
    inn_circ = Circle((0, 0),
                       state['src_radius_arcsec'] * state['bkg_inner_factor'],
                       edgecolor='gold', facecolor='none',
                       lw=1.4, ls=':', zorder=5, label='Bkg inner')
    for c in (src_circ, out_circ, inn_circ):
        ax_img.add_patch(c)

    ch = state['src_radius_arcsec'] * 0.2
    cross_h, = ax_img.plot([-ch, ch], [0, 0], '-', color='tomato', lw=1.0, zorder=6)
    cross_v, = ax_img.plot([0, 0], [-ch, ch], '-', color='tomato', lw=1.0, zorder=6)
    bkg_ch_h, = ax_img.plot([0, 0], [0, 0], '-', color='orange', lw=1.0,
                             zorder=6, visible=False)
    bkg_ch_v, = ax_img.plot([0, 0], [0, 0], '-', color='orange', lw=1.0,
                             zorder=6, visible=False)
    ax_img.legend(loc='upper right', fontsize=11, framealpha=0.7)

    # Store original view limits for "Reset view" button.
    # Use the pre-computed ext directly — avoids calling get_xlim()/get_ylim()
    # before the figure is drawn, which can trigger a premature render on some backends.
    orig_xlim = [ext[0], ext[1]]
    orig_ylim = [ext[2], ext[3]]

    # ---- Info panel ---------------------------------------------------------
    ax_info.axis('off')
    ax_info.set_title('Current selection', fontsize=15, pad=4)
    info_txt = ax_info.text(0.5, 0.97, '', transform=ax_info.transAxes,
                             va='top', ha='center', fontsize=13,
                             fontfamily='monospace',
                             bbox=dict(boxstyle='round', fc='#f5f5f5', alpha=0.9))

    # ---- Mode radio buttons -------------------------------------------------
    radio = RadioButtons(ax_radio, ('Move:  Source', 'Move:  Background'),
                         activecolor='tomato')
    # Style the radio labels
    for lbl in radio.labels:
        lbl.set_fontsize(13)

    # ---- Sliders ------------------------------------------------------------
    sl_max_s = min(300.0, pad_arcsec * 0.55)
    sl_max_b = pad_arcsec * 0.88

    sl_src = Slider(ax_sl_s, 'Src radius (")',
                    5.0, sl_max_s,
                    valinit=state['src_radius_arcsec'], valstep=1.0,
                    color='tomato', initcolor='none')
    sl_bkg = Slider(ax_sl_b, 'Bkg outer (")',
                    20.0, sl_max_b,
                    valinit=state['bkg_radius_arcsec'], valstep=5.0,
                    color='orange', initcolor='none')
    sl_fac = Slider(ax_sl_f, 'Inner factor',
                    1.0, 3.0,
                    valinit=state['bkg_inner_factor'], valstep=0.05,
                    color='gold', initcolor='none')

    for sl in (sl_src, sl_bkg, sl_fac):
        sl.label.set_fontsize(13)
        sl.valtext.set_fontsize(13)

    # ---- Buttons ------------------------------------------------------------
    btn_rstv = Button(ax_rstv, 'Reset view', color='#e8e8e8', hovercolor='#d0d0d0')
    btn_ok   = Button(ax_ok,   'Confirm',    color='#d4edda', hovercolor='#c3e6cb')
    btn_rst  = Button(ax_rst,  'Reset',      color='#f8d7da', hovercolor='#f5c6cb')
    btn_rstv.label.set_fontsize(13)
    btn_ok.label.set_fontsize(14)
    btn_rst.label.set_fontsize(14)

    # =========================================================================
    # Helpers
    # =========================================================================

    def _separate():
        return (abs(state['bkg_cx'] - state['cx']) > 1.0 or
                abs(state['bkg_cy'] - state['cy']) > 1.0)

    def _update_info():
        sdx, sdy = _pix_to_as(state['cx'],     state['cy'])
        bdx, bdy = _pix_to_as(state['bkg_cx'], state['bkg_cy'])
        sep = _separate()
        r_in = state['src_radius_arcsec'] * state['bkg_inner_factor']
        lines = [
            f"  Src centre",
            f"    dX : {sdx:+7.1f} \"",
            f"    dY : {sdy:+7.1f} \"",
            f"  Src radius  : {state['src_radius_arcsec']:5.0f} \"",
            f"  Bkg mode    : {'manual' if sep else 'annulus'}",
        ]
        if sep:
            lines += [
                f"  Bkg centre",
                f"    dX : {bdx:+7.1f} \"",
                f"    dY : {bdy:+7.1f} \"",
            ]
        else:
            lines.append(f"  Bkg inner   : {r_in:5.0f} \"")
        lines.append(f"  Bkg outer   : {state['bkg_radius_arcsec']:5.0f} \"")
        info_txt.set_text('\n'.join(lines))

    def _redraw():
        sdx, sdy = _pix_to_as(state['cx'],     state['cy'])
        bdx, bdy = _pix_to_as(state['bkg_cx'], state['bkg_cy'])
        rs   = state['src_radius_arcsec']
        ro   = state['bkg_radius_arcsec']
        ri   = rs * state['bkg_inner_factor']
        ch   = rs * 0.2
        sep  = _separate()
        bkch = ro * 0.08

        src_circ.center = (sdx, sdy);  src_circ.radius = rs
        out_circ.center = (bdx, bdy);  out_circ.radius = ro
        inn_circ.center = (sdx, sdy);  inn_circ.radius = ri
        inn_circ.set_visible(not sep)

        cross_h.set_data([sdx - ch, sdx + ch], [sdy, sdy])
        cross_v.set_data([sdx, sdx],            [sdy - ch, sdy + ch])

        bkg_ch_h.set_data([bdx - bkch, bdx + bkch], [bdy, bdy])
        bkg_ch_v.set_data([bdx, bdx],                [bdy - bkch, bdy + bkch])
        bkg_ch_h.set_visible(sep)
        bkg_ch_v.set_visible(sep)

        _update_info()
        fig.canvas.draw_idle()

    _redraw()

    # =========================================================================
    # Callbacks
    # =========================================================================

    def on_click(event):
        if event.inaxes is not ax_img or event.xdata is None:
            return
        if 'Background' in radio.value_selected:
            state['bkg_cx'] = cx_evt + event.xdata / pscale_evt
            state['bkg_cy'] = cy_evt + event.ydata / pscale_evt
        else:
            state['cx'] = cx_evt + event.xdata / pscale_evt
            state['cy'] = cy_evt + event.ydata / pscale_evt
        _redraw()

    def on_src_slider(val):
        state['src_radius_arcsec'] = val
        if val * state['bkg_inner_factor'] >= state['bkg_radius_arcsec']:
            new_o = min(val * state['bkg_inner_factor'] * 1.5, sl_bkg.valmax)
            state['bkg_radius_arcsec'] = new_o
            sl_bkg.set_val(new_o)
        _redraw()

    def on_bkg_slider(val):
        state['bkg_radius_arcsec'] = val
        _redraw()

    def on_fac_slider(val):
        state['bkg_inner_factor'] = val
        _redraw()

    def on_scroll(event):
        """Zoom in/out centred on the cursor position."""
        if event.inaxes is not ax_img or event.xdata is None:
            return
        factor = 0.65 if event.button == 'up' else 1.55
        xc, yc = event.xdata, event.ydata
        xl, xr = ax_img.get_xlim()
        yb, yt = ax_img.get_ylim()
        ax_img.set_xlim([xc + (xl - xc) * factor, xc + (xr - xc) * factor])
        ax_img.set_ylim([yc + (yb - yc) * factor, yc + (yt - yc) * factor])
        fig.canvas.draw_idle()

    def on_reset_view(_):
        ax_img.set_xlim(orig_xlim)
        ax_img.set_ylim(orig_ylim)
        fig.canvas.draw_idle()

    def on_confirm(_):
        state['confirmed'] = True
        plt.close(fig)

    def on_reset(_):
        for k in ('cx', 'cy', 'bkg_cx', 'bkg_cy',
                  'src_radius_arcsec', 'bkg_radius_arcsec', 'bkg_inner_factor'):
            state[k] = defaults[k]
        sl_src.set_val(state['src_radius_arcsec'])
        sl_bkg.set_val(state['bkg_radius_arcsec'])
        sl_fac.set_val(state['bkg_inner_factor'])
        _redraw()

    fig.canvas.mpl_connect('button_press_event', on_click)
    fig.canvas.mpl_connect('scroll_event',       on_scroll)
    sl_src.on_changed(on_src_slider)
    sl_bkg.on_changed(on_bkg_slider)
    sl_fac.on_changed(on_fac_slider)
    btn_rstv.on_clicked(on_reset_view)
    btn_ok.on_clicked(on_confirm)
    btn_rst.on_clicked(on_reset)

    # ---- Block until window closes ------------------------------------------
    plt.show(block=False)
    plt.pause(0.1)
    while plt.fignum_exists(fig.number) and not state['confirmed']:
        plt.pause(0.05)

    matplotlib.rcParams['figure.dpi'] = _saved_dpi

    # ---- Return -------------------------------------------------------------
    if not state['confirmed']:
        print("  [GUI] Closed without Confirm — using original config values.")
        return {k: defaults[k] for k in
                ('cx', 'cy', 'bkg_cx', 'bkg_cy',
                 'src_radius_arcsec', 'bkg_radius_arcsec', 'bkg_inner_factor')}

    sdx, sdy = _pix_to_as(state['cx'], state['cy'])
    sep = _separate()
    print(f"  [GUI] Confirmed:  src={state['src_radius_arcsec']:.0f}\"  "
          f"bkg outer={state['bkg_radius_arcsec']:.0f}\"  "
          f"src offset dX={sdx:+.1f}\" dY={sdy:+.1f}\"  "
          f"bkg={'separate' if sep else 'annulus'}")

    return {k: state[k] for k in
            ('cx', 'cy', 'bkg_cx', 'bkg_cy',
             'src_radius_arcsec', 'bkg_radius_arcsec', 'bkg_inner_factor')}
