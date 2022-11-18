#!/usr/bin/env python
import polatrace as pl
import PySimpleGUI as sg
from matplotlib import pyplot as plt
import matplotlib.animation as animation
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, FigureCanvasAgg

import numpy as np
import time

def draw_figure(canvas, figure):
    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side='top', fill='both', expand=1)
    return figure_canvas_agg


def main():
    layout = [
        [ sg.Text('Polarization Ellipse', justification='center', size=(60, 1), relief=sg.RELIEF_SUNKEN),sg.Text('Poincare Sphere', justification='center', size=(60, 1), relief=sg.RELIEF_SUNKEN) ],
        [sg.Canvas(key='-CANVAS1-', size=(400, 480)),sg.Canvas(key='-CANVAS2-', size=(400, 480)),  sg.Checkbox('Normal Sphere', default=True, key='-Normal Sphere-',enable_events=True)],
        [sg.Canvas(key='-FORMULA-', size=(600,80))],
    # Polarization Input:
         [sg.T('SOP of Input:')],
         [sg.Text('S0:',size=(5,1), justification='right', font='ARIAL 12') , sg.Input(key='-s0-', default_text='1', size=(10,1),enable_events=True, font='ARIEL,12'),
          sg.Text('s1 (xS0):',size=(8,1), justification='right', font='ARIAL 12') , sg.Input(key='-s1-', default_text='0', size=(10,1),enable_events=True, font='ARIEL,12'),
          sg.Text('s2 (xS0):', size=(8, 1),justification='right', font='ARIAL 12'), sg.Input(key='-s2-',default_text='1', size=(10,1), enable_events=True, font='ARIEL,12'),
          sg.Text('s3 (xS0):', size=(8, 1),justification='right', font='ARIAL 12'), sg.Input(key='-s3-',default_text='0', size=(10,1), enable_events=True, font='ARIEL,12')],
        [sg.T('Jones Vector of Input:')],
        [sg.Text('ax:', size=(5, 1),justification='right', font='ARIAL 12'), sg.Input(key='-wave_ax-', default_text='1', size=(10, 1), enable_events=True, font='ARIEL,12'),
         sg.Text('ay:', size=(5, 1),justification='right', font='ARIAL 12'), sg.Input(key='-wave_ay-', default_text='1', size=(10, 1), enable_events=True, font='ARIEL,12'),
         sg.Text('\N{GREEK SMALL LETTER DELTA} (x \N{GREEK SMALL LETTER PI})',justification='right', size=(5, 1), font='ARIAL 14'),
         sg.Slider(size=(40, 10), range=(-1, 1), default_value=0, resolution=.01, orientation='h', enable_events=True,key='-wave_dphase-')],
        [sg.T('')],
    # Spun fiber setting
        [sg.Text('\N{GREEK SMALL LETTER DELTA}n (x 10^-6):', size=(15, 1), font='ARIAL 14'),
         sg.Input(key='-dn-', default_text='50', size=(10, 1), enable_events=True, font='ARIEL,12'),
         sg.Text('Rotate rate (\N{GREEK SMALL LETTER PI}/meter ):', size=(20, 1), justification='right', font='ARIAL 12'),
         sg.Input(key='-rr-', default_text='100', size=(15, 1), enable_events=True, font='ARIEL,12'),
         sg.Text('Length(meter):', size=(15, 1), justification='right', font='ARIAL 12'),
         sg.Input(key='-fiberlength-', default_text='0.5', size=(10, 1), enable_events=True, font='ARIEL,12')]]


    window = sg.Window('Spun fiber simulation', layout, finalize=True, resizable=True)
    canvas1_elem = window['-CANVAS1-']
    canvas1 = canvas1_elem.TKCanvas
    canvas2_elem = window['-CANVAS2-']
    canvas2 = canvas2_elem.TKCanvas


    fig_ellipse = plt.figure(figsize=(5, 5,), dpi=80)
    fig_poincare_sphere = plt.figure(figsize=(5, 5), dpi=80)
    fig_ellipse.subplots_adjust(left=0.2, bottom=0.2, right=0.8, top=0.8, wspace=1, hspace=1)
    fig_poincare_sphere.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=1, hspace=1)

    ax_ellipse = fig_ellipse.add_subplot(111)
    ax_pointcare_sphere = fig_poincare_sphere.add_subplot(projection='3d')
    jvinput = pl.jonescalculus(1, 0, 1, 0)
    jvinput.draw_ellipse(ax_ellipse, True, 'red')
    mmatrix=[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
    jvinput.draw_sphere(ax_pointcare_sphere,1,True,False,'blue', 'Input SOP')
    dnpm= 50
    rotaterate = 100*np.pi
    fiberlength = 0.5
    sf=pl.spunfiber(dnpm,rotaterate,fiberlength,1550, 1000,jvinput)
    jvinput.draw_sphere(ax_pointcare_sphere, 1, True, False, 'blue','Input SOP')
    sf.draw_trace_on_sphere(ax_pointcare_sphere, 'blue')


    fig_agg1 = draw_figure(canvas1, fig_ellipse)
    fig_agg2 = draw_figure(canvas2, fig_poincare_sphere)

    while True:
        event, values = window.read()
        if event == None:
            break

        if event == '-wave_ax-' or event == '-wave_ay-' or event == '-wave_dphase-':
            wave_ax = float(values['-wave_ax-'])
            wave_ay = float(values['-wave_ay-'])
            wave_dphase = float(values['-wave_dphase-']) * np.pi
            jvinput = pl.jonescalculus(wave_ax, 0, wave_ay, wave_dphase)
            SOP = jvinput.get_4x1_stokes()
            window.Element('-s0-').update("{:.3f}".format(SOP[0]))
            window.Element('-s1-').update("{:.3f}".format(SOP[1] / SOP[0]))
            window.Element('-s2-').update("{:.3f}".format(SOP[2] / SOP[0]))
            window.Element('-s3-').update("{:.3f}".format(SOP[3] / SOP[0]))

        if event == '-s0-' or event == '-s1-' or event == '-s2-' or event == '-s3-':
            SOP_s0 = float(values['-s0-'])
            SOP_s1 = float(values['-s1-'])
            SOP_s2 = float(values['-s2-'])
            SOP_s3 = float(values['-s3-'])

            SOP_pol = np.sqrt(SOP_s1 * SOP_s1 + SOP_s2 * SOP_s2 + SOP_s3 * SOP_s3)
            if SOP_pol == 0:
                SOP_pol = 1
            SOP_s0 = SOP_pol
            SOP_s1 = SOP_s1 / SOP_pol
            SOP_s2 = SOP_s2 / SOP_pol
            SOP_s3 = SOP_s3 / SOP_pol
            sop = pl.stokes(SOP_s0, SOP_s1, SOP_s2, SOP_s3)
            jvinput = sop.get_JonesVector_polarized_part()
            window.Element('-wave_ax-').update("{:.3f}".format(jvinput.x_am))
            window.Element('-wave_ay-').update("{:.3f}".format(jvinput.y_am))
            window.Element('-wave_dphase-').update("{:.3f}".format((jvinput.y_phase - jvinput.x_phase) / np.pi))


        jvinput.draw_ellipse(ax_ellipse, True, 'blue')
        dnpm= float(values['-dn-'])
        rotaterate = float(values['-rr-'])*np.pi
        fiberlength = float (values['-fiberlength-'])
        sf=pl.spunfiber(dnpm,rotaterate,fiberlength,1550, 1000,jvinput)
        jvinput.draw_sphere(ax_pointcare_sphere, 1, True, False, 'red','Input SOP')
        sf.draw_trace_on_sphere(ax_pointcare_sphere, 'blue')
        fig_agg1.draw()
        fig_agg2.draw()
        time.sleep(0.1)

    window.close()

if __name__ == '__main__':
    main()