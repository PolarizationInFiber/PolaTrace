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


def draw_ellipse(ax,fig,ampx, phase_x, ampy, phase_y):
    pointsoncurve=200
    jv=pl.jonescalculus(ampx, phase_x, ampy, phase_y)
    ellipse = jv.get_ellipse()
    ax.cla()  # clear the subplot
    ax.grid()  # draw the grid
    x, y = ellipse.get_ellipse_trace(0, 2 * np.pi, pointsoncurve)  # get the curve of ellipse curve
    max = np.max([np.fabs(ampx), np.fabs(ampy)])

    plt.xlim(-max, max)
    plt.ylim(-max, max)

    ax.plot(x, y, color='purple')
    ax.plot([ellipse.semi_major_a*np.cos(ellipse.azimuth),-ellipse.semi_major_a*np.cos(ellipse.azimuth)],
            [ellipse.semi_major_a*np.sin(ellipse.azimuth),-ellipse.semi_major_a*np.sin(ellipse.azimuth)],
            color='gray',linewidth=1)

    ax.plot([ellipse.semi_minor_b * np.cos(ellipse.azimuth+np.pi/2), -ellipse.semi_minor_b * np.cos(ellipse.azimuth+np.pi/2)],
            [ellipse.semi_minor_b* np.sin(ellipse.azimuth+np.pi/2), -ellipse.semi_minor_b * np.sin(ellipse.azimuth+np.pi/2)],
            color='gray', linewidth=1)

    plt.arrow(x[50], y[50], x[50]-x[51],y[50]-y[51],width=max*0.02,head_width=max*0.05, color='red')
    ax.spines['left'].set_position('center')
    ax.spines['bottom'].set_position('center')
    plt.text(max+max*0.05, 0, r'$E_x=A_x$', fontsize=15)
    plt.text(-max*0.3,max*1.15,r'$E_y=A_y*cos(\omega t+(\delta_y-\delta_x))$',fontsize=15)
    azimuthanglestr=r'$\psi=$'f'{ellipse.azimuth*180/np.pi:.2f}'r'$^o$'
    plt.text(ellipse.semi_major_a*np.cos(ellipse.azimuth),ellipse.semi_major_a*np.sin(ellipse.azimuth), azimuthanglestr,fontsize=15)
    if np.fabs(jv.get_3x1_stokes()[2])>0.001:
        if ellipse.sense =='RH':
          plt.text(-max*0.4, -max*1.3, 'Right-hand Polarization',fontsize=15)
        else:
          plt.text(-max*0.4, -max * 1.3, 'Left-hand Polarization',fontsize=15)
    else:
        plt.text(-max*0.4, -max * 1.3, 'Linear Polarization',fontsize=15)

def main():
    layout = [
        [sg.Text('Polarization Ellipse Demo', justification='center', size=(80, 1), relief=sg.RELIEF_SUNKEN)],
        [sg.Canvas(key='-CANVAS-', size=(640, 640))],
        [sg.Text('amplitude_x', font='COURIER 14'),
         sg.Input(key='ampx', size=(20, 1), default_text=1, enable_events=True),
         sg.Text('     phase_x = 0', font='COURIER 14')],
        [sg.Text('amplitude_y', font='COURIER 14'),
         sg.Input(key='ampy', default_text=1, size=(20, 1), enable_events=True)],
        [sg.Text('phase_y - phase_x ', font='COURIER 14'), sg.Text('(x  \N{GREEK SMALL LETTER PI}):', font='14')],
        [sg.Slider(size=(50, 15), range=(-1, 1), default_value=0, resolution=.01, orientation='h', enable_events=True,
                   key='slider_phase')]]

    window = sg.Window('Polarization Ellipse Demo', layout, finalize=True, resizable=True)
    canvas_elem = window['-CANVAS-']
    slider_elem = window['slider_phase']
    canvas = canvas_elem.TKCanvas
   # slider_elem.bind("<ButtonRelease-1>", "buttonrelease")

    fig = plt.figure(figsize=(8, 8,), dpi=80)
    plt.subplots_adjust(left=0.15, bottom=0.15, right=0.85, top=0.85, wspace=1, hspace=1)
    ax = fig.add_subplot(111)
    fig_agg = draw_figure(canvas, fig)
    draw_ellipse(ax, fig, 1, 0, 1, 0)
    fig_agg.draw()

    while True:
        event, values = window.read()
        if event == None:
            break
        try:
            ampx = float(values['ampx'])
            ampy = float(values['ampy'])
            dphase = float(values['slider_phase']) * np.pi
            draw_ellipse(ax, fig, ampx, 0, ampy, dphase)
            fig_agg.draw()
        except:
            pass

        time.sleep(0.1)

    window.close()


if __name__ == '__main__':
    main()
