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

def draw_ellipse(ampx, phase_x, ampy,phase_y):
    ellipse = pl.jonescalculus(ampx, phase_x, ampy, phase_y).get_ellipse()
    plt.cla()  # clear the subplot
    plt.grid()  # draw the grid
    x, y = ellipse.get_ellipse_trace(0, 2 * np.pi, 200)  # get the curve of ellipse curve
    max=np.max([np.fabs(ampx),np.fabs(ampy)])
    plt.xlim(-max, max)
    plt.ylim(-max, max)
    graph,=plt.plot(x, y, color='purple')
    plt.xlabel ( "Ex")
    plt.ylabel ("Ey")
    return x,y,graph





def main():
    layout = [
        [sg.Text('Polarization Ellipse Demo', justification='center', size=(60, 1), relief=sg.RELIEF_SUNKEN)],
        [sg.Canvas(key='-CANVAS-', size=(480, 480))],
        [sg.Text('amplitude_x', font='COURIER 14'),
         sg.Input(key='ampx', size=(20, 1), default_text=1, enable_events=True),
         sg.Text('     phase_x = 0', font='COURIER 14')],
        [sg.Text('amplitude_y', font='COURIER 14'),
         sg.Input(key='ampy', default_text=1, size=(20, 1), enable_events=True)],
        [sg.Text('phase_y - phase_x ', font='COURIER 14'), sg.Text('(x  \N{GREEK SMALL LETTER PI})', font='14')],
        [sg.Slider(size=(50, 15), range=(-1, 1), default_value=0, resolution=.01, orientation='h', enable_events=False,
                   key='slider_phase')]]

    window = sg.Window('Polarization Ellipse Demo', layout, finalize=True, resizable=True)
    canvas_elem = window['-CANVAS-']
    slider_elem = window['slider_phase']
    canvas = canvas_elem.TKCanvas
    slider_elem.bind("<ButtonRelease-1>", "buttonrelease")

    fig = plt.figure(figsize=(6,6,),dpi=80)
    fig_agg = draw_figure(canvas, fig)
    x,y,graph = draw_ellipse(1,0,1,0)
    fig_agg.draw()



    while True:


        event, values = window.read()
        if event == None:
            break
        try:
            print('this is test')
            ampx = float(values['ampx'])
            ampy = float(values['ampy'])
            dphase = float(values['slider_phase']) * np.pi
            draw_ellipse(ampx,0,ampy,dphase)
            fig_agg.draw()
        except:
            pass

        time.sleep(0.1)

    window.close()


if __name__ == '__main__':
    main()
