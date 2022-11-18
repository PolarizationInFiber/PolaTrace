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
        [ sg.Text('PMD', justification='center', size=(80, 1), relief=sg.RELIEF_SUNKEN),sg.Text('Pointcare Sphere', justification='center', size=(70, 1), relief=sg.RELIEF_SUNKEN) ],
        [sg.Canvas(key='-CANVAS1-', size=(600, 480)),sg.Canvas(key='-CANVAS2-', size=(600, 480))],
    #    [sg.Canvas(key='-FORMULA-', size=(600,80))],
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
    # wavelength
         [sg.Text('Central Wavelength (nm):', size=(15, 1), font='ARIAL 14'),
          sg.Input(key='-wavelength-', default_text='1550', size=(10, 1), enable_events=True, font='ARIEL,12'),
          sg.Text('Start WL(nm):',  justification='right',size=(15, 1), font='ARIAL 14'),
          sg.Input(key='-startWl-', default_text='1548', size=(10, 1),enable_events=True, font='ARIEL,12',),
          sg.Text('End WL(nm):', justification='right',size=(15, 1), font='ARIAL 14'),
          sg.Input(key='-endWl-', default_text='1552', size=(10, 1), enable_events=True, font='ARIEL,12'),
          sg.Text('Points:', justification='right',size=(8, 1), font='ARIAL 12'),
          sg.Input(key='-points-', default_text='200', size=(10, 1), enable_events=True, font='ARIEL,12')
          ],
    # DGDs:
        [ sg.Text('DGDs(ps):', size=(10, 1), font='ARIAL 14'),
          sg.Input(key='-dgd1-', default_text='1', size=(8, 1), enable_events=True, font='ARIEL,12'),
          sg.Input(key='-dgd2-', default_text='1', size=(8, 1), enable_events=True, font='ARIEL,12'),
          sg.Input(key='-dgd3-', default_text='1', size=(8, 1), enable_events=True, font='ARIEL,12'),
          sg.Input(key='-dgd4-', default_text='1', size=(8, 1), enable_events=True, font='ARIEL,12'),
          sg.Input(key='-dgd5-', default_text='1', size=(8, 1), enable_events=True, font='ARIEL,12'),
          sg.Input(key='-dgd6-', default_text='1', size=(8, 1), enable_events=True, font='ARIEL,12')],
    # axis angles:
        [sg.Text('Axis (x \N{GREEK SMALL LETTER PI} ):', size=(10, 1), font='ARIAL 14'),
         sg.Input(key='-angle1-', default_text='0', size=(8, 1), enable_events=True, font='ARIEL,12'),
         sg.Input(key='-angle2-', default_text='0.25', size=(8, 1), enable_events=True, font='ARIEL,12'),
         sg.Input(key='-angle3-', default_text='0', size=(8, 1), enable_events=True, font='ARIEL,12'),
         sg.Input(key='-angle4-', default_text='0.25', size=(8, 1), enable_events=True, font='ARIEL,12'),
         sg.Input(key='-angle5-', default_text='0', size=(8, 1), enable_events=True, font='ARIEL,12'),
         sg.Input(key='-angle6-', default_text='0.25', size=(8, 1), enable_events=True, font='ARIEL,12'),
         ],
        [sg.Button('Update Display', key='-update-', size=(150, 1))]]

    window = sg.Window('Polarization Mode Dispersion Simulation', layout, finalize=True, resizable=True)
    canvas1_elem = window['-CANVAS1-']
    canvas1 = canvas1_elem.TKCanvas
    canvas2_elem = window['-CANVAS2-']
    canvas2 = canvas2_elem.TKCanvas
    fig_PMD, axs_PMD = plt.subplots(3, sharex=True, sharey=False)
    fig_poincare_sphere = plt.figure(figsize=(6.5, 6.5), dpi=80)
    fig_PMD.subplots_adjust(left=0.15, bottom=0.2, right=0.95, top=0.9, wspace=1, hspace=0.2)
    fig_poincare_sphere.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=1, hspace=1)
    dgd=np.zeros(6)
    angle=np.zeros(6)
    ax_pointcare_sphere = fig_poincare_sphere.add_subplot(projection='3d')
    jvinput = pl.jonescalculus(1, 0, 1, 0)
    PMD=pl.PMD([1,1,1,1,1,1],[0,np.pi/4,0,np.pi/4,0,np.pi/4],1550)
    start_wl=1548
    end_wl=1552
    points=100
    DGD,PSP,PDCD,Depolarization,wl,angularfreq = PMD.get_PMD_Spectrum(start_wl,end_wl,points)
    axs_PMD[0].plot(wl,np.multiply(DGD,1e12))
    axs_PMD[1].plot(wl,np.multiply(PDCD,1e24))
    axs_PMD[2].plot(wl, np.multiply(Depolarization,1e24))
    axs_PMD[1].set_ylabel('PDCD(ps^2)')
    axs_PMD[0].set_ylabel('DGD(ps)')
    axs_PMD[2].set_ylabel('Depol.(ps^2)')
    axs_PMD[2].set_xlabel('Wavelength(nm)')
    jvinput.draw_sphere(ax_pointcare_sphere,1,True,False,'blue', 'Input SOP')
    PMD.draw_output_SOP_on_sphere(ax_pointcare_sphere, start_wl, end_wl, points, jvinput, 'blue')
    fig_agg1 = draw_figure(canvas1, fig_PMD)
    fig_agg2 = draw_figure(canvas2, fig_poincare_sphere)
    updateflag=0
    while True:
        event, values = window.read()
        if event == None:
            break
      #  try:
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
            updateflag=1

        if event == '-s0-' or event == '-s1-' or event == '-s2-' or event == '-s3-':
                SOP_s0 = float(values['-s0-'])
                SOP_s1 = float(values['-s1-'])
                SOP_s2 = float(values['-s2-'])
                SOP_s3 = float(values['-s3-'])

                SOP_pol = np.sqrt(SOP_s1*SOP_s1+SOP_s2*SOP_s2+SOP_s3*SOP_s3)
                if SOP_pol==0:
                    SOP_pol=1
                SOP_s0=SOP_pol
                SOP_s1=SOP_s1/SOP_pol
                SOP_s2 = SOP_s2 / SOP_pol
                SOP_s3 = SOP_s3 / SOP_pol
                sop=pl.stokes(SOP_s0,SOP_s1,SOP_s2,SOP_s3)
                jvinput=sop.get_JonesVector_polarized_part()
                window.Element('-wave_ax-').update("{:.3f}".format(jvinput.x_am))
                window.Element('-wave_ay-').update("{:.3f}".format(jvinput.y_am))
                window.Element('-wave_dphase-').update("{:.3f}".format((jvinput.y_phase-jvinput.x_phase)/np.pi))

           #except:
           #    pass
        if event=='-update-' or updateflag==1:
            dgd[0]=float(values['-dgd1-'])
            dgd[1] = float(values['-dgd2-'])
            dgd[2] = float(values['-dgd3-'])
            dgd[3] = float(values['-dgd4-'])
            dgd[4] = float(values['-dgd5-'])
            dgd[5] = float(values['-dgd6-'])
            angle[0] = float(values['-angle1-'])*np.pi
            angle[1] = float(values['-angle2-'])*np.pi
            angle[2] = float(values['-angle3-'])*np.pi
            angle[3] = float(values['-angle4-'])*np.pi
            angle[4] = float(values['-angle5-'])*np.pi
            angle[5] = float(values['-angle6-'])*np.pi
            centerWl=float(values['-wavelength-'])
            PMD = pl.PMD(dgd, angle, centerWl)
            start_wl = float(values['-startWl-'])
            end_wl = float(values['-endWl-'])
            points = int(values['-points-'])
            DGD, PSP, PDCD, Depolarization, wl, angularfreq = PMD.get_PMD_Spectrum(start_wl, end_wl, points)
            axs_PMD[0].cla()
            axs_PMD[1].cla()
            axs_PMD[2].cla()
            axs_PMD[0].plot(wl, np.multiply(DGD,1e12))
            axs_PMD[1].plot(wl, np.multiply(PDCD,1e24))
            axs_PMD[2].plot(wl, np.multiply(Depolarization,1e24))
            axs_PMD[1].set_ylabel('PDCD(ps^2)')
            axs_PMD[0].set_ylabel('DGD(ps)')
            axs_PMD[2].set_ylabel('Depol.(ps^2)')
            axs_PMD[2].set_xlabel('Wavelength(nm)')
            ax_pointcare_sphere.cla()
            jvinput.draw_sphere(ax_pointcare_sphere, 1, True, False, 'blue', 'Input SOP')
            PMD.draw_output_SOP_on_sphere(ax_pointcare_sphere, start_wl, end_wl, points, jvinput, 'blue')

        #fig_agg1 = draw_figure(canvas1, fig_PMD)
        #fig_agg2 = draw_figure(canvas2, fig_poincare_sphere)
            fig_agg1.draw()
            fig_agg2.draw()
            updateflag=0

        time.sleep(0.1)

    window.close()


if __name__ == '__main__':
    main()