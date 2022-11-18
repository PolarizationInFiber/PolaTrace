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

def display_formula(ax, jv_input, jv_output, jmatrix,mmatrix):

    ax.plot([0,1], [1,1], color='purple')
    ax.set_ylim(0,1)
 #   print(jv_input)
# jones ouput
    ax.cla()
    ax.text(0.05,0.8, r'$J_{output}$', fontsize=15)
    ax.text(0.20, 0.8, 'Jones Matrix', fontsize=15)
    ax.text(0.5, 0.8, r'$J_{input}$', fontsize=15)
    ax.text(0.65,0.85,'Muller Matrix', fontsize=15)

    ax.text(0.01,0.5, r'$J_x =  $' + "{:.3f}".format(jv_output[0]), fontsize=13)
    ax.text(0.01, 0.3, r'$J_x =  $' + "{:.3f}".format(jv_output[1]), fontsize=13)

    ax.text(0.18, 0.5, "("+"{:.3f}".format(jmatrix[0][0])+ '  ,  '+ "{:.3f}".format(jmatrix[0][1])+")", fontsize=13)
    ax.text(0.18, 0.3, "("+"{:.3f}".format(jmatrix[1][0]) + '  ,  ' + "{:.3f}".format(jmatrix[1][1])+")", fontsize=13)


    ax.text(0.48, 0.5, r'$J_x =  $' + "{:.3f}".format(jv_input[0]), fontsize=13)
    ax.text(0.48, 0.3, r'$J_y =  $' + "{:.3f}".format(jv_input[1]), fontsize=13)

    m1 = "("+"{: .3f}".format(mmatrix[0][0]) + "{: .3f}".format(mmatrix[0][1]) + "{: .3f}".format(
        mmatrix[0][2]) + "{: .3f}".format(mmatrix[0][3])+")"
    m2 ="("+ "{: .3f}".format(mmatrix[1][0]) + "{: .3f}".format(mmatrix[1][1]) + "{: .3f}".format(
        mmatrix[1][2]) + "{: .3f}".format(mmatrix[1][3])+")"
    m3 ="("+ "{: .3f}".format(mmatrix[2][0]) + "{: .3f}".format(mmatrix[2][1]) + "{: .3f}".format(
        mmatrix[2][2]) + "{: .3f}".format(mmatrix[2][3])+")"
    m4 ="("+ "{: .3f}".format(mmatrix[3][0]) + "{: .3f}".format(mmatrix[3][1]) + "{: .3f}".format(
        mmatrix[3][2]) + "{: .3f}".format(mmatrix[3][3])+")"

    ax.text(0.68, 0.7, m1, fontsize=15)
    ax.text(0.68, 0.5, m2, fontsize=15)
    ax.text(0.68, 0.3, m3, fontsize=15)
    ax.text(0.68, 0.1, m4, fontsize=15)


    #ax.text(0,0.6,  str_value, fontsize=20)
   # ax_result.text(0, 0.6, r'| 1,2,3,4|', fontsize=20)
   # ax_result.text(0, 0.4, r'| 1,2,3,4|', fontsize=20)
   # ax_result.text(0, 0.2, r'| 1,2,3,4|', fontsize=20)




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
    # waveplate setting:
        [sg.Radio('Waveplate', 'Optical Elements', key='-WP_checked-', size=(10, 1), default=True, enable_events=True, font='14'),
         sg.Text('\N{GREEK SMALL LETTER DELTA} (x \N{GREEK SMALL LETTER PI} ):', size=(6, 1), font='ARIAL 14'),
         sg.Slider(size=(35, 10), range=(0, 2), default_value=0, resolution=.01, orientation='h', enable_events=True, key='-WP_phase-'),
         sg.Text('Fast-Axis(deg):', size=(15, 1), font='14'),
         sg.Slider(size=(30, 10), range=(0, 180), default_value=0, resolution=.1, orientation='h', enable_events=True, key='-WP_angle-')],
    # rotator setting:
        [sg.Radio('Rotator', 'Optical Elements', key='-R_checked-', size=(10, 1), enable_events=True, font='14'),
         sg.Text('Rotation Angle (x\N{GREEK SMALL LETTER PI} ):', size=(15, 1), font='ARIAL 14'),
         sg.Slider(size=(50, 10), range=(0, 180), default_value=0, resolution=.01, orientation='h', enable_events=True, key='-R_angle-')],
    # polarizer setting:
        [sg.Radio('Polarizer', 'Optical Elements', key='-P_checked-', size=(10, 1), enable_events=True, font='14'),
         sg.Text('ER(dB):', size=(6, 1), font='ARIAL 14'),
         sg.Slider(size=(35, 10), range=(0, 60), default_value=0, resolution=.01, orientation='h', enable_events=True, key='-P_ER-'),
         sg.Text('Axis Angle(deg):', size=(20, 1), font='14'),
         sg.Slider(size=(25, 10), range=(0, 180), default_value=0, resolution=.01, orientation='h', enable_events=True, key='-P_angle-')]]

    window = sg.Window('Polarization Elements Demo', layout, finalize=True, resizable=True)
    canvas1_elem = window['-CANVAS1-']
    canvas1 = canvas1_elem.TKCanvas
    canvas2_elem = window['-CANVAS2-']
    canvas2 = canvas2_elem.TKCanvas
    canvas3_elem = window['-FORMULA-']
    canvas3 = canvas3_elem.TKCanvas
   # slider_elem = window['slider_phase']

   # slider_elem.bind("<ButtonRelease-1>", "buttonrelease")

    fig_ellipse = plt.figure(figsize=(5, 5,), dpi=80)
    fig_poincare_sphere = plt.figure(figsize=(5, 5), dpi=80)
    fig_results = plt.figure(figsize=(10.2,1.5),dpi=80)
    fig_ellipse.subplots_adjust(left=0.2, bottom=0.2, right=0.8, top=0.8, wspace=1, hspace=1)
    fig_poincare_sphere.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=1, hspace=1)
    fig_results.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=1, hspace=1)


    ax_ellipse = fig_ellipse.add_subplot(111)
    ax_pointcare_sphere = fig_poincare_sphere.add_subplot(projection='3d')
    jvinput = pl.jonescalculus(1, 0, 1, 0)
    jvinput.draw_ellipse(ax_ellipse, True, 'red')
    mmatrix=[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
    jvinput.draw_sphere(ax_pointcare_sphere,1,True,False,'blue','Input SOP')
    ax_result = fig_results.add_subplot(111)
    fig_agg1 = draw_figure(canvas1, fig_ellipse)
    fig_agg2 = draw_figure(canvas2, fig_poincare_sphere)
    fig_agg3 = draw_figure(canvas3,fig_results)



    while True:
        event, values = window.read()
        if event == None:
            break
      #  try:

        if event=='-wave_ax-' or event=='-wave_ay-' or event=='-wave_dphase-':
                 wave_ax = float(values['-wave_ax-'])
                 wave_ay = float(values['-wave_ay-'])
                 wave_dphase = float(values['-wave_dphase-'])*np.pi
                 jvinput=pl.jonescalculus(wave_ax,0,wave_ay,wave_dphase)
                 SOP=jvinput.get_4x1_stokes()

                 window.Element('-s0-').update("{:.3f}".format(SOP[0]))
                 window.Element('-s1-').update("{:.3f}".format(SOP[1]/SOP[0]))
                 window.Element('-s2-').update("{:.3f}".format(SOP[2]/SOP[0]))
                 window.Element('-s3-').update("{:.3f}".format(SOP[3]/SOP[0]))

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

        if values['-WP_checked-'] == True:
             wp_phase = float(values['-WP_phase-']) * np.pi
             wp_angle = float(values['-WP_angle-']) / 180.0 * np.pi
             wp = pl.waveplate(wp_phase, wp_angle)

             jv_output=np.matmul(wp.jmatrix,jvinput.jvector)
             jvoutput=pl.jvtoexp(jv_output)
             jmatrix=wp.jmatrix
             mmatrix = wp.mmatrix
             jvinput.draw_ellipse(ax_ellipse,True,'blue')
             jvinput.draw_sphere(ax_pointcare_sphere,1,True,False,'blue','Input SOP')
             jvoutput.draw_ellipse(ax_ellipse, False, 'purple')
             jvoutput.draw_sphere(ax_pointcare_sphere,1,False,False,'purple','Output SOP')
             wp.draw_trace_on_sphere(ax_pointcare_sphere,jvinput,'red','blue')


        if values['-R_checked-'] == True:
             r_angle = float(values['-R_angle-'])/180*np.pi
             r=pl.rotator(r_angle)
             jv_output=np.matmul(r.jmatrix,jvinput.jvector)
             jvoutput=pl.jvtoexp(jv_output)
             jmatrix = r.jmatrix
             mmatrix = r.mmatrix
             jvinput.draw_ellipse(ax_ellipse,True,'blue')
             jvinput.draw_sphere(ax_pointcare_sphere,1,True,False,'blue')
             jvoutput.draw_ellipse(ax_ellipse, False, 'purple')
             jvoutput.draw_sphere(ax_pointcare_sphere,1,False,False,'purple')
             r.draw_trace_on_sphere(ax_pointcare_sphere,jvinput,'red','blue')

        if values['-P_checked-'] == True:
            p_ER= float(values['-P_ER-'])
            p_angle = float(values['-P_angle-']) / 180.0 * np.pi
            p=pl.polarizer(p_ER,p_angle)
            jv_output = np.matmul(p.jmatrix, jvinput.jvector)
            jvoutput = pl.jvtoexp(jv_output)
            mmatrix = p.mmatrix
            jmatrix=p.jmatrix
            jvinput.draw_ellipse(ax_ellipse, True, 'blue')
            jvinput.draw_sphere(ax_pointcare_sphere,1,True, False, 'blue')
            jvoutput.draw_ellipse(ax_ellipse, False, 'purple')
            if values['-Normal Sphere-']==True:
                ratio=1
            else:
                ratio=jvoutput.intensity/jvinput.intensity
            jvoutput.draw_sphere(ax_pointcare_sphere,ratio,False, False, 'purple')
            normalsphere=values['-Normal Sphere-']
            p.draw_trace_on_sphere(ax_pointcare_sphere, jvinput, 'red',normalsphere)


            '''
            ampx = float(values['ampx'])
            ampy = float(values['ampy'])
            dphase = float(values['slider_phase']) * np.pi
            draw_ellipse(ax1, ampx, 0, ampy, dphase)
            draw_sphere(ax2,ampx, 0, ampy, dphase)
            fig_agg1.draw()
            fig_agg2.draw()
            '''
           #except:
           #    pass
        display_formula(ax_result,jvinput.jvector,jvoutput.jvector,jmatrix,mmatrix)
        fig_agg1.draw()
        fig_agg2.draw()
        fig_agg3.draw()
        time.sleep(0.1)

    window.close()


if __name__ == '__main__':
    main()
