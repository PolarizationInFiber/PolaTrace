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

def draw_ellipse(ax,ampx, phase_x, ampy, phase_y):
    pointsoncurve=200
    jv=pl.jonescalculus(ampx, phase_x, ampy, phase_y)
    ellipse = jv.get_ellipse()
    ax.cla()  # clear the subplot
    ax.grid()  # draw the grid
    x, y = ellipse.get_ellipse_trace(0, 2 * np.pi, pointsoncurve)  # get the curve of ellipse curve
    max = np.max([np.fabs(ampx), np.fabs(ampy)])
    ax.set_xlim([-max,max])
    ax.set_ylim([-max,max])

    ax.plot(x, y, color='purple')

    ax.plot([ellipse.semi_major_a*np.cos(ellipse.azimuth),-ellipse.semi_major_a*np.cos(ellipse.azimuth)],
            [ellipse.semi_major_a*np.sin(ellipse.azimuth),-ellipse.semi_major_a*np.sin(ellipse.azimuth)],
            color='gray',linewidth=1)

    ax.plot([ellipse.semi_minor_b * np.cos(ellipse.azimuth+np.pi/2), -ellipse.semi_minor_b * np.cos(ellipse.azimuth+np.pi/2)],
            [ellipse.semi_minor_b* np.sin(ellipse.azimuth+np.pi/2), -ellipse.semi_minor_b * np.sin(ellipse.azimuth+np.pi/2)],
            color='gray', linewidth=1)
    ax.arrow(x[50], y[50], x[50]-x[51],y[50]-y[51],width=max*0.02,head_width=max*0.05, color='red')
    ax.spines['left'].set_position('center')
    ax.spines['bottom'].set_position('center')
    ax.text(max+max*0.05, 0, r'$E_x=A_x$', fontsize=15)
    ax.text(-max*0.3,max*1.15,r'$E_y=A_y*cos(\omega t+(\delta_y-\delta_x))$',fontsize=15)
    azimuthanglestr=r'$\psi=$'f'{ellipse.azimuth*180/np.pi:.2f}'r'$^o$'
    ax.text(ellipse.semi_major_a*np.cos(ellipse.azimuth),ellipse.semi_major_a*np.sin(ellipse.azimuth), azimuthanglestr,fontsize=15)
    if np.fabs(jv.get_3x1_stokes()[2])>0.001:
        if ellipse.sense =='RH':
          ax.text(-max*0.4, -max*1.3, 'Right-hand Polarization',fontsize=15)
        else:
          ax.text(-max*0.4, -max * 1.3, 'Left-hand Polarization',fontsize=15)
    else:
        ax.text(-max*0.4, -max * 1.3, 'Linear Polarization',fontsize=15)

def draw_sphere(ax,ampx, phase_x, ampy, phase_y):
    jv=pl.jonescalculus(ampx, phase_x, ampy, phase_y)
    stokes = jv.get_3x1_stokes()
    azimuth = jv.get_ellipse().azimuth
    elliplicity = jv.get_ellipse().ellipticity
    print(elliplicity)
    print(stokes)
    ax.cla()  # clear the subplot

    ax.set_xlabel('S1')
    ax.set_ylabel('S2')
    ax.set_zlabel('S3')
    ax.set_box_aspect([1, 1, 0.9])
    # draw sphere
    u = np.linspace(0, 2 * np.pi, 20)
    v = np.linspace(0, np.pi, 20)
    two_psi = np.linspace(0,2*azimuth,20)
    two_chi = np.linspace(0,2*np.arctan(elliplicity),20)

    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))

    # step 1 Plot the Poincare sphere
    ax.plot_surface(x, y, z,color="w", edgecolor="cornflowerblue", alpha=0.1, linewidth=0.1)
    ax.axis('off')   # hide axes
    ax.grid(False)   # hide grid
    ax.plot([-1,1], [0,0],[0,0], color="gray",linewidth=0.5)   # draw s1-axis
    ax.quiver(0, 0, 0, 1 ,0,0, length=1, color="gray", arrow_length_ratio=0.1,linewidth=0.5) # draw s1  arrow
    ax.text(1.2, 0, 0, s="S1")
    ax.plot([0, 0], [-1, 1], [0, 0], color="gray", linewidth=0.5)  # draw s2-axis
    ax.quiver(0, 0, 0, 0, 1, 0, length=1, color="gray", arrow_length_ratio=0.1,linewidth=0.5) # draw s2-arrow
    ax.text(0, 1.2, 0, s="S2")
    ax.plot([0, 0], [0, 0], [-1, 1], color="gray",linewidth=0.5)  # draw s3-axis
    ax.quiver(0, 0, 0, 0, 0, 1, length=1, color="gray", arrow_length_ratio=0.1,linewidth=0.5) # draw s3-arrow
    ax.text(0, 0, 1.2, s="S3")

   # step 2 draw auxiliary line
    s1=np.full([20],stokes[0])                    # draw the moving circle when phase varies
    r=np.sqrt(1-np.multiply(s1,s1))
    ax.plot(s1,np.multiply(r,np.cos(u)),np.multiply(r, np.sin(u)), color='gray', linewidth=2, linestyle ='dotted')
    #draw arc of two_psi and two_chi

    ax.plot(np.cos(two_psi),np.sin(two_psi), np.multiply(0,two_psi), color='blue', linewidth=1, linestyle = 'dashed') #draw arc of two-psi
    two_psi_str = r'$2\psi=$'f'{2*azimuth * 180 / np.pi:.2f}'r'$^o$'
    ax.text(np.cos(azimuth)*1.1,np.sin(azimuth)*1.1,0.0, two_psi_str, fontsize=15)
    ax.plot(np.multiply(np.cos(2*azimuth),np.cos(two_chi)), np.multiply(np.sin(2*azimuth),np.cos(two_chi)), np.sin(two_chi), color='blue', linewidth=1, linestyle='dashed')
    two_chi_str = r'$2\chi=$'f'{2 * np.arctan(elliplicity) * 180 / np.pi:.2f}'r'$^o$'
    ax.text(np.cos(2*azimuth) * np.cos(2*np.arctan(elliplicity))*1.1, np.sin(2*azimuth) * np.cos(2*np.arctan(elliplicity))*1.1, np.sin(np.arctan(elliplicity)), two_chi_str, fontsize=15)

   # step 3 draw Stokes point
    ax.plot([0,stokes[0]], [0,stokes[1]], [0,stokes[2]], color="red",linewidth=2,linestyle = '-.')

   # step4 draw projection lines
    ax.plot([0,np.cos(2*jv.get_ellipse().azimuth)],[0,np.sin(2*jv.get_ellipse().azimuth)],[0,0],color="red",linewidth=1, linestyle='dotted')
    ax.plot([stokes[0],stokes[0]], [stokes[1],stokes[1]], [0,stokes[2]], color='blue', linewidth=1, linestyle='dotted')
    sopstr="sop=[ " + f'{stokes[0]:.3f}' + ", " + f'{stokes[1]:.3f}' + ", " + f'{stokes[2]:.3f}' + "]"
    ax.scatter(stokes[0], stokes[1], stokes[2],  marker ="o", color ="red",label=sopstr)
    ax.legend(fontsize=15, loc='lower center')



#    ax.set_aspect('auto')
#    ax.set_box_aspect(np.ptp(limits, axis=1))



def main():
    layout = [
        [sg.Text('Polarization Ellipse', justification='center', size=(80, 1), relief=sg.RELIEF_SUNKEN),sg.Text('Poincare Sphere', justification='center', size=(80, 1), relief=sg.RELIEF_SUNKEN) ],
        [sg.Canvas(key='-CANVAS1-', size=(640, 640)),sg.Canvas(key='-CANVAS2-', size=(640, 640))],
        [sg.Text('amplitude_x', font='COURIER 14'),
         sg.Input(key='ampx', size=(20, 1), default_text=1, enable_events=True),
         sg.Text('     phase_x = 0', font='COURIER 14')],
        [sg.Text('amplitude_y', font='COURIER 14'),
         sg.Input(key='ampy', default_text=1, size=(20, 1), enable_events=True)],
        [sg.Text('phase_y - phase_x ', font='COURIER 14'), sg.Text('(x  \N{GREEK SMALL LETTER PI}):', font='14')],
        [sg.Slider(size=(50, 15), range=(-1, 1), default_value=0, resolution=.01, orientation='h', enable_events=True,
                   key='slider_phase')]]
    window = sg.Window('Polarization Ellipse Demo', layout, finalize=True, resizable=True)
    canvas1_elem = window['-CANVAS1-']
    canvas1 = canvas1_elem.TKCanvas
    canvas2_elem = window['-CANVAS2-']
    canvas2 = canvas2_elem.TKCanvas

    slider_elem = window['slider_phase']

   # slider_elem.bind("<ButtonRelease-1>", "buttonrelease")

    fig1 = plt.figure(figsize=(8, 8,), dpi=80)
    fig2 = plt.figure(figsize=(8, 8), dpi=80)
    fig1.subplots_adjust(left=0.15, bottom=0.15, right=0.85, top=0.85, wspace=1, hspace=1)
    fig2.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=1, hspace=1)

    ax1 = fig1.add_subplot(111)
    ax2 = fig2.add_subplot(projection='3d')

    fig_agg1 = draw_figure(canvas1, fig1)
    fig_agg2 = draw_figure(canvas2, fig2)

    draw_ellipse(ax1, 1, 0, 1, 0)
    draw_sphere(ax2,1,0,1,0)
    fig_agg1.draw()
    fig_agg2.draw()
    while True:
        event, values = window.read()
        if event == None:
            break
        try:
            ampx = float(values['ampx'])
            ampy = float(values['ampy'])
            dphase = float(values['slider_phase']) * np.pi
            draw_ellipse(ax1, ampx, 0, ampy, dphase)
            draw_sphere(ax2,ampx, 0, ampy, dphase)
            fig_agg1.draw()
            fig_agg2.draw()
        except:
            pass

        time.sleep(0.1)

    window.close()


if __name__ == '__main__':
    main()
