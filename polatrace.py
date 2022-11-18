import numpy as np
#import matplotlib as plt
from matplotlib import pyplot as plt
from math import *
import cmath

# by Xiaojun (James) Chen in Inline Photonics Inc.   email: jchen@inlinephotonics.com
#

# common functions
def deg2rad(degree):
    # a function to convert an angle in degree to radian
    radian = np.multiply(degree, pi / 180.0)
    return radian


def rad2deg(radian):
    # a function to convert an angle in radian to degree
    degree = np.multipy(radian, 180.0 / pi)
    return degree


def cnum(amplitude, phase):
    # To define a complex number by its amplitude and phase
    if np.ndim(amplitude) == np.ndim(np.zeros(1)):
        cnumber = np.zeros(len(amplitude)).astype(complex)
        for index in range(len(amplitude)):
            cnumber[index] = complex(np.multiply(amplitude[index], np.cos(phase[index])),
                                     np.multiply(amplitude[index], np.sin(phase[index])))

    else:
        cnumber = complex(np.multiply(amplitude, np.cos(phase)), np.multiply(amplitude, np.sin(phase)))

    return cnumber

def normal_vector(vector):
    # returns the unit vector along the input vector
    magnitude = np.linalg.norm(vector)
    if (magnitude != 0):
        normal = np.divide(np.array(vector), magnitude)
    else:
        normal = vector
    return normal


def cart2sph(x, y, z):
    rho = np.sqrt(np.multiply(x, x) + np.multiply(y, y))
    r = np.sqrt(np.multiply(rho, rho) + np.multiply(z, z))
    elevation = np.arctan2(z, rho)  # theta
    azimuth = np.arctan2(x, y)
    return r, azimuth, elevation

def jm2mm(j2by2):  #
    a = np.array([[1, 0, 0, 1], [1, 0, 0, -1], [0, 1, 1, 0], [0, 1j, -1j, 0]])
    inva = np.array([[1, 1, 0, 0], [0, 0, 1,  - 1j], [0,0, 1, 1j], [1, -1, 0, 0]])
    inva = np.divide(inva, 2)
    dim = np.ndim(j2by2)
    jshape = np.shape(j2by2)
    if dim == 2:
         j = np.kron(j2by2, np.conjugate(j2by2))
         m = np.real(np.matmul(np.matmul(a, j), inva))
    else:
        j=np.zeros((jshape[0], 4, 4)).astype(complex)
        m=np.zeros((jshape[0], 4, 4))
        for index in range (jshape[0]):
             j[index] = np.kron(j2by2[index], j2by2[index])
             m[index] = np.real(np.matmul(np.matmul(a, j[index]), inva))
    return m

def jvtoexp(jonesvector):
    x_phase = np.angle(jonesvector[0])
    x_amp = np.sqrt(np.real(np.multiply(jonesvector[0],np.conjugate(jonesvector[0]))))
    y_phase = np.angle(jonesvector[1])
    y_amp = np.sqrt(np.real(np.multiply(jonesvector[1],np.conjugate(jonesvector[1]))))
    jv=jonescalculus(x_amp, x_phase, y_amp, y_phase)
    return jv
#
class monplaneEMW:
    def __init__(self, angular_freq, k0_unit_vector, cE1, cE2, n1, n2):
        # cE1 and cE2 two orthogonal components of instantaneous electric field at time=0 and the origin O.
        self.angular_freq = angular_freq
        self.k0_unit_vector = k0_unit_vector
        self.cE1 = cE1
        self.cE1 = cE2
        self.n1 = n1
        self.n2 = n2


class ellispe:
    def __init__(self, semi_major_a, semi_minor_b, azimuth, sense):
        # the x-axis is horizontal and the y-axis is vertical.
        ndim = np.ndim(semi_major_a)
        if ndim == 0:
            semi_major_a = np.array([semi_major_a,1])
            semi_minor_b = np.array([semi_minor_b, 1])
            azimuth = np.array([azimuth,0])
            sense = np.array([sense,"RH"])

        self.semi_major_a = semi_major_a
        self.semi_minor_b = semi_minor_b
        self.azimuth = azimuth
        self.sense = sense
        self.ellipticity = np.divide(semi_minor_b,semi_major_a)

        for index in range (len(semi_major_a)):
            if semi_major_a[index] > semi_minor_b[index]:
              self.semi_major_a[index] = semi_major_a[index]  # semi-major axis   # a>=0
              self.semi_minor_b[index] = semi_minor_b[index]  # semi-minor axis   # b>=0
            else:
              self.semi_major_a[index] = semi_minor_b[index]  # semi-major axis   # a>=0
              self.semi_minor_b[index] = semi_major_a[index]  # semi-minor axis   # b>=0

             # the angle of major axis from x-axis  # [0,pi)
            if sense[index] == "CW" or sense[index] == "RH":
                 self.sense[index] = 'RH'

            elif sense[index] == "CCW" or sense[index] == "LH":
                self.sense[index] = "LH"
                self.ellipticity[index] = -1*self.ellipticity[index]
            else:
                self.sense[index] = "RH"
        self.azimuth = azimuth
        if ndim == 0:
            self.semi_major_a = self.semi_major_a[0]
            self.semi_minor_b = self.semi_minor_b[0]
            self.azimuth = self.azimuth[0]
            self.sense = self.sense[0]
            self.ellipticity=self.ellipticity[0]

    def get_ellipse_trace(self, angle1, angle2, points):

        if 1 >= points:
            points = 2
        if self.sense == 'LH' or self.sense == 'CCW':
            start = max(angle1, angle2)
            stop = min(angle1, angle2)
        else:
            start = min(angle1, angle2)
            stop = max(angle1, angle2)

        step = (stop - start) / points
        angle = np.arange(start, stop+1e-10, step)  # the angle in polar coordinate
        temp_x = np.array(self.semi_major_a * np.cos(angle))  # temp_x when azimuth angle=0
        temp_y = np.array(self.semi_minor_b * np.sin(angle))  # temp_y when azimuth angle=0
        x = np.multiply(temp_x, cos(self.azimuth)) - np.multiply(temp_y, sin(self.azimuth))  # rotate temp_x and temp_y counterclockwise to the azimuth angle
        y = np.multiply(temp_x, sin(self.azimuth)) + np.multiply(temp_y, cos(self.azimuth))  # rotate temp_x and temp_y counterclockwise to the azimuth angle
        return x, y

    def get_stokes(self):  # get the 4 Stokes parameters of the corresponding elliptical polarization
        temp_s0 = self.semi_major_a ** 2 + self.semi_minor_b ** 2
        temp_s1 = self.semi_major_a ** 2 - self.semi_minor_b ** 2
        temp_s2 = 0
        if self.sense == 'RH':
            temp_s3 = 2 * self.semi_major_a * self.semi_minor_b
        else:
            temp_s3 = -2 * self.semi_major_a * self.semi_minor_b
        s0 = temp_s0
        s1 = temp_s1 * cos(2 * self.azimuth)
        s2 = temp_s1 * sin(2 * self.azimuth)
        s3 = temp_s3
        return s0, s1, s2, s3

    def get_jonesvector(self):
        temp_ex = complex(self.semi_major_a, 0)
        if self.sense == "RH":
            phase = -pi / 2
        else:
            phase = pi / 2
        temp_ey = cnum(self.semi_minor_b, phase)
        ex = temp_ex * cos(self.azimuth) - temp_ey * sin(self.azimuth)
        ey = temp_ex * sin(self.azimuth) + temp_ey * cos(self.azimuth)
        return ex, ey

    def get_normjonesvector(self):
        temp_ex = complex(self.semi_major_a, 0)
        if self.sense == "RH":
            phase = -pi / 2
        else:
            phase = pi / 2
        temp_ey = cnum(self.semi_minor_b, phase)
        ex = temp_ex * cos(self.azimuth) - temp_ey * sin(self.azimuth)
        ey = temp_ex * sin(self.azimuth) + temp_ey * cos(self.azimuth)
        I = sqrt(self.semi_major_a ** 2 + self.semi_minor_b ** 2)
        if I == 0:
            I = 1
        return ex / I, ey / I

    def get_reducedstokes(self):  # get the 3 reduced Stokes parameters of the corresponding elliptical polarization
        s = self.get_stokes()
        if s[0] == 0:
            s[0] = 1
        rs1 = s[1] / s[0]
        rs2 = s[2] / s[0]
        rs3 = s[3] / s[0]
        return rs1, rs2, rs3

class jonescalculus:
    def __init__(self, x_am, x_phase, y_am, y_phase):  # without parameter time
        self.x_am = x_am
        self.x_phase = x_phase
        self.y_am = y_am
        self.y_phase = y_phase
        self.jx = cnum(x_am, x_phase)
        self.jy = cnum(y_am, y_phase)
        self.intensity = np.real(np.multiply(self.jx, np.conjugate(self.jx))+np.multiply(self.jy, np.conjugate(self.jy)))
        self.jvector = np.matrix.transpose(np.array([self.jx,self.jy]))
        self.njvector = np.divide(self.jvector, np.sqrt(self.intensity))
        self.ex = cnum(self.x_am, self.x_phase)
        self.ey = cnum(self.y_am, self.x_phase)

    def get_cmatrix(self):
        if np.ndim(self.jx) == np.ndim(np.zeros(1)):
          cm=np.zeros((len(self.jx),2,2)).astype(complex)
          cm[:, 0, 0] = self.jx * np.conjugate(self.jx)
          cm[:, 0, 1] = self.jx * np.conjugate(self.jy)
          cm[:, 1, 0] = self.jy * np.conjugate(self.jx)
          cm[:, 1, 1] = self.jy * np.conjugate(self.jy)

        else:
            cm = np.zeros((2, 2)).astype(complex)
            cm[0, 0] = self.jx * np.conjugate(self.jx)
            cm[0, 1] = self.jx * np.conjugate(self.jy)
            cm[1, 0] = self.jy * np.conjugate(self.jx)
            cm[1, 1] = self.jy * np.conjugate(self.jy)

        return cm

    def get_4x1_stokes(self):
        cm = self.get_cmatrix()
        if np.ndim(cm) == 3:
            sop = np.zeros((len(cm),4))
            sop[:,0] = np.real((cm[:,0,0] + cm[:,1,1]))
            sop[:,1] = np.real(cm[:,0,0] - cm[:,1,1])
            sop[:,2] = np.real(cm[:,0,1] + cm[:,1,0])
            sop[:,3] = np.real(np.multiply((cm[:,0,1] - cm[:,1,0]), complex(0, 1)))

        else:
            sop=np.zeros(4)
            sop[0]= np.real(cm[0,0] + cm[1,1])
            sop[1] = np.real(cm[0,0] - cm[1,1])
            sop[2] = np.real(cm[0,1] + cm[1,0])
            sop[3] = np.real(np.multiply((cm[0,1] - cm[1,0]), complex(0, 1)))

        return sop

    def get_3x1_stokes(self):  # 3x1 array
        s = self.get_4x1_stokes()
        if np.ndim(s)==2:
            nsop = np.zeros( (len(s),3))
            nsop[:,0] =np.divide(s[:,1],s[:,0])
            nsop[:, 1] = np.divide(s[:, 2], s[:, 0])
            nsop[:, 2] = np.divide(s[:, 3], s[:, 0])
        else:
            nsop = np.zeros(3)
            nsop[0] = np.divide(s[1], s[0])
            nsop[1] = np.divide(s[2], s[0])
            nsop[2] = np.divide(s[3], s[0])
        return nsop

    def get_ellipse(self):
        s = self.get_4x1_stokes()
        if np.ndim(s)!=2:
            stokes=np.array([s, [1,0,0,1]])
        else:
            stokes = np.copy(s)

        azimuth =np.zeros(len(stokes))
        sense=np.full((len(stokes)),"aa")
        for item in range (len(sense)):

            if stokes[item][3] >= 0:
               sense[item] = "RH"
            else:
               sense[item] = "LH"

        azimuth = np.divide(np.arctan2(stokes[:,2], stokes[:,1]), 2)
        ellipticity = np.tan(np.multiply(np.arcsin(np.divide(stokes[:,3], stokes[:,0])),0.5))
        a1 =np.sqrt(np.divide(stokes[:,0], 1 + np.multiply(ellipticity, ellipticity)))
        a2 = np.sqrt(np.fabs(np.subtract(stokes[:,0],np.multiply(a1,a1))))
        semi_major =np.copy(a1)
        semi_minor = np.copy(a2)
        for item in range(len(a1)):
           if a1[item] < a2[item]:
                semi_major[item] = a2[item]
                semi_minor[item] = a1[item]

        if np.ndim(s) == 1:
            semi_major = semi_major[0]
            semi_minor = semi_minor[0]
            azimuth = azimuth[0]
            sense = sense[0]
        return ellispe(semi_major, semi_minor, azimuth, sense)

    def draw_ellipse(self, ax, clear, colorofellipse):
        pointsoncurve = 200

        ellipse = self.get_ellipse()
        if clear==True:
            ax.cla()  # clear the subplot
            ax.grid()  # draw the grid

        x, y = ellipse.get_ellipse_trace(0, 2 * np.pi, pointsoncurve)  # get the curve of ellipse curve
        max = sqrt(self.x_am*self.x_am+self.y_am*self.y_am)

        if clear == True:
            ax.set_xlim([-max, max])
            ax.set_ylim([-max, max])

        ax.plot(x, y, color=colorofellipse)

        ax.plot([ellipse.semi_major_a * np.cos(ellipse.azimuth), -ellipse.semi_major_a * np.cos(ellipse.azimuth)],
                [ellipse.semi_major_a * np.sin(ellipse.azimuth), -ellipse.semi_major_a * np.sin(ellipse.azimuth)],
                color='gray', linewidth=1)
        ax.plot([ellipse.semi_minor_b * np.cos(ellipse.azimuth + np.pi / 2),
                 -ellipse.semi_minor_b * np.cos(ellipse.azimuth + np.pi / 2)],
                [ellipse.semi_minor_b * np.sin(ellipse.azimuth + np.pi / 2),
                 -ellipse.semi_minor_b * np.sin(ellipse.azimuth + np.pi / 2)],
                color='gray', linewidth=1)
        ax.arrow(x[50], y[50], x[50] - x[51], y[50] - y[51], width=max * 0.02, head_width=max * 0.05, color='red')
        ax.spines['left'].set_position('center')
        ax.spines['bottom'].set_position('center')
        if clear==True:
            ax.text(max + max * 0.05, 0, r'$E_x=A_x$', fontsize=15,color=colorofellipse)
            ax.text(-max * 0.3, max * 1.15, r'$E_y=A_y*cos(\omega t+(\delta_y-\delta_x))$', fontsize=15,color=colorofellipse)
        azimuthanglestr = r'$\psi=$'f'{ellipse.azimuth * 180 / np.pi:.2f}'r'$^o$'
        ax.text(ellipse.semi_major_a * np.cos(ellipse.azimuth), ellipse.semi_major_a * np.sin(ellipse.azimuth),
                azimuthanglestr, fontsize=12,color=colorofellipse)
        polnameposition=1.5
        if clear==True:
            polnameposition=1.3

        if np.fabs(self.get_3x1_stokes()[2]) > 0.001:

            if ellipse.sense == 'RH':
                ax.text(-max * 0.5, -max * polnameposition, 'Right-hand Polarization', fontsize=15, color=colorofellipse)
            else:
                ax.text(-max * 0.5, -max * polnameposition, 'Left-hand Polarization',fontsize=15, color=colorofellipse)
        else:
            ax.text(-max * 0.5, -max * polnameposition, 'Linear Polarization', fontsize=15,color=colorofellipse)

    def draw_sphere(self,ax,S0,clear,auxiliarycircle, colorofpoint,legendText):
        stokes =np.multiply( self.get_3x1_stokes(),S0)
        azimuth = self.get_ellipse().azimuth
        elliplicity = self.get_ellipse().ellipticity
        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, np.pi, 20)
        two_psi = np.linspace(0, 2 * azimuth, 20)
        two_chi = np.linspace(0, 2 * np.arctan(elliplicity), 20)

        if clear==True:
                ax.cla()  # clear the subplot
                ax.set_xlabel('S1')
                ax.set_ylabel('S2')
                ax.set_zlabel('S3')
            # draw sphere
                x = np.outer(np.cos(u), np.sin(v))
                y = np.outer(np.sin(u), np.sin(v))
                z = np.outer(np.ones(np.size(u)), np.cos(v))
           # step 1 Plot the Poincare sphere
                ax.plot_surface(x, y, z,color="w", edgecolor="cornflowerblue", alpha=0.1, linewidth=0.1)
                ax.plot([-1,1], [0,0],[0,0], color="gray",linewidth=0.5)   # draw s1-axis
                ax.quiver(0, 0, 0, 1 ,0,0, length=1, color="gray", arrow_length_ratio=0.1,linewidth=0.5) # draw s1  arrow
                ax.text(1.2, 0, 0, s="S1")
                ax.plot([0, 0], [-1, 1], [0, 0], color="gray", linewidth=0.5)  # draw s2-axis
                ax.quiver(0, 0, 0, 0, 1, 0, length=1, color="gray", arrow_length_ratio=0.1,linewidth=0.5) # draw s2-arrow
                ax.text(0, 1.2, 0, s="S2")
                ax.plot([0, 0], [0, 0], [-1, 1], color="gray",linewidth=0.5)  # draw s3-axis
                ax.quiver(0, 0, 0, 0, 0, 1, length=1, color="gray", arrow_length_ratio=0.1,linewidth=0.5) # draw s3-arrow
                ax.text(0, 0, 1.2, s="S3")

        ax.set_box_aspect([1, 1, 0.9])
        ax.axis('off')  # hide axes
        ax.grid(False)  # hide grid

        if auxiliarycircle==True:
         # step 2 draw auxiliary line
            s1 = np.full([20], stokes[0])  # draw the moving circle when phase varies
            r = np.sqrt(1 - np.multiply(s1, s1))
            ax.plot(s1, np.multiply(r, np.cos(u)), np.multiply(r, np.sin(u)), color='gray', linewidth=2,
                    linestyle='dotted')

         # draw arc of two_psi and two_chi
            ax.plot(np.cos(two_psi),np.sin(two_psi), np.multiply(0,two_psi), color='blue', linewidth=1, linestyle = 'dashed') #draw arc of two-psi
            two_psi_str = r'$2\psi=$'f'{2*azimuth * 180 / np.pi:.2f}'r'$^o$'
            ax.text(np.cos(azimuth)*1.1,np.sin(azimuth)*1.1,0.0, two_psi_str, fontsize=15)
            ax.plot(np.multiply(np.cos(2*azimuth),np.cos(two_chi)), np.multiply(np.sin(2*azimuth),np.cos(two_chi)), np.sin(two_chi), color='blue', linewidth=1, linestyle='dashed')
            two_chi_str = r'$2\chi=$'f'{2 * np.arctan(elliplicity) * 180 / np.pi:.2f}'r'$^o$'
            ax.text(np.cos(2*azimuth) * np.cos(2*np.arctan(elliplicity))*1.1, np.sin(2*azimuth) * np.cos(2*np.arctan(elliplicity))*1.1, np.sin(np.arctan(elliplicity)), two_chi_str, fontsize=15)

        # step 3 draw Stokes point
        ax.plot([0,stokes[0]], [0,stokes[1]], [0,stokes[2]], color=colorofpoint,linewidth=2,linestyle = '-.')

        # step4 draw projection lines
        ax.plot([0,np.cos(2*self.get_ellipse().azimuth)],[0,np.sin(2*self.get_ellipse().azimuth)],[0,0],color=colorofpoint,linewidth=1, linestyle='dotted')
        ax.plot([stokes[0],stokes[0]], [stokes[1],stokes[1]], [0,stokes[2]], color=colorofpoint, linewidth=1, linestyle='dotted')
        sopstr=legendText+"=[ " + f'{stokes[0]:.3f}' + ", " + f'{stokes[1]:.3f}' + ", " + f'{stokes[2]:.3f}' + "]"
        ax.scatter(stokes[0], stokes[1], stokes[2],  marker ="o", color =colorofpoint,label=sopstr)
        ax.legend(fontsize=15, loc='lower center')


class stokes:
    def __init__(self, s0, s1, s2, s3):  # without parameter time
        self.s0 = s0
        self.s1 = s1
        self.s2 = s2
        self.s3 = s3
        self.pol = np.sqrt(np.multiply(s1,s1)+np.multiply(s2,s2)+np.multiply(s3,s3))
        if self.s0 ==0 :
            self.s0 = 0.000001
        self.DOP =np.divide (self.pol,self.s0)

        if np.size(s0) != 1:
            self.sop = np.zeros((len(s0,4)))
            self.nsop = np.zeros((len(s0,3)))
            self.DOP=np.zeros(len(s0))
            for index in range(len(s0)):
                self.sop[index][0] = s0[index]
                self.sop[index][1] =  s1[index]
                self.sop[index][2] = s2[index]
                self.sop[index][3] = s3[index]
                pol=sqrt(s1[index]*s2[index]+s2[index]*s2[index]+s3[index]*s3[index])
                if pol==0:
                    pol=1
                self.nsop[index][0] = s1[index] / pol
                self.nsop[index][1] = s2[index] / pol
                self.nsop[index][2] = s3[index] / pol
                if self.sop[index][0]==0:
                   self.DOP[index]=0
                else:
                    self.DOP[index]=self.pol/self.sop[index][0]
        else:

            pol = sqrt(self.s1*self.s1+self.s2*self.s2+self.s3*self.s3)
            if pol == 0:
                pol = 1
            self.nsop = s1 / pol
            self.nsop = s2 / pol
            self.nsop = s3 / pol
            if self.s0==0:
                self.DOP=0
            else:
               self.DOP=pol/self.s0

    def get_JonesVector_polarized_part(self):
        ampx =np.sqrt(np.divide(np.add (self.pol,self.s1),2))
        ampy = np.sqrt(np.multiply(ampx,ampx)-np.multiply(self.s1, self.s1))
        phase= np.arctan2(self.s3,self.s2)
        return jonescalculus(ampx,0,ampy,phase)

class rotator:

    def __init__(self, angle):  # positive ccW
        self.angle = angle
        self.jmatrix = self.get_r_jonesmatrix()
        self.invjmatrix = self.get_inv_jonesmatrix()
        self.mmatrix = self.get_r_mullermatrix()
        self.invmmatrix = self.get_inv_mullermatrix()

    def get_r_jonesmatrix(self):
        jm00 = np.cos(self.angle)
        jm01 = -np.sin(self.angle)
        jm10 = -jm01
        jm11 = jm00
        if type(jm00) == type(np.zeros(1)):
            jm = np.zeros((len(jm00), 2, 2))
            for index in range(len(jm00)):
                jm[index][:][:] = [jm00[index], jm01[index]], [jm10[index], jm11[index]]
        else:
            jm = [[jm00, jm01], [jm10, jm11]]
        return jm

    def get_r_mullermatrix(self):
        m11 = np.cos(np.multiply(self.angle, 2))
        m12 = -np.sin(np.multiply(self.angle, 2))
        m21 = -m12
        m22 = m11
        m01 = np.abs(np.multiply(m11, 0))
        m02 = m03 = m10 = m20 = m30 = m13 = m23 = m31 = m32 = m01
        m00 = m33 = np.add(m01, 1)
        if type(m11) == type(np.zeros(1)):
            mm = np.zeros((len(m11), 4, 4))
            for index in range(len(m11)):
                mm[index] = [[m00[index], m01[index], m02[index], m03[index]],
                             [m10[index], m11[index], m12[index], m13[index]],
                             [m20[index], m21[index], m22[index], m23[index]],
                             [m30[index], m31[index], m32[index], m33[index]]]
        else:
            mm = [[m00, m01, m02, m03], [m10, m11, m12, m13], [m20, m21, m22, m23], [m30, m31, m32, m33]]
        return mm

    def get_inv_jonesmatrix(self):
        jm00 = np.cos(np.multiply(self.angle, -1))
        jm01 = -np.sin(np.multiply(self.angle, -1))
        jm10 = -jm01
        jm11 = jm00
        if type(jm00) == type(np.zeros(1)):
            jm = np.zeros((len(jm00), 2, 2))
            for index in range(len(jm00)):
                jm[index][:][:] = [jm00[index], jm01[index]], [jm10[index], jm11[index]]
        else:
            jm = [[jm00, jm01], [jm10, jm11]]
        return jm

    def get_inv_mullermatrix(self):
        m11 = np.cos(np.multiply(self.angle, -1))
        m12 = -np.sin(np.multiply(self.angle, -1))
        m21 = -m12
        m22 = m11
        m01 = np.abs(np.multiply(m11, 0))
        m02 = m03 = m10 = m20 = m30 = m13 = m23 = m31 = m32 = m01
        m00 = m33 = np.add(m01, 1)
        if type(m11) == type(np.zeros(1)):
            mm = np.zeros((len(m11), 4, 4))
            for index in range(len(m11)):
                mm[index] = [[m00[index], m01[index], m02[index], m03[index]],
                             [m10[index], m11[index], m12[index], m13[index]],
                             [m20[index], m21[index], m22[index], m23[index]],
                             [m30[index], m31[index], m32[index], m33[index]]]
        else:
            mm = [[m00, m01, m02, m03], [m10, m11, m12, m13], [m20, m21, m22, m23], [m30, m31, m32, m33]]
        return mm

    def draw_trace_on_sphere(self, ax, jvinput, colorofphase, colorofangle):
        u = np.linspace(0, np.pi, 60)
        sop = np.zeros((3, 60))
        for i in range(60):
            jv_output_phase = np.matmul(rotator(u[i]).jmatrix, jvinput.jvector)
            jvoutputphase = jvtoexp(jv_output_phase)
            stokes = jvoutputphase.get_3x1_stokes()
            sop[0][i] = stokes[0]
            sop[1][i] = stokes[1]
            sop[2][i] = stokes[2]
        ax.plot(sop[0, :], sop[1, :], sop[2, :], color=colorofphase, linewidth=1, linestyle='dashed')


class waveplate:
    def __init__(self, dphase, azimuth):  # dphase=phase_y-phase_x
        self.dphase = dphase
        self.azimuth = azimuth
        self.jmatrix = self.get_jonesmatrix()
        self.mmatrix = self.get_mmatrix()

    def get_jonesmatrix(self):
        jm00 = cnum(np.add(np.multiply(self.dphase, 0), 1), np.divide(self.dphase, 2.0))
        jm01 = np.multiply(jm00, 0)
        jm10 = np.multiply(jm00, 0)
        jm11 = cnum(np.add(np.multiply(self.dphase, 0), 1), np.divide(self.dphase, -2.0))
        if type(jm00) == type(np.zeros(1)):
            jm = np.zeros((len(self.dphase), 2, 2)).astype(complex)
            for index in range(len(self.dphase)):
                jm[index] = [[jm00[index], jm01[index]], [jm10[index], jm11[index]]]
        else:
            jm = np.array([[jm00, jm01], [jm10, jm11]])
        r = rotator(self.azimuth)
        jm = np.matmul(r.jmatrix, np.matmul(jm, r.invjmatrix))
        return jm

    def get_mmatrix(self):
        mmatrix = jm2mm(self.jmatrix)
        return mmatrix

    def draw_trace_on_sphere(self,ax,jvinput, colorofphase,colorofangle):
        u = np.linspace(0, 2 * np.pi, 60)
        v = np.linspace(0, np.pi, 60)
        phase=np.zeros((3,60))
        angle=np.zeros((3,60))

        for i in range (60):
            matrix_phase=waveplate(u[i],self.azimuth).jmatrix
            matrix_angle=waveplate(self.dphase,v[i]).jmatrix

            jv_output_phase = np.matmul(matrix_phase,jvinput.jvector)
            jvoutputphase=jvtoexp(jv_output_phase)
            stokes= jvoutputphase.get_3x1_stokes()
            phase[0][i]=stokes[0]
            phase[1][i]=stokes[1]
            phase[2][i]=stokes[2]

            jv_output_angle = np.matmul(matrix_angle, jvinput.jvector)
            jvoutputangle = jvtoexp(jv_output_angle)
            stokes = jvoutputangle.get_3x1_stokes()
            angle[0][i] = stokes[0]
            angle[1][i] = stokes[1]
            angle[2][i] = stokes[2]
        ax.plot(phase[0,:],phase[1,:],phase [2,:],color=colorofphase,linewidth=1,linestyle = 'dashed')
        ax.plot(angle[0,:],angle[1,:],angle [2,:],color=colorofangle,linewidth=1,linestyle = 'dashed')

class polarizer:
    def __init__(self,ER, azimuth):  # dphase=phase_y-phase_x
        self.ER = np.fabs(ER)
        self.azimuth = azimuth
        self.p1 = np.add(np.multiply(ER,0),1)
        self.p2 = np.sqrt(np.divide(1, np.power(10,np.divide(ER,10))))
        self.jmatrix=self.get_jonesmatrix()
        self.mmatrix=self.get_mmatrix()


    def get_jonesmatrix(self):
        jm00 =self.p1
        jm01 =0.0
        jm10 =0.0
        jm11= self.p2
        if type(jm00) == type(np.zeros(1)):
            jm = np.zeros((len(jm00), 2, 2))
            for index in range(len(jm00)):
               jm[index][:][:] = [jm00[index], jm01[index]], [jm10[index], jm11[index]]
        else:
            jm = [[jm00, jm01], [jm10, jm11]]
        r = rotator(self.azimuth)
        jm = np.matmul(r.jmatrix, np.matmul(jm, r.invjmatrix))

        return jm

    def get_mmatrix(self):
        mmatrix = jm2mm(self.jmatrix)
        return mmatrix

    def draw_trace_on_sphere(self, ax, jvinput, colorofangle,normalsphere):
        u = np.linspace(0, np.pi, 60)
        er_array= np.linspace(0,100,60)
        sop = np.zeros((3, 60))
        sop2 = np.zeros((3, 60))
        intensity =np.full(60,1.0)
        for i in range(60):
            jv_output = np.matmul(polarizer(self.ER,u[i]).jmatrix, jvinput.jvector)
            jvoutput = jvtoexp(jv_output)
            stokes = jvoutput.get_4x1_stokes()

            if(normalsphere==True):
                ratio=1
            else:
                ratio=jvoutput.intensity/jvinput.intensity
            sop[0][i] = stokes[1] / stokes[0]*ratio
            sop[1][i] = stokes[2] / stokes[0]*ratio
            sop[2][i] = stokes[3] / stokes[0]*ratio
        ax.plot(sop[0, :], sop[1, :], sop[2, :], color=colorofangle, linewidth=1, linestyle='dashed')

        for i in range(60):
            jv_output = np.matmul(polarizer(er_array[i], self.azimuth).jmatrix, jvinput.jvector)
            jvoutput = jvtoexp(jv_output)
            stokes = jvoutput.get_4x1_stokes()
            if (normalsphere == True):
                ratio = 1
            else:
                ratio = jvoutput.intensity / jvinput.intensity

            sop2[0][i] = stokes[1] / stokes[0]*ratio
            sop2[1][i] = stokes[2] / stokes[0]*ratio
            sop2[2][i] = stokes[3] / stokes[0]*ratio
        ax.plot(sop2[0, :], sop2[1, :], sop2[2, :], color='blue', linewidth=1, linestyle='dashed')

class spunfiber:
    def __init__(self, dn,rotate_rate,fiberlength,wavelengthinnm, segmentNumber, jvinput):  # dphase=phase_y-phase_x
        self.dn = dn
        self.rotate_rate = rotate_rate
        self.fiberlength = fiberlength
        self.jvinput=jvinput
        self.segmentNumber=segmentNumber
        self.sop=self.get_SOP_distribution()

    def get_SOP_distribution(self):
        dphase=np.zeros(self.segmentNumber)*np.pi
        angle=np.zeros((self.segmentNumber))
        s1=np.zeros(self.segmentNumber)
        s2 = np.zeros(self.segmentNumber)
        s3 = np.zeros(self.segmentNumber)
        jv=jvtoexp(self.jvinput.jvector)


        for i in range(self.segmentNumber):
           dphase[i]=self.dn*self.fiberlength/(self.segmentNumber-1)/1550*1e3*2*np.pi

           angle[i] =self.rotate_rate*self.fiberlength/(self.segmentNumber-1) *i
         #  print('dphase[', i, ']=', dphase[i], angle[i])
           wp=waveplate(dphase[i], angle[i])
           jv=jvtoexp(np.matmul(wp.jmatrix, jv.jvector))
           SOP=jv.get_3x1_stokes()
           s1[i] = SOP[0]
           s2[i] = SOP[1]
           s3[i] = SOP[2]
        return s1,s2,s3

    def draw_trace_on_sphere(self,ax, coloroftrace):
         s1,s2,s3=self.get_SOP_distribution()

         ax.plot(s1,s2,s3,color=coloroftrace,linewidth=1,linestyle = 'dashed')

class PMD:    # assuming phase velocity=group velocity
    def __init__(self, DGD_array_ps,angle_array,wavelength_nm):  # dphase=phase_y-phase_x
        self.DGD_array = DGD_array_ps
        self.angle_array = angle_array
        self.wavelength = wavelength_nm
        self.DGD,self.PSP, self.PDCD, self.depolarization=self.get_PMD(wavelength_nm)

    def get_jonesmatrix(self, angularfreq):
        phase = np.multiply(np.multiply(self.DGD_array, angularfreq),1e-12)
        len = np.size(phase)
        jonesmatrix= [[1 + 0j, 0], [0, 1 + 0j]]
        for i in range(len):
            wp = waveplate(phase[i], self.angle_array[i])
            jonesmatrix= np.matmul(wp.jmatrix, jonesmatrix)

        return jonesmatrix

    def get_PMD(self, wavelength):
        MaxDGD = np.sum(self.DGD_array)
        dw = MaxDGD/4*0.1 *2*np.pi/wavelength/ wavelength*2.99792458e17
        angularfreq = 2*np.pi/wavelength*2.99792458e17
        matrix_w = self.get_jonesmatrix(angularfreq)
        matrix_wsubdw = self.get_jonesmatrix(angularfreq-dw)
        matrix_wplusdw = self.get_jonesmatrix(angularfreq+dw)
        Tmatrix01 = np.matmul(matrix_w, np.linalg.inv(matrix_wsubdw))
        Tmatrix12 = np.matmul(matrix_wplusdw, np.linalg.inv(matrix_w))
        Tmatrix02 = np.matmul(matrix_wplusdw, np.linalg.inv(matrix_wsubdw))
        eig1,vector1=np.linalg.eig(Tmatrix01)
        eig2,vector2 = np.linalg.eig(Tmatrix12)
        eig, vector = np.linalg.eig(Tmatrix02)
        if(np.angle(eig1[0])>np.angle(eig1[1])):  #slow axis
            psp1=vector1[0]
        else:
            psp1=vector1[1]

        if (np.angle(eig2[0]) > np.angle(eig2[1])):
            psp2 = vector2[0]
        else:
            psp2 = vector2[1]

        if (np.angle(eig[0]) > np.angle(eig[1])):
            psp = vector[0]
        else:
            psp = vector[1]

        DGD1 = fabs((np.angle(eig1[0]) - np.angle(eig1[1])) / dw)
        DGD2 = fabs((np.angle(eig2[0]) - np.angle(eig2[1])) / dw)
        DGD = fabs((np.angle(eig[0]) - np.angle(eig[1])) /2/dw)
        dpsp=np.subtract(jvtoexp(psp2).get_3x1_stokes(),jvtoexp(psp1).get_3x1_stokes())
        pdcd=np.fabs(DGD1-DGD2)/dw
        depolarization=DGD*np.linalg.norm(dpsp)/dw
        return DGD,psp,pdcd,depolarization

    def get_PMD_Spectrum(self,start_wl,end_wl,points):
        start_angularfreq = 2 * np.pi / start_wl * 2.99792458e17
        end_angularfreq = 2 * np.pi / end_wl * 2.99792458e17
        angularfreq=np.linspace(start_angularfreq,end_angularfreq,points)
        wl = np.divide (2*np.pi* 2.99792458e17,angularfreq)
        DGD = np.zeros(points)
        PDCD = np.zeros(points)
        Depolarization = np.zeros(points)
        PSP =np.full([points,2],[1+0j])
        for i in range(points):
            DGD[i],PSP[i], PDCD[i], Depolarization[i] = self.get_PMD(wl[i])
        return DGD,PSP,PDCD,Depolarization,wl,angularfreq

    def draw_output_SOP_on_sphere(self, ax, start_wl,end_wl, points, jvinput,  coloroftrace):
        start_angularfreq = 2 * np.pi / start_wl * 2.99792458e17
        end_angularfreq = 2 * np.pi / end_wl * 2.99792458e17
        angularfreq = np.linspace(start_angularfreq, end_angularfreq, points)
        s1 = np.zeros(points)
        s2 = np.zeros(points)
        s3 = np.zeros(points)
        s3 = np.zeros(points)
        sop1psp=np.zeros(points)
        sop2psp = np.zeros(points)
        sop3psp = np.zeros(points)
        DGD, PSP, PDCD, Depolarization, wl, angularfreq=self.get_PMD_Spectrum(start_wl,end_wl, points)

        for i in range(points):
            spsp=jvtoexp(PSP[i]).get_3x1_stokes()
            sop1psp[i]=spsp[0]
            sop2psp[i] = spsp[1]
            sop3psp[i] = spsp[2]
            jvouput = np.matmul(self.get_jonesmatrix(angularfreq[i]), jvinput.jvector)
            nsop = jvtoexp(jvouput).get_3x1_stokes()
            s1[i]=nsop[0]
            s2[i]=nsop[1]
            s3[i]=nsop[2]
        ax.scatter(s1[0], s2[0], s3[0],  marker ="o", color=coloroftrace,label='Red line:PSP;   Blue line: Output SOP')
        ax.legend(fontsize=15, loc='lower center')
        ax.plot(s1, s2, s3, color=coloroftrace, linewidth=1, linestyle='dashed')
        ax.scatter(sop1psp[0], sop2psp[0], sop3psp[0], marker="o", color='red', label='Output')
        ax.plot(sop1psp, sop2psp, sop3psp, color='red', linewidth=1, linestyle='dashed')
        ax.quiver(s1[0], s2[0], s3[0], s1[1]-s1[0], s2[1]-s2[0], s3[1]-s3[0], length=2, color='red', arrow_length_ratio=0.5,linewidth=2)
        ax.quiver(sop1psp[0], sop2psp[0], sop3psp[0], sop1psp[1] - sop1psp[0], sop2psp[1] - sop2psp[0], sop3psp[1] - sop3psp[0], length=2, color='blue',
                  arrow_length_ratio=0.5, linewidth=2)

class broadband_light:

    def __init__(self, center_wavelength_nm, linewidth_nm):  # wavelength in vacuum
        self.centerwl = center_wavelength_nm
        self.linewidth = linewidth_nm
        self.linewidth_freq= linewidth_nm /center_wavelength_nm/ center_wavelength_nm*2.99792458e17
        self.centerfreq=self.wl_to_freq(center_wavelength_nm)

    def wl_to_angularfreq(wavelength_nm):
        angularfreq=2*np.pi*2.99792458e17/wavelength_nm
        return angularfreq

    def wl_to_freq(wavelength_nm):
        angularfreq =  2.99792458e17 / wavelength_nm
        return angularfreq

    def angularfreq_to_wl(angularfreq):
        wl_nm = 2 * np.pi * 2.99792458e17 / angularfreq
        return wl_nm

    def freq_to_wl(freq):
        wl = 2.99792458e17 / freq
        return wl

    def get_rectangular_spectrum(self, start_wavelength_nm, end_wavelength_nm, points):
        start_freq = self.wl_to_freq(start_wavelength_nm)
        end_freq = self.wl_to_freq(end_wavelength_nm)
        freq=np.linspace(start_freq,end_freq,points)
        e=np.zeros(points)
        for i in range (points):
            if freq[i]>=(self.centerfreq-self.linewidth_freq/2)  and freq[i]<=(self.centerfreq-self.linewidth_freq/2):
                e[i]=1/self.linewidth_freq
        return e,freq

    def get_Gaussian_spectrum(self, start_wavelength_nm, end_wavelength_nm, points):
        start_freq = self.wl_to_freq(start_wavelength_nm)
        end_freq = self.wl_to_freq(end_wavelength_nm)
        freq=np.linspace(start_freq,end_freq,points)
        e=np.zeros(points)
        s=np.zeros(points)
        for i in range (points):
            s[i]=2*sqrt(log(2))/sqrt(np.pi)/self.linewidth_freq*exp(-(2*sqrt(log(2))/self.linewidth_freq*(freq[i]-self.centerfreq)^2))
            e[i]=sqrt(s[i])
        return s, e, freq

    def get_Lorentzian_spectrum(self, start_wavelength_nm, end_wavelength_nm, points):
        start_freq = self.wl_to_freq(start_wavelength_nm)
        end_freq = self.wl_to_freq(end_wavelength_nm)
        freq=np.linspace(start_freq,end_freq,points)
        e=np.zeros(points)
        s=np.zeros(points)
        for i in range (points):
            s[i]=1/np.pi*self.linewidth_freq/2/((freq[i]-self.centerfreq)^2+(self.linewidth_freq/2)^2)
            e[i]=sqrt(s[i])
        return s, e,freq

class depolarizer:
    def __init__(self, dn_array,length_array,angle_array,wavelength_nm):  # dphase=phase_y-phase_x
        self.dn_array = dn_array
        self.length_array = length_array
        self.angle_array = angle_array
        self.wavelength = wavelength_nm


'''
class Reflection_ISO:
     def __init__(self, normal, ni, nr):
         # normal of reflective surface
         # ni:  the refractive index of the medium where the incident ray travels
         # nr:  the refractive index of the medium where the refractiveray travels
         self.normal=normal
         self.ni=ni
         self.nr=nr

     def reflected_ray(self, incident_ray):
         reflectedRay= np.dot(self.normal,incident_ray)    -incident_ray


 class PolEllispe
     # the x-axis is horizontal and the y-axis is vertical.
     semi_major_a =1  # semi-major axis
     semi_minor_b =1  # semi-minor axis
     azimuth=0        # the angle of major axis from x-axis
     sense ="CW"
     def get_

'''
'''
DGD_array=[1,2,3,2,5]
angle_array=[0,np.pi/4,0,np.pi/3,np.pi/6]
PMD1=PMD(DGD_array,angle_array, 1550)
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
DGD,PSP,PDCD,Depolarization,wl,angularfreq=PMD1.get_PMD_Spectrum(1540,1570,500)
ax.plot(wl,Depolarization)
plt.show()

'''