import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from numpy import pi, sin, cos, arcsin, dot

class Design:
    def __init__(self, ambient, thinfilm, substrate, wl, angle = 0):
        self.wavelength = wl
        self.ambient = chromatic_n(ambient['a'], self.wavelength)
        self.substrate = chromatic_n(substrate['s'], self.wavelength)
        self.thinfilm = thinfilm
        self.angle = angle*pi/180
        
    def eq_tf_matrix(self, pl):
        eq_matrix = identity_matrix(self.wavelength)
        for i in range(np.size(self.thinfilm)):
            eq_matrix = matrix_dot(
                tf_matrix(self.thinfilm[-i-1], self.wavelength, 
                          pl, self.ambient, self.angle), eq_matrix)
        return eq_matrix

    def eq_admittance(self, pl):
        eq_matrix = self.eq_tf_matrix(pl)
        theta_s = arcsin(self.ambient/self.substrate*sin(self.angle))
        if pl == "S":
            eq_y = bc(eq_matrix, self.substrate*cos(theta_s)/cos(self.angle), self.wavelength)
        elif pl == "P":
            eq_y = bc(eq_matrix, self.substrate*cos(self.angle)/cos(theta_s), self.wavelength)
        else:
            eq_y = bc(eq_matrix, self.substrate, wl)
        return eq_y
    
    def eq_Y(self, pl, rt):
        eq_y = self.eq_admittance(pl)
        if rt == "T":
            return eq_y
        elif rt == "R":
            Y = eq_y['C']/eq_y['B']
            return Y.values.reshape(np.size(self.wavelength), 1)
    
    def transmittance(self, pl):
        eq_y = self.eq_Y(pl, "T")
        t = (self.ambient*eq_y["B"]+eq_y["C"]).values
        T = 4*self.ambient*np.real(self.substrate)/(t*t.conjugate())
        return T
    
    def T(self, pl = None):
        if self.angle != 0:
            if pl == "S":
                return self.transmittance("S")
            elif pl == "P":
                return self.transmittance("P")
            else:
                T_S = self.transmittance("S")
                T_P = self.transmittance("P")
                return (T_S+T_P)/2
        else:
            return self.transmittance(None)
    
    def reflectance(self, pl):
        eq_Y = self.eq_Y(pl, "R").flatten()
        r = (self.ambient-eq_Y)/(self.ambient+eq_Y)
        reflectance = np.reshape(r*r.conjugate(), np.size(eq_Y))
        return np.real(reflectance)
    
    def R(self, pl = None):
        if self.angle != 0:
            if pl == "S":
                return self.reflectance("S")
            elif pl == "P":
                return self.reflectance("P")
            else:
                R_S = self.reflectance("S")
                R_P = self.reflectance("P")
                return (R_S+R_P)/2
        else:
            return self.reflectance(None)

def bc(eq, ns, wl):
    m = np.size(wl)
    ita_s = np.reshape(pd.DataFrame({"one":np.ones(m),"ita":ns}).values.reshape(-1, 1), (m, 2, 1))
    YY = [dot(eq[i], ita_s[i]) for i in range(m)]
    bc = pd.DataFrame(np.reshape(YY, (m,2)), columns = ['B','C'])
    return bc

def chromatic_n(m, wl, pl=0, n0=0, theta0=0):
    n, k = globals()[material[m]['type']](material[m], wl)
    theta_n = arcsin(n0*sin(theta0)/n)
    if pl == "S":
        return (n - 1j*k)*cos(theta_n)/cos(theta0)
    elif pl == "P":
        return (n - 1j*k)*cos(theta0)/cos(theta_n)
    else:
        return n - 1j*k

def tf_matrix(layer, wl, pl, n0, theta0):
    m = matrix(layer['m'], layer['d'], wl, pl, n0, theta0)
    return m

def matrix(m, t, wl, pl, n0, theta0):
    ita = chromatic_n(m, wl, pl, n0, theta0) 
    if pl == "S" or pl == "P":
        delta = 2*pi*ita*t/wl*cos(theta0)
    else:
        delta = 2*pi*ita*t/wl
    element = matrix_element(ita, delta)
    return np.reshape(element.values.reshape(1,-1), (np.size(wl), 2, 2))

def matrix_element(ita, delta):
    e = pd.DataFrame(
        {'e1':cos(delta), 'e2':1j/ita*sin(delta), 
         'e3':1j*ita*sin(delta), 'e4':cos(delta)})
    return e
    
def matrix_dot(layer_up, layer_bot): 
    w, _, _ = np.shape(layer_up)
    eq = [dot(layer_up[i], layer_bot[i]) for i in range(w)]
    return eq

def identity_matrix(wl):
    m = np.size(wl)
    i = pd.DataFrame({'e1':np.ones(m), 'e2':np.zeros(m), 'e3':np.zeros(m), 'e4':np.ones(m)})
    i_matrix = np.reshape(i.values.reshape(1,-1), (m, 2, 2))
    return i_matrix
    
material = {
    'TiO2':{'A':5.913, 'B':0.2441, 'C':0.0803, 'type':'TiO2'},
    'SiO2':{'A':0.6961663,'B':0.0684043,'C':0.4079426,'D':0.1162414,'E':0.8974794,'F':9.896161, 'type':'SiO2'},
    'sodalime':{'type':'sodalime', 'material':'sodalime', 'A':1.5130, 'B':0.003169, 'C':0.003962},
    'air':{'type':'air', 'material':'air'},
    'ITO':{'type':'ITO', 'material':'ITO'},
    'SiN':{'type':'SiN', 'material':'SiN'},
    'acrylic':{'A':1.1819, 'B':0.011313, 'type':'acrylic'},
    'acrylic_h':{'type':'acrylic_h'},
    'CrYAG':{'type':'CrYAG'},
    'LOCA':{'type':'LOCA'},
    'GG3':{'type':'GG3'}
}
def sodalime(m, x):
    x = x/1000
    n = m['A']-m['B']*x**2+m['C']*x**-2
    file = pd.read_csv('{}.csv'.format(m['material']))
    f_k = interp1d([float(i) for i in file['wl']], [float(i) for i in file['k']])
    return n, f_k(x)

def ITO(m, x):
    file = pd.read_csv('{}.csv'.format(m['material']))
    f_n = interp1d([float(i) for i in file[:381]['wl']], [float(i) for i in file[:381]['n']])
    f_k = interp1d([float(i) for i in file[:381]['wl']], [float(i) for i in file[382:]['n']])
    #k = 1e-2* np.ones(np.size(x))
    return f_n(x/1000), f_k(x/1000)

def SiN(m,x):
    file = pd.read_csv('{}.csv'.format(m['material']))
    f_n = interp1d([float(i) for i in file[:146]['wl']], [float(i) for i in file[:146]['n']])
    f_k = interp1d([float(i) for i in file[:146]['wl']], [float(i) for i in file[147:]['n']])
    #k = 1e-2* np.ones(np.size(x))
    return f_n(x/1000), f_k(x/1000)

def TiO2(m, x):
    x = x/1000
    n = (m['A']+m['B']/(x**2-m['C']))**.5
    k = np.zeros(np.size(x))
    return n, k

def SiO2(m, x):
    x = x/1000
    n = (1+m['A']/(1-(m['B']/x)**2)+m['C']/(1-(m['D']/x)**2)+m['E']/(1-(m['F']/x)**2))**.5
    k = 1e-3*np.ones(np.size(x))
    return n, k

def acrylic(m, x):
    x = x/1000
    n = (1+m['A']/(1-m['B']/x**2))**.5
    k = 1e-4*np.ones(np.size(x))
    return n, k

def acrylic_h(m, x):
    n = 1.65*np.ones(np.size(x))
    k = 0*np.ones(np.size(x))
    return n, k

def CrYAG(m, x):
    n = 1.8*np.ones(np.size(x))
    k = 0*np.ones(np.size(x))
    return n, k

def LOCA(m, x):
    n = 1.41*np.ones(np.size(x))
    k = 1e-6*np.ones(np.size(x))
    return n, k

def GG3(m, x):
    n = 1.51*np.ones(np.size(x))
    k = 1e-5*np.ones(np.size(x))
    return n, k

def air(m, x):
    n = np.ones(np.size(x))
    k = np.zeros(np.size(x))
    return n, k

def sec_reflec(T_measure, n0, ns):
    T_sec = 1-((n0-ns)/(n0+ns))**2
    return (100/T_measure - 1/T_sec + 1)**-1

def trans(n0, n1):
    return 1-((n0-n1)/(n0+n1))**2