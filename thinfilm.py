import numpy as np
import pandas as pd
import copy
from numpy import pi, sin, cos, arcsin, dot

class Design:
    def __init__(self, material, thickness):
        layer = list(map(list, zip(material, thickness)))
        self.ambient = layer[0][0]
        self.substrate = layer[-1][0]
        self.middle = layer[1:-1]
    
    def matrix(self, layer, wl, angle, pl):
        if angle != 0:
            theta_n = arcsin(self.ambient.nvalues(wl)*sin(angle)/layer[0].nvalues(wl))
            ita = angular_dispersion(layer, wl, angle, pl, theta_n)
            delta = 2*pi*ita*layer[1]*cos(theta_n)/wl
        else:
            ita = layer[0].nk(wl)
            delta = 2*pi*ita*layer[1]/wl
        element = matrix_element(ita, delta)
        return np.reshape(element.values.reshape(1,-1), (np.size(wl), 2, 2))
    
    def eq_tf_matrix(self, wl, angle, pl):
        eq_matrix = identity_matrix(wl)
        for layer in self.middle[::-1]:
            eq_matrix = matrix_dot(self.matrix(layer, wl, angle, pl), eq_matrix)
        return eq_matrix

    def eq_admittance(self, wl, angle, pl):
        eq_matrix = self.eq_tf_matrix(wl, angle, pl)
        theta_s = arcsin(self.ambient.nvalues(wl)/self.substrate.nvalues(wl)*sin(angle))
        if pl == "S":
            eq_y = bc(eq_matrix, self.substrate.nk(wl)*cos(theta_s)/cos(angle), wl)
        elif pl == "P":
            eq_y = bc(eq_matrix, self.substrate.nk(wl)*cos(angle)/cos(theta_s), wl)
        else:
            eq_y = bc(eq_matrix, self.substrate.nk(wl), wl)
        return eq_y
    
    def eq_Y(self, wl, angle, pl, rt):
        eq_y = self.eq_admittance(wl, angle, pl)
        if rt == "T":
            return eq_y
        elif rt == "R":
            Y = eq_y['C']/eq_y['B']
            return Y.values.reshape(np.size(wl), 1)
    
    def _transmittance(self, wl, angle, pl):
        eq_y = self.eq_Y(wl, angle, pl, "T")
        t = (self.ambient.nk(wl)*eq_y["B"]+eq_y["C"]).values
        T = 4*self.ambient.nk(wl)*np.real(self.substrate.nk(wl))/(t*t.conjugate())
        return T.real

    def transmittance(self, wl, angle = 0, pl = None):
        if angle != 0:
            angle = angle*pi/180
            if pl == "S":
                return self._transmittance(wl, angle, "S")
            elif pl == "P":
                return self._transmittance(wl, angle, "P")
            else:
                return self._transmittance(wl, angle, "avg")
        else:
            return self._transmittance(wl, angle, pl)
    
    def _reflectance(self, wl, angle, pl):
        eq_Y = self.eq_Y(wl, angle, pl, "R").flatten()
        r = (self.ambient.nk(wl)-eq_Y)/(self.ambient.nk(wl)+eq_Y)
        reflectance = np.reshape(r*r.conjugate(), np.size(eq_Y))
        return np.real(reflectance)   
        
    def reflectance(self, wl, angle = 0, pl = None):
        if angle != 0:
            angle = angle*pi/180
            if pl == "S":
                return self._reflectance(wl, angle, "S")
            elif pl == "P":
                return self._reflectance(wl, angle, "P")
            else:
                return self._reflectance(wl, angle, "avg")
        else:
            return self._reflectance(wl, angle, pl)

def angular_dispersion(layer, wl, theta_0, pl, theta_n):
    if pl == "S":
            return layer[0].nk(wl)*cos(theta_n)/cos(theta_0)
    elif pl == "P":
            return layer[0].nk(wl)*cos(theta_0)/cos(theta_n)
    elif pl == 'avg':
            return layer[0].nk(wl)*(cos(theta_0)/cos(theta_n)+cos(theta_n)/cos(theta_0))/2
            
def identity_matrix(wl):
    m = np.size(wl)
    i = pd.DataFrame({'e1':np.ones(m), 'e2':np.zeros(m), 'e3':np.zeros(m), 'e4':np.ones(m)})
    i_matrix = np.reshape(i.values.reshape(1,-1), (m, 2, 2))
    return i_matrix

def matrix_element(ita, delta):
    e = pd.DataFrame(
        {'e1':cos(delta), 'e2':1j/ita*sin(delta), 
         'e3':1j*ita*sin(delta), 'e4':cos(delta)})
    return e

def matrix_dot(layer_up, layer_bot): 
    w, _, _ = np.shape(layer_up)
    eq = [dot(layer_up[i], layer_bot[i]) for i in range(w)]
    return eq

def bc(eq, ns, wl):
    m = np.size(wl)
    ita_s = np.reshape(pd.DataFrame({"one":np.ones(m),"ita":ns}).values.reshape(-1, 1), (m, 2, 1))
    YY = [dot(eq[i], ita_s[i]) for i in range(m)]
    bc = pd.DataFrame(np.reshape(YY, (m,2)), columns = ['B','C'])
    return bc

def margin(model, tol, wl):
    margin_test = copy.deepcopy(model)
    layer_margin = []
    for i, layer in enumerate(margin_test.middle):
        init_d = layer[-1]
        margin = []
        for j in np.linspace(-tol, tol, 20*tol+1):
            margin_test.middle[i][-1] = init_d+j
            margin.append(np.mean(margin_test.reflectance(wl)))
        layer_margin.append([max(margin), min(margin), max(margin)-min(margin)])
        margin_test.middle[i][-1] = init_d
    return layer_margin[::-1]