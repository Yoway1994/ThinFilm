import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from os import path
from scipy.interpolate import interp1d
from scipy.optimize import basinhopping

class Non_Dispersion:
	
	def __init__(self, n, k = 0):
		self.n = n
		self.k = k
	def nvalues(self, wl):
		n = self.n*np.ones(np.size(wl))
		return n
		
	def kvalues(self, wl):
		k = self.k*np.ones(np.size(wl))
		return k
		
	def nk(self, wl):
		n = self.nvalues(wl)
		k = self.kvalues(wl)
		return n - 1j*k

class Material:
    def __init__(self, wl, n, k = None, wl_k = None):
        self.wl = wl
        self.n = n
        self.k = np.zeros(np.size(wl))
        self.wl_k = wl

    def nvalues(self, wl):
        f = interp1d(self.wl, self.n, kind = 'cubic')
        return f(wl)
    
    def kvalues(self, wl):
        f = interp1d(self.wl_k, self.k, kind = 'cubic')
        return f(wl)

    def nk(self, wl):
        return self.nvalues(wl) - 1j*self.kvalues(wl)

class Sellmeier:
    def __init__(self, theta, k = 0, wl_k = 0):
        self.theta = theta
        self.k = k
        self.wl_k = wl_k
        
    def nvalues(self, wl):
        theta = self.theta
        n = theta[0] + theta[1]/(1-theta[2]/wl**2) + theta[3]/(1-theta[4]/wl**2) + theta[5]/(1-theta[6]/wl**2)
        return n**.5
    
    def kvalues(self, wl):
        try:
            f = interp1d(self.wl_k, self.k, kind = 'cubic')
            print('k values interpolated')
            return f(wl)
        except:
            print('single k values ')
            return self.k*np.ones(np.size(wl))
        
    def nk(self, wl):
        return self.nvalues(wl) - 1j*self.kvalues(wl)

def sellmeier_fitting(target_w, target_n, init = np.ones(7), save = False):
    wl = np.array(target_w)
    hypo = lambda theta: theta[0] + theta[1]/(1-theta[2]/wl**2) + theta[3]/(1-theta[4]/wl**2) + theta[5]/(1-theta[6]/wl**2)
    fom = lambda theta: sum(abs(hypo(theta) - np.array(target_n)**2))
    res = basinhopping(fom, init)
    print(res.message[0])
    if save: save_material(res.x)
    return res.x

def save_material(m_data, m_name):
    if path.exists('material.csv'):
        file = pd.read_csv('material.csv', index_col=0)
        file[m_name] = pd.Series(m_data)
        file.to_csv('material.csv')
    else:
        with open('material.csv', 'w+'):
            pd.DataFrame({}).to_csv('material.csv')

def open_material(m_name):
    try:
        material = pd.read_csv('material.csv', index_col = 0)
        print('material database open')
        return Sellmeier(material[m_name])
    except:
        print('material database not founded')