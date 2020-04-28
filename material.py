import numpy as np
import pandas as pd
from os import path
from scipy.interpolate import interp1d
from scipy.optimize import basinhopping, minimize

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
        try:
            return f(wl)
        except:
            return f(wl/1000)
        
    def kvalues(self, wl):
        f = interp1d(self.wl_k, self.k, kind = 'cubic')
        try:
            return f(wl)
        except:
            return f(wl/1000)

    def nk(self, wl):
        return self.nvalues(wl) - 1j*self.kvalues(wl)

class Sellmeier:
	def __init__(self, theta, k = 0, wl_k = 0):
		self.theta = theta
		self.k = k
		self.wl_k = wl_k

	def nvalues(self, wl):
		theta = self.theta
		wl = wl/1000
		n = theta[0] + theta[1]/(1-theta[2]/wl**2) + theta[3]/(1-theta[4]/wl**2) + theta[5]/(1-theta[6]/wl**2)
		return n**.5

	def kvalues(self, wl):
		try:
			f = interp1d(self.wl_k, self.k, kind = 'cubic')
			return f(wl)
		except:
			return self.k*np.ones(np.size(wl))

	def nk(self, wl):
		return self.nvalues(wl) - 1j*self.kvalues(wl)

def sellmeier_fitting(target_w, target_n, init = np.zeros(7), save = False, name = None):
    wl = np.array(target_w)/1000
    hypo = lambda theta: theta[0] + theta[1]/(1-theta[2]/wl**2) + theta[3]/(1-theta[4]/wl**2) + theta[5]/(1-theta[6]/wl**2)
    fom = lambda theta: sum(abs(hypo(theta) - np.array(target_n)**2))
    res = basinhopping(fom, init)
    print(res.message[0])
    if save: save_material(res.x, name)
    return res.x

### material_book ###
global m_path
m_path = 'material.csv'

def open_material_book():
    file = pd.read_csv(m_path, index_col = 0)
    for i in sorted(file.columns.values):
        print(i)
    
def save_material(m_data, m_name, ext):
    suffix = ['_SE', '_w', '_n', '_wk', '_k']
    if ext in suffix:
        _save(m_data, m_name, ext)   
    else:
        print('use {} as suffix'.format(suffix))

def _save(m_data, m_name, ext):
    name = m_name + ext
    if path.exists(m_path):
        file = pd.read_csv(m_path, index_col=0)
        if name in file.columns:
            file[name] = pd.Series(m_data)
        else:
            data = pd.DataFrame({name:m_data})
            new_file = pd.concat([file, data], axis = 1)
        new_file.to_csv(m_path)
    else:
        new_material_book(m_path)

def new_material_book():
    with open(m_path, 'w+'):
        pd.DataFrame({}).to_csv(m_path)
    print('new_material_book')

def open_material(m_name, object_kind = None):
    m_file = pd.read_csv(m_path, index_col=0)
    if object_kind == 'Material':
        m = Material(nan_remover(m_file[m_name+'_w']), nan_remover(m_file[m_name+'_n']))
    elif object_kind == 'Sellmeier':
        m = Sellmeier(nan_remover(m_file[m_name+'_SE']))
    else:
        print('material database not founded')
    try:
        m.wl_k = nan_remover(m_file[m_name+'_wk'].values)
        m.k = nan_remover(m_file[m_name+'_k'].values)
    except:
        m.wl_k = nan_remover(m_file[m_name+'_w'].values)
        m.k = nan_remover(m_file[m_name+'_k'].values)
    finally:
        return m

def nan_remover(v):
    return [x for x in v if str(x) != 'nan']