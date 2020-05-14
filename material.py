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
global m_path, m_book
m_path = './material/data.csv'
m_book = './material/book.csv'

def new_material_book():
    with open(m_path, 'w+'):
        pd.DataFrame({}).to_csv(m_path)
    with open(m_book, 'w+'):
        pd.DataFrame({}).to_csv(m_book)
    print('new_material_book')

def open_book():
    bk = pd.read_csv(m_book, index_col = 0)
    print(bk.columns[0], bk[0:].values[0][0])

def save_material(m, name):
    cls_name = type(m).__name__
    _save_book(name, cls_name)
    if cls_name == 'Non_Dispersion':
        _save_material_data(m.n, name, '_mono_n')
        _save_material_data(m.k, name, '_mono_k')
    elif cls_name == 'Material':
        _save_material_data(m.n, name, '_n')
        _save_material_data(m.k, name, '_k')
        _save_material_data(m.wl, name, '_w')
        _save_material_data(m.wl_k, name, '_wk')
    elif cls_name == 'Sellmeier':
        _save_material_data(m.theta, name, '_SE')
        _save_material_data(m.k, name, '_k')
        _save_material_data(m.wl_k, name, '_wk')
    else:
        print('material type undefined')

def _save_book(name, cls_name, cover = False):
    bk = pd.read_csv(m_book)
    if (name in bk.columns) and cover == False:
        print('Data already exist, set cover = True to overwrite the previous data')
    else:
        _save_material_data(cls_name, name, '')

def _save_material_data(m_data, m_name, ext):
    suffix = ['_SE', '_w', '_n', '_wk', '_k', '_mono_n', '_mono_k']
    data_name = m_name + ext
    if ext in suffix:
        _save(m_data, data_name, m_path)
    else:
        _save([m_data], data_name, m_book)
        
def _save(m_data, data_name, m_path):
    name = data_name
    if path.exists(m_path):
        file = pd.read_csv(m_path, index_col=0)
        if name in file.columns:
            file[name] = pd.Series(m_data)
        else:
            data = pd.DataFrame({name:m_data})
            new_file = pd.concat([file, data], axis = 1)
        new_file.to_csv(m_path)
        print('{} has saved successfully'.format(m_path))
    else:
        new_material_book(m_path)

def load_material_all():
    bk = pd.read_csv('./material/book.csv')
    
def open_material(m_name, object_kind = None):
    m_file = pd.read_csv(m_path, index_col=0)
    if object_kind == 'Material':
        m = Material(nan_remover(m_file[m_name+'_w']), nan_remover(m_file[m_name+'_n']))
    elif object_kind == 'Sellmeier':
        m = Sellmeier(nan_remover(m_file[m_name+'_SE']))
    elif object_kind == 'Non_Dispersion':
        m = Non_Dispersion(nan_remover(m_file[m_name+'_mono_n']), nan_remover(m_file[m_name+'_mono_k']))
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