import numpy as np
import pandas as pd

r = pd.read_csv('./CIE/CIE1931.csv')['R'].values
g = pd.read_csv('./CIE/CIE1931.csv')['G'].values
b = pd.read_csv('./CIE/CIE1931.csv')['B'].values

r_it = sum(r)
g_it = sum(g)
b_it = sum(b)

def f(sigma):
    if sigma > 0.008856:
        return np.cbrt(sigma)
    else:
        return (12/841)*sigma+(4/29)

L = lambda sp: 116*f(sum(sp*g)/g_it)-16
a_star = lambda sp: 500*(f(sum(sp*r)/r_it)-f(sum(sp*g)/g_it))
b_star = lambda sp: 200*(f(sum(sp*g)/g_it)-f(sum(sp*b)/b_it))

def Lab(spe):
    return L(spe), a_star(spe), b_star(spe)