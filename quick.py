import numpy as np
import pandas as pd
import material as ml
import thinfilm as tm
import matplotlib.pyplot as plt

air = ml.Non_Dispersion(1)
sodalime = ml.open_material('sodalime', 'Material')
SiO2 = ml.open_material('SiO2_LENS', 'Sellmeier')
SiN = ml.open_material('SiN_LENS', 'Sellmeier')
ITO = ml.open_material('ITO_LENS', 'Sellmeier')
OC = ml.open_material('OC_LENS', 'Material')

hypo_bri = lambda th: tm.Design(
    [air, OC, SiN, SiO2, ITO, OC, ITO, SiO2, SiN, sodalime],
    [None, th[0], th[1], th[2], th[3], th[4], th[5], th[6], th[7], None]
)

hypo_dia = lambda th: tm.Design(
    [air, OC, SiN, SiO2, ITO, SiO2, SiN, sodalime],
    [None, th[0], th[1], th[2], th[3], th[4], th[5], None]
)

hypo_tra = lambda th: tm.Design(
    [air, OC, SiN, SiO2, SiO2, SiN, sodalime],
    [None, th[0], th[1], th[2], th[3], th[4], None]
)

fit_bri = lambda th: tm.Design(
    [air, OC, SiN, SiO2, ITO, OC, ITO, SiO2, SiN, sodalime],
    [None, th[0]-th[4], th[1], th[2], th[3], th[4], th[5], th[6], th[7], None]
)

fit_dia = lambda th: tm.Design(
    [air, OC, SiN, SiO2, ITO, SiO2, SiN, sodalime],
    [None, th[0], th[1], th[2], th[5], th[6], th[7], None]
)

fit_tra = lambda th: tm.Design(
            [air, OC, SiN, SiO2,SiO2, SiN, sodalime],
    [None, th[0], th[1], th[2], th[6], th[7], None]
)