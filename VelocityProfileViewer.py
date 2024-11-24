# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 22:13:21 2022

@author: Jonathan Cox

This script is used to plot individual velocity-profile input functions for
visualization purposes.
"""

from matplotlib import pyplot as plt
from matplotlib import rcParams as Settings

from Scripts.DataGenerators import PiecewiseRandom as pRand


Settings['figure.dpi'] = 300

yBounds = (0, 10)
steps = 1000
odds = (steps / 10) ** -1

tSpace = tuple(n * 10 / steps for n in range(steps + 1))

dFunc = pRand(yBounds, tSpace, odds, noise=0.05)

ftSpace, pData, nData = dFunc.generatePlotData(tSpace, noisy=True)
iData = [dFunc.integrate(t) for t in tSpace]

Fig, Ax = plt.subplots()
Ax.plot(ftSpace, pData, label='True Values', color='blue')
Ax.plot(ftSpace, nData, alpha=0.6, label='Noisy Values', linewidth=0.5, color='orange')
Ax.legend()
Ax.grid()
