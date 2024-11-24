# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 18:44:44 2022

@author: Jonathan Cox

This script was used to visualize the effect of range measurement noise on
the trilateration of a single particle.
"""

import numpy as np
from matplotlib import pyplot as plt

from Scripts.DataGenerators import VectorPiecewiseRandom as VecRand
from Scripts.Multilaterators import Trilaterator as Tri

trueStyle = {
    'linestyle' : '-'
    }

errStyle = {
    'linestyle' : '--',
    'linewidth' : 0.5,
    'zorder' : 0
    }

rank = 3
steps = 1000

yBounds = ((-10, 10), (-10, 10), (0, 10))
tSpace = np.linspace(0, 1, steps)
shiftChances = [0.01 for i in range(rank)]

pointWalk = VecRand(yBounds, tSpace, shiftChances, rank)
trilat = Tri(0.01)

trueCoords = []
fuzzCoords = []
trilatCoords = []
for time in tSpace:
    truCoord = pointWalk.integrate(time)
    trilatCoord = trilat.ranges2Point(trilat.point2Ranges(truCoord))
    trilatCoords.append(trilatCoord)
    trueCoords.append(truCoord)
    fuzzCoords.append(trilat.fuzzPoint(truCoord))

trueCoords = np.array(trueCoords).T
fuzzCoords = np.array(fuzzCoords).T
trilatCoords = np.array(trilatCoords).T
error = np.sum(np.abs(trueCoords - fuzzCoords), 0)

Fig, Ax = plt.subplots(2, 1, sharex=True)
Ax[0].plot(tSpace, trueCoords[0], color='b', **trueStyle)
Ax[0].plot(tSpace, fuzzCoords[0], color='b', **errStyle)
Ax[0].plot(tSpace, trueCoords[1], color='g', **trueStyle)
Ax[0].plot(tSpace, fuzzCoords[1], color='g', **errStyle)
Ax[0].plot(tSpace, trueCoords[2], color='r', **trueStyle)
Ax[0].plot(tSpace, fuzzCoords[2], color='r', **errStyle)
Ax[1].plot(tSpace, error)

Fig2, Ax2 = plt.subplots()
Ax2.plot(trueCoords[2], error)

