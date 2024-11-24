# -*- coding: utf-8 -*-
"""
Created on Sat Nov  5 14:48:04 2022

@author: Jonathan Cox
"""

import numpy as np
from scipy.optimize import least_squares as gaussNewton
from .Orientations import Orientation3D, Position, Rotation3D

EULER_BOUNDS = np.array(((-np.inf, np.inf),
                         (-np.inf, np.inf),
                         (0, np.inf),
                         (-np.pi - 1, np.pi + 1),
                         (-np.pi / 2 - 1, np.pi / 2 + 1),
                         (-np.pi - 1, np.pi + 1))).T

class MultilaterationSim3D():
    def __init__(self, trilaterator, velFunc, omegaFunc, body, startingOrientation):
        # Saves components
        self.trilaterator = trilaterator
        self.body = body
        self.startingOrientation = startingOrientation
        self.velFunc = velFunc
        self.omegaFunc = omegaFunc
        # Initializes lists for estimation methods.
        # The first term in these lists must be handled manually, because
        # any dead-reckoning attempt will require at least one previous point.
        self.trueOrientation = [startingOrientation]
        self.tSpace = [0]
        self.GNOrientation = [startingOrientation]
        self.reconciledOrientation = [startingOrientation]
        self.deadReckonedOrientation = [startingOrientation]
        self.noisyVel = [velFunc(0)]
        self.trueVel = [velFunc.trueValue(0)]
        self.noisyOmega = [omegaFunc(0)]
        self.trueOmega = [omegaFunc.trueValue(0)]
        
    def _getTrueValue(self, time):
        positionChange = Position(self.velFunc.integrate(time))
        rotationChange = Rotation3D(self.omegaFunc.integrate(time))
        OrientationChange = Orientation3D(positionChange, rotationChange)
        newOrientation = OrientationChange @ self.startingOrientation
        return newOrientation
    
    def _getGNRaw(self, time, ftol=1e-8, referenceIndex=None):
        # Loads frequently called attributes into the method frame
        body = self.body
        tLat = self.trilaterator
        currentOrientation = self._getTrueValue(time)
        # Obtains noise-masked ranges
        trueCoords = body.getTrackedVerticies(currentOrientation)
        fuzzRanges = np.concatenate([tLat.point2FuzzyRanges(trueCoord) for
                                     trueCoord in trueCoords])
        # Defines the residual function, which is then fed to the solver
        def residuals(generalCoords):
            orientation = Orientation3D.from_generalCoords(generalCoords)
            testCoords = body.getTrackedVerticies(orientation)
            gnRanges = np.concatenate([tLat.point2Ranges(coord)
                                       for coord in testCoords])
            return gnRanges - fuzzRanges
        # The guessing behavior, uses the most recent orientation estimation
        # or the true orientation if that isn't available.
        if referenceIndex != None:
            guess = self.reconciledCoords[referenceIndex]
        else:
            guess = self.startingOrientation.generalCoords
        # Performes the optimization and returns the generalized coordinates
        # I could possibly add a verbose mode to this... It's something to
        # think about.
        generalCoords = gaussNewton(residuals, guess, jac='cs', bounds=EULER_BOUNDS,
                           ftol=ftol)['x']
        return Orientation3D.from_generalCoords(generalCoords)
    
    def _deadReckon(self, time, referenceIndex):
        # Loads in reference data
        referenceOrientation = self.reconciledOrientation[referenceIndex]
        referenceTime = self.tSpace[referenceIndex]
        elapsedTime = time - referenceTime
        
        # This section handles the change in rotation
        steadyOmega = self.noisyOmega[referenceIndex]
        angularSpeed = np.linalg.norm(steadyOmega)
        angularDirection = steadyOmega / angularSpeed
        x, y, z = angularDirection
        K = np.matrix([[ 0, -z,  y],
                       [ z,  0, -x],
                       [-y,  x,  0]])
        # Applies the Rodriguez rotation formula to calculate rotation
        # matrix for the time period
        angle = angularSpeed * elapsedTime
        rotation = np.identity(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
        rotation = Rotation3D(rotation)
        
        # This section handles the translation
        steadyVelocity = self.noisyVel[referenceIndex]
        translation = Position(steadyVelocity * elapsedTime)
        
        # Combines rotation and translation into a new orientation
        orientationChange = Orientation3D(translation, rotation)
        newOrientation = orientationChange @ referenceOrientation
        return newOrientation
    
    def _getReconciled(self, gnRaw, deadReckon):
        reconciledOrientation = gnRaw.reconcile(deadReckon)
        return reconciledOrientation
    
    def getEstimate(self, time, ftol=1e-8, verbosity='quiet'):
        # Checks if the requested time has already been evaluated
        if time in self.tSpace:
            index = np.where(self.tSpace == time)[0][0]
            if verbosity == 'loud':
                return (self.reconciledOrientation[index],
                        self.trueOrientation[index],
                        self.GNOrientation[index],
                        self.deadReckonedOrientation[index])
            if verbosity == 'quiet':
                return self.reconciledOrientation[index]
            return None
        # Identifies the nearest previous evaluated time to pass to
        # _getGNRaw() and _deadReckon()
        referenceIndex = np.searchsorted(self.tSpace, time, 'right')
        self.trueOmega.insert(referenceIndex, self.omegaFunc.trueValue(time))
        self.trueVel.insert(referenceIndex, self.velFunc.trueValue(time))
        self.noisyOmega.insert(referenceIndex, self.omegaFunc(time))
        self.noisyVel.insert(referenceIndex, self.velFunc(time))
        # Evaluates estimation methods
        trueValue = self._getTrueValue(time)
        gnRaw = self._getGNRaw(time, ftol)
        deadReckon = self._deadReckon(time, referenceIndex - 1)
        reconciled = self._getReconciled(gnRaw, deadReckon)
        # Records values
        self.tSpace.insert(referenceIndex, time)
        self.trueOrientation.insert(referenceIndex, trueValue)
        self.GNOrientation.insert(referenceIndex, gnRaw)
        self.deadReckonedOrientation.insert(referenceIndex, deadReckon)
        self.reconciledOrientation.insert(referenceIndex, reconciled)
        # Return behavior
        if verbosity == 'loud':
            return reconciled, trueValue, gnRaw, deadReckon
        if verbosity == 'quiet':
            return reconciled
            
            
            
        