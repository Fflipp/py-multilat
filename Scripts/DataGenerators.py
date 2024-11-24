# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 18:00:40 2022

@author: Jonathan Cox
"""

import numpy as np

from random import gauss as normal
from random import random as rand


def step(x):
    """Heaviside step function"""
    if x < 0:
        return 0
    return 1


def ramp(x):
    """Ramp function (integral of heaviside step)"""
    if x < 0:
        return 0
    return x


def randRange(lowerBound, upperBound):
    """
    Wrapper function to simplify the notation of an arbitrary 
    uniform distribution.
    """
    return lowerBound + rand() * (upperBound - lowerBound)

class PiecewiseRandom():
    def __init__(self, yBounds, domain, shiftChance, noise=0):
        """
        A class which represents a piecewise-constant function which takes on 
        random values at randomly assigned breakpoints.

        Parameters
        ----------
        yBounds : Interval-like; Aka, a length-2 iterable which contains
        numeric values in the form (a, b) such that a < b. 
            This specifies the range of values which the funciton may take.
        domain : List-like of numeric values
            Specifies the possible points at which the funciton may change
            values. Additionally, the function will be undefined outside of
            (min(domain), max(domain)).
        shiftChance : A float between 0 and 1 (inclusive)
            The likelyhood that the function will change value at any of the
            points listed within domain.
        noise : float, optional
            Determines the level of noise injected into each function call.
            The noise is gaussianly distributed, with a mean of 0 and a
            standard devation of noise * trueMeasurement. The default is 0.
        """
        values = [randRange(yBounds[0], yBounds[1])]
        breakpoints = [domain[0]]
        # Loops over domain points and selects which will become breakpoints
        # at which the function changes value
        breakIndicies = []
        for stepIndex in range(len(domain)):
            if rand() < shiftChance:
                values.append(randRange(yBounds[0], yBounds[1]))
                breakIndicies.append(stepIndex)
                breakpoints.append(domain[stepIndex])
        # In order to properly integrate, it is necessary that the final
        # breakpoint lie some arbitrarrily small distance beyond the
        # domain of the function.
        breakpoints.append(domain[-1] + 1)
        
        # Stores object Data
        self.noise = noise
        self.domain = np.array([domain[0], domain[-1]])
        self.values = np.array(values)
        self.breakpoints = np.array(breakpoints)
        self.breakIndicies = np.array(breakIndicies)
    
    # Note, fault calling behavior of the function is noisy
    def __call__(self, x):
        trueValue = self.trueValue(x)
        noise = normal(0, self.noise * abs(trueValue))
        return trueValue + noise
    
    # Checks that the test value lies within the domain of the function
    def _domainCheck(self, x):
        domain = self.domain
        if (x < domain[0]) or (x > domain[1]):
            raise ValueError("Input value " + str(x) + "is outside "
                             + "function domain: [" + str(domain[0]) + ", "
                             + str(domain[1]) + "]")
        return None
    
    # Returns the value of the function at x without noise
    def trueValue(self, x):
        self._domainCheck(x)
        # Iterate to find the first breakpoint greater than t
        i = 0
        breakpoints = self.breakpoints
        while breakpoints[i] <= x:
            i += 1
        # We take the value at [i-1] because t is associated with the value
        # of the nearest *preceeding* breakpoint
        return self.values[i-1]

    # Returns the definite integral from self.domain[0] to x
    def integrate(self, x):
        self._domainCheck(x)
        output = 0
        values = self.values
        breakpoints = self.breakpoints
        for i in range(len(breakpoints) - 1):
            lowerBreak = (x - breakpoints[i])
            upperBreak = (x - breakpoints[i+1])
            output += (values[i] * (ramp(lowerBreak) - ramp(upperBreak)))
        return output
    
    # Returns an arbitrary bounded integral from x0 to xf
    def boundedIntegral(self, x0, xf):
        self._domainCheck(x0)
        self._domainCheck(xf)
        return self.integrate(xf) - self.integrate(x0)

    def generatePlotData(self, tSpace, noisy=True):
        """
        Formats plotting data for easier display with numpy. Inserts nan
        values at points of discontinuity so that they do not appear connected
        on the graph.

        Parameters
        ----------
        tSpace : list-like of numeric values.
            The points at which the function is to be evaluated.
        noisy : Boolean, optional
            A flag that indicates whether the function should also return
            noise-masked values. The default is True.

        Returns
        -------
        The following returns are all vectors with the same length as the
        supplied tSpace vector. (plus the number of discontinuites)
        
        ftSpace
            tSpace formatted with nan breaks.
        pData
            function outputs formatted with nan breaks.
        nData
            noise-masked function outputs formatted with nan breaks.
        """
        ftSpace = []
        pData = []
        nData = []
        for i in range(len(tSpace)):
            # Inserts a 'nan' value into every relevant array at each index
            # where a discontinuity occurs
            if i in self.breakIndicies:
                ftSpace.append(float('nan'))
                nData.append(float('nan'))
                pData.append(float('nan'))
            ftSpace.append(tSpace[i])
            pData.append(self.trueValue(tSpace[i]))
            nData.append(self(tSpace[i]))
            
        ftSpace = np.array(ftSpace)
        pData = np.array(pData)
        nData = np.array(nData)
        if noisy == False:
            return ftSpace, pData
        return ftSpace, pData, nData
        

class VectorPiecewiseRandom():
    def __init__(self, yBounds, domain, shiftChance, rank, noise=None):
        """
        A vectorized form of PiecewiseRandom.

        Parameters
        ----------
        yBounds : n-dimension List-like of interval-like values
            Describes the maximum possible range of each function.
        domain : List-like of numeric values
            Specifies the possible points at which the funcitons may change
            values. Additionally, the functions will be undefined outside of
            (min(domain), max(domain)). 
        shiftChance : List-like of numeric values. 
            The likelyhood that the functions will change value at any of the
            points listed within domain.
        rank : integer
            The rank of the vector which the function will return.
        noise : list-like of floats, optional
            Determines the level of noise injected into each function call.
            The noise is gaussianly distributed, with a mean of 0 and a
            standard devation of noise * trueMeasurement. The default is None.
        """
        self.rank = rank
        if noise == None:
            noise = [0 for i in range(rank)]
        self.scalarFunctions = [PiecewiseRandom(yBounds[i], domain,
                                                shiftChance[i], noise=noise[i])
                           for i in range(rank)]
    
    """
    The following functions simply loop over the constituent scalar functions
    and collect their values into a vector.
    """
    def __call__(self, x):
        return np.array([self.scalarFunctions[i](x)
                         for i in range(self.rank)])
    
    def trueValue(self, x):
        return np.array([self.scalarFunctions[i].trueValue(x)
                         for i in range(self.rank)])
    
    def integrate(self, x):
        return np.array([self.scalarFunctions[i].integrate(x)
                         for i in range(self.rank)])
    
    def boundedIntegral(self, x0, xf):
        return np.array([self.scalarFunctions[i].boundedIntegral(x0, xf)
                         for i in range(self.rank)])

    def generatePlotData(self, tSpace, noisy=True):
        """
        Formats plotting data similar to its scalar counterpart. This function,
        however, returns 2 dimensional arrays.

        Parameters
        ----------
        tSpace : list-like of numeric values.
            The points at which the function is to be evaluated.
        noisy : Boolean, optional
            A flag that indicates whether the function should also return
            noise-masked values. The default is True.

        Returns
        -------
        The following returns are all n x m matricies, where n is the rank of
        the function and m is the length of the supplied tSpace vector
        (plus the number of continuities).
        
        ftSpace
            tSpace formatted with nan breaks.
        pData
            function outputs formatted with nan breaks.
        nData
            noise-masked function outputs formatted with nan breaks.
        """
        ftSpace = []
        pData = []
        nData = []
        # Iterates over each scalar function
        for i in range(self.rank):
            # Generates plot data for each function individually and adds the
            # resulting vectors to the output arrays
            singlePlotData = \
                self.scalarFunctions[i].generatePlotData(tSpace, noisy=True)
            ftSpace.append(singlePlotData[0])
            pData.append(singlePlotData[1])
            nData.append(singlePlotData[2])
        if noisy == False:
            return ftSpace, pData
        return ftSpace, pData, nData


class RotationPiecewiseRandom(VectorPiecewiseRandom):
    def __init__(self, yBounds, domain, shiftChance, noise=None):
        rank = 3
        super().__init__(yBounds, domain, shiftChance, rank, noise)
        # Merges individual breakpoints & corresponding indicies into a
        # single sorted list. This is necessary because for rotation, the
        # individual function values can no longer be treated as independent
        # when integrating.
        breakpoints = np.concatenate([[0]] + [function.breakpoints[1:-1] for
                                           function in self.scalarFunctions])
        breakpoints.sort(kind='mergesort')
        self.breakpoints = np.append(breakpoints, [domain[-1] + 1])
    
    def integrate(self, x):
        # Loads discontinuity times into the function
        breakpoints = self.breakpoints
        # This is a list of rotations which result from periods of constant
        # angular velocity
        sequentialRotations = []
        # Iterates over points of discontintuity to calculate the rotation
        # which occurs under each period of constant angular velocity
        for i in range(len(breakpoints) - 1):
            # Time elapsed under the given period. Uses ramp functions to
            # greatly simplify the algorithm at the minor cost of computational
            # efficiency.
            elapsedTime = ramp(x - breakpoints[i]) - ramp(x - breakpoints[i+1])
            # Loads up angular velocity vector from constituent scalar
            # functions
            angularVelocity = self.trueValue(breakpoints[i])
            # Calculates magnitude and unit vector
            angularSpeed = np.linalg.norm(angularVelocity)
            angularDirection = angularVelocity / angularSpeed
            # Unpacks the unit vector into components and constructs the
            # skew-symmetric cross-product matrix
            a, b, c = angularDirection
            K = np.matrix([[ 0, -c,  b],
                           [ c,  0, -a],
                           [-b,  a,  0]])
            # Applies the Rodriguez rotation formula to calculate rotation
            # matrix for the time period
            angle = angularSpeed * elapsedTime
            rotation = np.identity(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
            sequentialRotations.append(rotation)
        # Composes the identified rotation components into the net rotation
        finalRotation = np.identity(3)
        for rotation in sequentialRotations:
            finalRotation = rotation @ finalRotation
        return finalRotation
    
    def boundedIntegral(self, x0, xf):
        # Solves for the rotation matrix relating the body's orientation at
        # times x0 and xf
        Rxf = self.integrate(xf)
        Rx0 = self.integrate(x0)
        return Rxf @ Rx0.I