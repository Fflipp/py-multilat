# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 18:30:27 2022

@author: Jonathan Cox
"""

import numpy as np
from random import gauss as normal

class Multilaterator():
    def __init__(self, anchorPoints, noise=0):
        self.noise = noise
        self.anchorPoints = anchorPoints
    """
    Parent class for n-dimensional multilateration methods. __init__ is
    ommitted because the stored data for each dimension is unique, but several
    easily vector-scaling functions are included here.
    """
    def point2Ranges(self, coordinate):
        """
        Takes in a point in cartesian space and returns the distance of that
        point from each anchor point.

        Parameters
        ----------
        coordinate : n-dimensional list-like of real numbers.
            Describes the position of a particle in n-dimensional space.

        Returns
        -------
        m-dimensional list-like of real numbers, where m is the number of
        anchor points.
            The euclidian distances of the point from each anchor point.
            The order corresponds to the order in which the anchor point
            coordinates are stored within the object. This is not standardized,
            take care.
        """
        ranges = []
        # Iterates over each anchor point seperately
        for point in self.anchorPoints:
            # Applies simple euclidean distance formula
            range2 = sum([(point[i] - coordinate[i]) ** 2
                          for i in range(len(coordinate))])
            ranges.append(range2 ** (1 / 2))
        return np.array(ranges)
    
    def point2FuzzyRanges(self, coordinate):
        # Masks Mulitlaterator.point2Ranges() with gaussian noise of a
        # magnitude proportional to the true measurement.
        ranges = self.point2Ranges(coordinate)
        for i in range(len(ranges)):
            ranges[i] += normal(0, self.noise * ranges[i] / 2.33)
        return ranges
    
    def fuzzPoint(self, coordinate):
        """
        1. Takes a coordinate in Cartesian space
        2. Identifies the ranges of that point from each anchor point
        3. Masks those ranges with gaussian noise
        4. Maps the noise-injected ranges back to a different cartesian
            coordinate

        Parameters
        ----------
        coordinate : n-dimensional list-like of real numbers.
            Describes the position of a particle in n-dimensional space.

        Returns
        -------
        fuzzCoord : n-dimensional numpy array of real numbers
            The trilaterated coordinates of the point when measruements are 
            subjected to gaussian noise.
        """
        ranges = self.point2FuzzyRanges(coordinate)
        fuzzCoord = self.ranges2Point(ranges)
        return fuzzCoord


class Trilaterator(Multilaterator):
    def __init__(self, legBase=1, noise=0):
        """
        Corresponds to the 3-D trilateration case. I would like to expand this
        to easily extend to additional and more flexible anchor-points on
        demand.

        Parameters
        ----------
        legBase : number, optional
            The distance of the non-origin anchor points from the origin.
            The default is 1.
        noise : float, optional
            Determines the level of noise injected into each function call.
            The noise is gaussianly distributed, with a mean of 0 and a
            standard devation of noise * trueMeasurement. The default is 0.
        """
        self.noise = noise
        # Anchor points are currently largely hard-coded. This is bad.
        self.anchorPoints = ((0, 0, 0), (legBase, 0, 0), (0, legBase, 0))
    
    def ranges2Point(self, ranges):
        # This partially depends on the enforced structure of the anchor points.
        # I dislike this strongly, but the good news is it doesn't even get
        # used anywhere I don't think. I'll have to run a scream test later.
        coordinate = [(1 + ranges[0] ** 2 - ranges[1] ** 2) / 2,
                      (1 + ranges[0] ** 2 - ranges[2] ** 2) / 2]
        z2 = ranges[0] ** 2 - coordinate[0] ** 2 - coordinate[1] ** 2
        if z2 < 0:
            return np.array([float('nan')] * 3)
        coordinate.append(np.sqrt(ranges[0] ** 2
                                  - coordinate[0] ** 2
                                  - coordinate[1] ** 2))
        return np.array(coordinate)

            
class Bilaterator(Multilaterator):
    def __init__(self, base=1, noise=0):
        """
        Corresponds to the 2-D bilateration case. I would like to expand this
        to easily extend to additional and more flexible anchor-points on
        demand.

        Parameters
        ----------
        base : number, optional
            The distance of the non-origin anchor point from the origin.
            The default is 1.
        noise : float, optional
            Determines the level of noise injected into each function call.
            The noise is gaussianly distributed, with a mean of 0 and a
            standard devation of noise * trueMeasurement. The default is 0.
        """
        self.noise = noise
        self.anchorPoints = ((0, 0), (base, 0))
    
    def ranges2Point(self, ranges):
        # This depends less strongly on the structure of the points.
        # It would be cool to get the arbitrary anchor coordinates + transform
        # thing going, but that sounds hellish and also like it would
        # dramatically amplify my error, which is already pretty bad.
        coordinate = []
        coordinate.append((self.anchorPoints[0][1] ** 2
                           + ranges[0] ** 2 - ranges[1] ** 2)
                          / (2 * self.anchorPoints[0][1]))
        coordinate.append(np.sqrt(ranges[0] ** 2 - coordinate[0] ** 2))
        return np.array(coordinate)























