# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 14:16:02 2022

@author: Jonathan Cox
"""

import numpy as np


class Position():
    def __init__(self, coordinates):
        """
        Simple container for translational states/transformations

        Parameters
        ----------
        coordinates : n-dimensional list-like of numbers
            Describes the position of a particle with respect to the ground
            frame.
        """
        self.coordinates = np.array(coordinates)
    
    def reconcile(self, other):
        """Reconciles with a competing position via averaging."""
        return Position((self.coordinates + other.coordinates) / 2)
    
    def __add__(self, other):
        """Addition behavior defined via vector addition."""
        return Position(self.coordinates + other.coordinates)
    
    
class Rotation():
    def __init__(self, matrix):
        self.matrix = np.matrix(matrix)
        self._updateAnglesFromMatrix()
    
    def __matmul__(self, other):
        newMatrix = self.matrix @ other.matrix
        return self.__class__(newMatrix)


def angle2Matrix(angle):
    C = np.cos(angle)
    S = np.sin(angle)
    R = np.matrix([[C, -S],
                   [S,  C]])
    return R


class Rotation2D(Rotation):
    def from_angle(cls, angle):
        matrix = angle2Matrix(angle)
        return cls(matrix)
    
    def _updateMatrixFromAngles(self):
        self.matrix = angle2Matrix(self.angle)
        
    def _updateAnglesFromMatrix(self):
        matrix = self.matrix
        self.angles = [np.arctan2(matrix[0, 1], matrix[0, 0])]
    
    def reconcile(self, other):
        sAngle = self.angles[0]
        oAngle = other.angles[0]
        sinA, cosA = np.sin(sAngle), np.cos(sAngle)
        sinO, cosO = np.sin(oAngle), np.cos(oAngle)
        avAngle = np.arctan2(sinA + sinO, cosA + cosO)
        return Rotation2D.from_angle(avAngle)
    

def euler2Matrix(angles):
    C = [np.cos(angle) for angle in angles]
    S = [np.sin(angle) for angle in angles]
    Rx = np.matrix([[1,    0,     0],
                    [0, C[0], -S[0]],
                    [0, S[0],  C[0]]])
    Ry = np.matrix([[C[1],  0, S[1]],
                    [  0,   1,    0],
                    [-S[1], 0, C[1]]])
    Rz = np.matrix([[C[2], -S[2], 0],
                    [S[2],  C[2], 0],
                    [   0,     0, 1]])
    matrix = Rz @ (Ry @ Rx)
    return matrix


class Rotation3D(Rotation):
    @classmethod
    def from_axisAngle(cls, axis, angle):
        unitAxis = axis / np.linalg.norm(axis)
        C = np.cos(angle)
        S = np.sin(angle)
        ux = np.matrix([[           0, -unitAxis[2],  unitAxis[1]],
                        [ unitAxis[2],            0, -unitAxis[0]],
                        [-unitAxis[1],  unitAxis[0],            0]])
        matrix = (C * np.identity(3) + S * ux
                  + (1 - C) * np.outer(unitAxis, unitAxis))
        return cls(matrix)
        
    @classmethod
    def from_eulerAngles(cls, angles):
        matrix = euler2Matrix(angles)
        return cls(matrix)
    
    def _updateMatrixFromAngles(self):
        """
        Translates euler angles into matrix form.
        Meant for internal use only.
        """
        self.matrix = euler2Matrix(self.eulerAngles)
        
    def _updateAnglesFromMatrix(self):
        matrix = self.matrix
        beta = -np.arcsin(matrix[2, 0])
        Cb = np.cos(beta) # Used to correct signs when calculating alpha/gamma
        if Cb == 0:
            raise ValueError("Gimbal Lock")
        # When the inputs are non-complex, atan2 is used to properly constrain
        # the output range of the euler angles.
        if not np.iscomplexobj(matrix):
            alpha = np.arctan2(matrix[2, 1] / Cb, matrix[2, 2] / Cb)
            gamma = np.arctan2(matrix[1, 0] / Cb, matrix[0, 0] / Cb)
        # Arctan in unambiguous on the complex plane, on which atan2 is also
        # undefined. This carve-out is necessary to enable complex-step
        # approximation of the jacobian.
        else:
            alpha = np.arctan(matrix[2, 1] / matrix[2, 2])
            gamma = np.arctan(matrix[1, 0] / matrix[0, 0])
        self.angles = np.array([alpha, beta, gamma])
        
    def reconcile(self, other):
        # Elementwise-average matrix of the two competing rotation matricies.
        Ravg = (self.matrix + other.matrix) / 2
        # Singlular value decomposition of Ravg. s is discarded.
        U, s, Vt = np.linalg.svd(Ravg, full_matrices=True)
        # Attempts to identify new 'closest' rotation matrix
        newMatrix = U @ Vt
        return Rotation3D(newMatrix)
        

class Orientation(): # My god I think this just natively works with 2D and 3D
    def __init__(self, position, rotation):
        self.position = position
        self.rotation = rotation
        self.generalCoords = np.concatenate([position.coordinates,
                                             rotation.angles])

    def reconcile(self, other):
        newPosition = self.position.reconcile(other.position)
        newRotation = self.rotation.reconcile(other.rotation)
        return self.__class__(newPosition, newRotation)

    def __matmul__(self, other):
        newRotation = other.rotation @ self.rotation
        newPosition = other.position + self.position
        return self.__class__(newPosition, newRotation)
    

class Orientation2D(Orientation):
    @classmethod
    def from_generalCoords(cls, generalCoords):
        positionCoords = generalCoords[:2]
        angle = generalCoords[2]
        position = Position(positionCoords)
        rotation = Rotation2D.from_angle(angle)
        return cls(position, rotation)


class Orientation3D(Orientation):
    @classmethod
    def from_generalCoords(cls, generalCoords):
        positionCoords = generalCoords[:3]
        angles = generalCoords[3:]
        position = Position(positionCoords)
        rotation = Rotation3D.from_eulerAngles(angles)
        return cls(position, rotation)


def subIndicies(targetList, valueList):
    newList = []
    for item in targetList:
        # Checks for non-string iterables
        if (hasattr(item, '__iter__')
            and not isinstance(item, str)):
            newList.append(subIndicies(item, valueList))
        else:
            newList.append(valueList[item])
    return newList


class Body():
    def __init__(self, verticies, trackedVerticies):
        self.verticies = np.array(verticies)
        self.trackedVerticies = np.array(trackedVerticies)

    def transformVertex(self, vertexIndex, transformation):
        bodyCoord = self.verticies[vertexIndex]
        groundCoord = bodyCoord
        groundCoord = transformation.rotation.matrix @ groundCoord
        groundCoord = groundCoord.A1
        groundCoord += transformation.position.coordinates
        return groundCoord
    
    def transformBody(self, transformation):
        groundVerticies = []
        for i in range(self.verticies.shape[0]):
            singleVertex = self.transformVertex(i, transformation)
            groundVerticies.append(singleVertex)
        groundVerticies = np.array(groundVerticies)
        return groundVerticies
    
    def getTrackedVerticies(self, transformation):
        groundVerticies = self.transformBody(transformation)
        return subIndicies(self.trackedVerticies, groundVerticies)


class Body2D(Body):
    def __init__(self, verticies, trackedVerticies, edges):
        super().__init__(verticies, trackedVerticies)
        self.edges = edges
    
    def getEdges(self, transformation):
        groundVerticies = self.transformBody(transformation)
        return subIndicies(self.edges, groundVerticies)


class Body3D(Body2D):
    def __init__(self, verticies, trackedVerticies, edges=None, faces=None):
        super().__init__(verticies, trackedVerticies, edges)
        self.faces = faces
        
    def getFaces(self, transformation):
        groundVerticies = self.transformBody(transformation)
        return subIndicies(self.faces, groundVerticies)
    





