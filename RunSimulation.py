# -*- coding: utf-8 -*-
"""
Created on Sun Oct 30 18:39:35 2022

@author: Jonathan Cox
"""
#------------------------------------------------------------------------------
"""
Import statements and boilerplate
"""

# Standard library imports
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import rcParams as Settings
from matplotlib.patches import Patch
from matplotlib.animation import FuncAnimation as FAnim
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection

# Manually sets the filepath for the image-processing library, a copy of which
# is included with this module
Settings['animation.ffmpeg_path'] = r".\\ffmpeg\\bin\\ffmpeg.exe"

# Imports from this module
from Scripts.Simulator import MultilaterationSim3D as MSim
from Scripts.DataGenerators import VectorPiecewiseRandom as VecRand
from Scripts.DataGenerators import RotationPiecewiseRandom as RotRand
from Scripts.Orientations import Body3D as Body
from Scripts.Orientations import Orientation3D
from Scripts.Multilaterators import Multilaterator as MLat

# Specifies the standard color scheme used in all plots
COLOR_MAP = {
    'TRUE' : '#127d0b',
    'DR' : '#ff0000',
    'LS' : '#0000ff',
    'RE' : '#9553b9',
    'ANCHOR' : '#500000'
    }

# Specifies the file in which plots and animations will be saved, relative
# to the location of this script when it is executed
OUTPUT_DESTINATION = r'Outputs//'

"""
Sets up and runs the simulation
"""
#%%----------------------------------------------------------------------------
# Specifies the vertecies of the body in body-frame coordinates
bodyVerticies = ((0, 0, 0),
                 (2.5, 0, 0),
                 (0, 2.5, 0),
                 (0, 0, 2.5))
# Lists which verticies serve as beacon locations.
# Values are indicies of the desired verticies within the bodyVerticies tuple.
beacons = (1, 2, 3)
# Creates the body object
body = Body(bodyVerticies, beacons)

# Defines the anchor points in ground-frame coordinates
anchorPoints = ((0, 0, 0),
                (10,  0,  0),
                (0, 10,  0),
                (0, 0, 10))
# Sets up the multilateration scheme
multilaterator = MLat(anchorPoints, 0.05)

# Sets up the domain for the velocity profiles. This is also recycled as the
# evaluation points of the simulation.
steps = 1000
tSpace = np.linspace(0, 5, steps)
# Specifies the expected number of discontinuities in the velocity profiles
shiftChances = [0.01 for i in range(3)]
# Specifies the bounds on the velocity profiles
vBounds = [(-2, 2), (-2, 2), (0, 1)]
wBounds = [(-2, 2), (-2, 2), (-2, 2)]
# Generates 3D linear velocity profile
bodyWalk = VecRand(vBounds, tSpace, shiftChances, 3)
# Generates 3D rotational velocity profile
bodySpin = RotRand(wBounds, tSpace, shiftChances)

# Specifies the starting orientation of the body.
# Pattern is [x, y, z, phi (x-angle), theta (y-angle), psi (z-angle)]
startingOrientation = Orientation3D.from_generalCoords([5, 5, 2.5, 0, 0, 0])

# creates the simulation object
simulation = MSim(multilaterator, bodyWalk, bodySpin, body, startingOrientation)

for t in tSpace:
    # Performs simulation evaluation at a single time step
    simulation.getEstimate(t, ftol=1e-14)
    # Handles console logged progress-reports
    print("Simulation progress: " + str(round(t, 2)) + "s/" + str(tSpace[-1]) + "s")


# Formats generalized coordinate data for ease of use
TrueCoords = np.array([simulation.trueOrientation[i].generalCoords for i in range(steps)]).T
GNCoords = np.array([simulation.GNOrientation[i].generalCoords for i in range(steps)]).T
DRCoords = np.array([simulation.deadReckonedOrientation[i].generalCoords for i in range(steps)]).T
RECoords = np.array([simulation.reconciledOrientation[i].generalCoords for i in range(steps)]).T
# Formats rotation matrix data for ease of use
TrueRot = np.array([simulation.GNOrientation[i].rotation.matrix for i in range(steps)])
GNRot = np.array([simulation.trueOrientation[i].rotation.matrix for i in range(steps)])
DRRot = np.array([simulation.deadReckonedOrientation[i].rotation.matrix for i in range(steps)])
RERot = np.array([simulation.reconciledOrientation[i].rotation.matrix for i in range(steps)])

"""
Plots the position-coordinates and euler-angles of the body individually.
This doesn't provide a very good measure of error on its own, instead, it's
mostly useful for recognizing trends or patterns that cause instability.
"""
#%%----------------------------------------------------------------------------
# The following dictionaries and tuples store plot-specific data and settings.
# This pattern was used to enable the generation of all six plots with a
# for-loop.
dataCycler = {
    'TRUE' : TrueCoords,
    'LS' : GNCoords,
    'DR' : DRCoords,
    'RE' : RECoords
    }
labelCycler = {
    'TRUE' : 'True Value',
    'LS' : 'Least-Squares Estimate',
    'DR' : 'Dead Reckoned Estimate',
    'RE' : 'Reconciled Estimate'
    }
zCycler = {
    'TRUE' : 6,
    'LS' : 3,
    'DR' : 4,
    'RE' : 5
    }
titles = ('x-position',
          'y-position',
          'z-position',
          r'$\phi$ (x-euler angle)',
          r'$\theta$ (y-euler angle)',
          r'$\psi$ (z-euler angle)')
yLims = ((None, None),
         (None, None),
         (None, None),
         (-np.pi, np.pi),
         (-np.pi / 2, np.pi / 2),
         (-np.pi, np.pi))
fileNames = ('xData.png',
             'yData.png',
             'zData.png',
             'phiData.png',
             'thetaData.png',
             'psiData.png')
# Used for manual adjustment of legend positions, leave values as 0
legendPositions=[0, 0, 0, 0, 0, 0]
# Creates each figure
coordPlots = [plt.subplots(dpi=300) for i in range(6)]
# Loops over figures to populate them with data
for plotIndex in range(6):
    for key in dataCycler:
        # Loads the properties of the desired plot
        props = {
            'linewidth' : 0.8,
            'label' : labelCycler[key],
            'color' : COLOR_MAP[key],
            'zorder' : zCycler[key]
            }
        # The extra [1] index is a result of the return signature of plt.subplots()
        # It accesses the figure's corresponding axis.
        # Only every 10th data point is plotted to avoid visual overload.
        coordPlots[plotIndex][1].plot(tSpace[::10], dataCycler[key][plotIndex][::10],
                                      **props)
    # Handles standard plot formatting
    coordPlots[plotIndex][1].margins(x=0)
    coordPlots[plotIndex][1].set_ylim(yLims[plotIndex])
    coordPlots[plotIndex][1].set_title(titles[plotIndex])
    coordPlots[plotIndex][1].set_xlabel('Elapsed Time $(s)$')
    coordPlots[plotIndex][1].grid()
    coordPlots[plotIndex][1].legend(loc=legendPositions[plotIndex])
    # Saves each figure to the specified output folder after it is completed
    coordPlots[plotIndex][0].savefig(OUTPUT_DESTINATION + fileNames[plotIndex])
    
"""
Plots positional and rotational errors
"""
#%%----------------------------------------------------------------------------
# Calculates the % difference between true and estimate position magnitudes.
# Repeated for each estimate.
GNPosErr = [100 * np.linalg.norm(TrueCoords[:3, i] - GNCoords[:3, i]) /
            np.linalg.norm(TrueCoords[:, i]) for i in range(steps)]
# Calculates the frobenius norm of the difference between true and estimate
# rotation matricies.
# Repeated for each estimate.
GNRotErr = [np.linalg.norm(TrueRot[i] - GNRot[i]) for i in range(steps)]

DRPosErr = [100 * np.linalg.norm(TrueCoords[:3, i] - DRCoords[:3, i]) /
            np.linalg.norm(TrueCoords[:, i]) for i in range(steps)]
DRRotErr = [np.linalg.norm(TrueRot[i] - DRRot[i]) for i in range(steps)]

REPosErr = [100 * np.linalg.norm(TrueCoords[:3, i] - RECoords[:3, i]) /
            np.linalg.norm(TrueCoords[:, i]) for i in range(steps)]
RERotErr = [np.linalg.norm(TrueRot[i] - RERot[i]) for i in range(steps)]

AvPosErrs = np.mean([GNPosErr, DRPosErr, REPosErr], axis=1)
AvRotErrs = np.mean([GNRotErr, DRRotErr, RERotErr], axis=1)

# Generates linear position error plot
peFig, peAx = plt.subplots(dpi=300)
peAx.set_title("Normalized Positioning Error of Multiple Estimation Methods")
peAx.set_ylabel("% Error")
peAx.set_xlabel("Time $(s)$")
# Only every 10th value is plotted to avoid visual overload
peAx.plot(tSpace[::10], GNPosErr[::10],
          color=COLOR_MAP['LS'], linewidth=0.9, label='Least-Squares~' + str(round(AvPosErrs[0], 3)))
peAx.plot(tSpace[::10], DRPosErr[::10],
          color=COLOR_MAP['DR'], linewidth=0.9, label='Dead Reckoned~' + str(round(AvPosErrs[1], 3)))
peAx.plot(tSpace[::10], REPosErr[::10],
          color=COLOR_MAP['RE'], linewidth=0.9, label='Reconciled~' + str(round(AvPosErrs[2], 3)))
peAx.set_xlim(tSpace[0], tSpace[-1])
peAx.set_ylim(0)
peAx.grid()
peAx.legend()
# Saves the figure to the specified output folder
peFig.savefig(OUTPUT_DESTINATION + 'positioningError.png')

# Generates orientation error plot
reFig, reAx = plt.subplots(dpi=300)
reAx.set_title("Normalized Orientation Error of Multiple Estimation Methods")
reAx.set_ylabel("Frobenious Norm of Error Matrix")
reAx.set_xlabel("Time $(s)$")
# Only every 10th value is plotted to avoid visual overload
reAx.plot(tSpace[::10], GNRotErr[::10],
          color=COLOR_MAP['LS'], linewidth=0.9, label='Least-Squares~' + str(round(AvRotErrs[0], 3)))
reAx.plot(tSpace[::10], DRRotErr[::10],
          color=COLOR_MAP['DR'], linewidth=0.9, label='Dead Reckoned~' + str(round(AvRotErrs[1], 3)))
reAx.plot(tSpace[::10], RERotErr[::10],
          color=COLOR_MAP['RE'], linewidth=0.9, label='Reconciled~' + str(round(AvRotErrs[2], 3)))
reAx.set_xlim(tSpace[0], tSpace[-1])
reAx.set_ylim(0)
reAx.grid()
reAx.legend()
# Saves the figure to the specified output folder
reFig.savefig(OUTPUT_DESTINATION + 'rotationError.png')


"""
Generates the 3D-animation of the body's motion
"""
#%%----------------------------------------------------------------------------

faceColors = [COLOR_MAP['TRUE'], 'grey', 'grey', 'grey']
polySet = Poly3DCollection([], facecolor=faceColors, alpha=0.7)
trackingLines = Line3DCollection([], color=COLOR_MAP['ANCHOR'], linestyle='--',
                                 linewidth=0.5)

Fig = plt.figure(dpi=200)
Ax = Axes3D(Fig, auto_add_to_figure=False)
Fig.add_axes(Ax)

bodyMarkers, = Ax.plot3D([], [], [], color=COLOR_MAP['TRUE'], linestyle='none',
                        marker='o', markersize=5)
gnTri, = Ax.plot3D([], [], [], color=COLOR_MAP['LS'])
drTri, = Ax.plot3D([], [], [], color=COLOR_MAP['DR'])
reTri, = Ax.plot3D([], [], [], color=COLOR_MAP['RE'])

gnLegendPatch = Patch(color=COLOR_MAP['LS'], label='Least-Squares Position')
drLegendPatch = Patch(color=COLOR_MAP['DR'], label='Dead Reckoned Position')
reLegendPatch = Patch(color=COLOR_MAP['RE'], label='Reconciled Position')

anchors = np.array(simulation.trilaterator.anchorPoints)

def animInit():
    Ax.add_collection3d(polySet)
    Ax.add_collection3d(trackingLines)
    Ax.set_xlim3d(0, 10)
    Ax.set_ylim3d(10, 0)
    Ax.set_zlim3d(0, 10)
    Ax.plot3D(anchors.T[0], anchors.T[1], anchors.T[2],
              linestyle='none', marker='o', color=COLOR_MAP['ANCHOR'],
              markersize=5)
    Ax.legend(handles=[drLegendPatch, reLegendPatch, gnLegendPatch])
    
def animate(frame):
    beacons = body.getTrackedVerticies(simulation.trueOrientation[frame])
    
    verticies = [beacons]
    for coordinateIndex in range(len(verticies[0])):
        flatVerts = np.copy(verticies[0])
        flatVerts[:, coordinateIndex] = 0
        verticies.append(flatVerts)
    verticies = np.array(verticies)
    polySet.set(verts=verticies)
    
    bodyMarkers.set_data_3d(np.array(beacons).T)
    
    trackingSegments = []
    for beacon in beacons:
        for anchor in anchors:
            trackingSegments.append((anchor, beacon))
    trackingLines.set_segments(trackingSegments)
        
    drData = body.getTrackedVerticies(simulation.deadReckonedOrientation[frame])
    drData.append(drData[0])
    drData = np.array(drData).T
    drTri.set_data_3d(drData)
    
    reData = body.getTrackedVerticies(simulation.reconciledOrientation[frame])
    reData.append(reData[0])
    reData = np.array(reData).T
    reTri.set_data_3d(reData)
    
    gnData = body.getTrackedVerticies(simulation.GNOrientation[frame])
    gnData.append(gnData[0])
    gnData = np.array(gnData).T
    gnTri.set_data_3d(gnData)
    
    print("Frame " + str(frame + 1) + "/" + str(steps) + " completed")
    
animation = FAnim(Fig, animate, frames=steps, init_func=animInit)

animation.save(OUTPUT_DESTINATION + '3DTriangle.mp4', fps=60)
