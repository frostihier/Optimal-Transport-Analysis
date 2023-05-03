# multidimensional analysis with optimal transport

from lib.header_notebook import *
from lib.misc import *
import lib.SinkhornNP as Sinkhorn
import matplotlib.pyplot as plt
import lib.LinW2 as LinW2
from numpy import load, loadtxt, median, mean


## settings for figures
import matplotlib.font_manager as fm
import matplotlib as mpl

# Collect all the font names available to matplotlib
font_names = [f.name for f in fm.fontManager.ttflist]
print(font_names)

# Edit the font, font size, and axes width
mpl.rcParams['font.family'] = 'Montserrat'
plt.rcParams['font.size'] = 18
plt.rcParams['axes.linewidth'] = 2


## determine paths:
filenameRoot = "Data/Data-for-TransportAnalysis/"
resultpath =  'Results/'

if not os.path.isdir(resultpath):
    os.makedirs(resultpath)

##  load point cloud for reference measure
ptsRef = load(filenameRoot + 'pts_ref.npy')
nPtsRef = ptsRef.shape[0]

## load analysis parameters
with open("Data/params.txt", "r") as f:
    params = json.load(f)
    f.close()

nSamples = params["nSamples"]
epsTarget = params["epsTarget"]
epsInit = params["epsInit"]
SinkhornErrorGoal = params["SinkhornErrorGoal"]
keep = params["keep"]

## classes, class names and colors
classes = np.array(loadtxt(filenameRoot + "/classes"), order='C', dtype=int).ravel()
classlabel = ["Control", "MS"]
featureNames = ["Volume", "Electron density", "Heterogeneity", "Sphericity", "Num-Neighbors", "NN-Distance"]
colors[0] = 'Blue'
colors[1] = 'Red'

## load cov matrices and means of samples
matList = np.array(load(filenameRoot + 'matList.npy'), order='C', dtype=np.double)
meanList = np.array(loadtxt(filenameRoot + 'meanList'), order='C', dtype=np.double)
nFeatures = matList.shape[1]

## Linear embeddings: point clouds

# Load linear embedding of sample point clouds

linEmb = np.array(load(filenameRoot + "/linEmb.npy"), order="C", dtype=np.double)
muRef = np.full(nPtsRef, 1. / nPtsRef)

# center linEmbs and do coordinate re-weighting to "standard Euclidean inner product"
linEmbMean = np.mean(linEmb, axis=0)
linEmbEucl = linEmb - linEmbMean
linEmbEucl = np.einsum(linEmbEucl, [0, 1, 2], np.sqrt(muRef), [1], [0, 1, 2])
dimtan = linEmbEucl.shape[1] * linEmbEucl.shape[2]
linEmbEucl = linEmbEucl.reshape((nSamples, dimtan))

##  Linear embedding of Gaussians

# compute barycenter in bures metric
weights = np.full(nSamples, 1. / nSamples, dtype=np.double)
matBar = barycenter(matList, weights)
meanBar = np.mean(meanList, axis=0)

matBarSqrt = scipy.linalg.sqrtm(matBar)
matBarSqrtInv = scipy.linalg.inv(matBarSqrt)

# calculate linear embedding of gaussian samples
linEmbMat = [LinW2LogMean(A, mA, matBar, meanBar, matBarSqrt, matBarSqrtInv) for A, mA in zip(matList, meanList)]
linEmbMat = np.array(linEmbMat)
linEmbMatMean = np.mean(linEmbMat, axis=0)
linEmbMat = linEmbMat - linEmbMatMean.reshape((1, -1))


## Additional sections, testing the quality of the gaussian approximation

##  load point cloud data
ptsData = [loadtxt(filenameRoot + "/pts_{:03d}".format(i)) for i in range(nSamples)]
ptsDataAll = np.vstack(ptsData)
nPtsAll = ptsDataAll.shape[0]
nPtsList = [x.shape[0] for x in ptsData]



##  Plot histograms of sample point clouds vs Gaussians

# #plot histograms of sample point clouds and corresponding Gaussian approximations along some 1d axis
# fig = plt.figure()
#
# # select axis
# v = np.array([1., 0., 0., 0., 0., 0.], dtype=np.double)
#
# xmin, xmax = [-5, 5]
# nBins = 100
# xspace = np.linspace(xmin, xmax, num=200)
#
# # select a some random samples
# for i, j in enumerate([1, 5, 8]):
#     # histogram data
#     pts = ptsData[j]
#     pts1d = np.einsum(v, [0], pts, [1, 0], [1])
#     frq, edges = np.histogram(pts1d, bins=nBins, range=[xmin, xmax])
#     # histogram plot
#     plt.step(edges, np.concatenate((np.array([0.]), frq)), c=colors[i])
#
#     # gaussian plot
#     mean1d = np.sum(v * meanList[j])
#     var1d = np.einsum(v, [0], v, [1], matList[j], [0, 1], [])
#
#     val = drawGaussian1D(mean1d, var1d, xspace) * pts.shape[0] * (xmax - xmin) / nBins
#
#     plt.plot(xspace, val, c=colors[i], ls="dashed")
#
# plt.tight_layout()
# plt.show()


## Compare reference measures: Gaussian and point cloud sampled from it

# fig = plt.figure()
#
# v = np.array([0., 0., 1., 0., 0., 0.], dtype=np.double)
#
# xmin, xmax = [-5, 5]
# nBins = 100
# xspace = np.linspace(xmin, xmax, num=50)
#
# # histogram data
# pts = ptsRef
# pts1d = np.einsum(v, [0], pts, [1, 0], [1])
# frq, edges = np.histogram(pts1d, bins=nBins, range=[xmin, xmax])
# # histogram plot
# plt.step(edges, np.concatenate((np.array([0.]), frq)), c=colors[0])
#
# # gaussian plot
# mean1d = np.sum(v * meanBar)
# var1d = np.einsum(v, [0], v, [1], matBar, [0, 1], [])
#
# val = drawGaussian1D(mean1d, var1d, xspace) * pts.shape[0] * (xmax - xmin) / nBins
#
# plt.plot(xspace, val, c=colors[0], ls="dashed")
#
# plt.tight_layout()
# plt.show()


## Compare sample point clouds with push-forward of ref point cloud

# # load a sample point cloud
# # and do push forward of ref point cloud by corresponding tangent vector
# # then compare 1d histograms of this
#
# fig = plt.figure()
#
# v = np.array([1., .5, 0., 0., 0., 0.], dtype=np.double)
#
# xmin, xmax = [-5, 5]
# nBins = 100
#
# j = 1
#
# # histogram of actual sample point cloud
# pts = ptsData[j]
# pts1d = np.einsum(v, [0], pts, [1, 0], [1])
# frq, edges = np.histogram(pts1d, bins=nBins, range=[xmin, xmax])
# # histogram plot
# plt.step(edges, np.concatenate((np.array([0.]), frq)), c=colors[0], label="sample point cloud")
#
# # take reference point cloud and apply transport map to it (should be close to previous)
# ptsRefPush = ptsRef + linEmb[j]
# pts1d = np.einsum(v, [0], ptsRefPush, [1, 0], [1])
# frq, edges = np.histogram(pts1d, bins=nBins, range=[xmin, xmax])
# # renormalize
# frq = frq / nPtsRef * pts.shape[0]
# # histogram plot
# plt.step(edges, np.concatenate((np.array([0.]), frq)), c=colors[1], label="push-fwd of ref cloud by ptcloud map")
#
# # take reference point cloud and apply transport map between Gaussian samples to it (should be close to below)
# # preprocess tangent mode, to extract transport map
# tan = linEmbMat[j] + linEmbMatMean
# S = tan[:nFeatures * nFeatures].reshape((nFeatures, nFeatures))
# S = matBarSqrtInv.dot(S) + np.identity(nFeatures)
# mS = tan[nFeatures * nFeatures:] + meanBar
# # apply transport map to ref points
# ptsRefPushLin = mS + np.einsum(S, [0, 1], ptsRef - meanBar, [2, 1], [2, 0])
# # histogram
# pts1d = np.einsum(v, [0], ptsRefPushLin, [1, 0], [1])
# frq, edges = np.histogram(pts1d, bins=nBins, range=[xmin, xmax])
# # renormalize
# frq = frq / nPtsRef * pts.shape[0]
# # histogram plot
# plt.step(edges, np.concatenate((np.array([0.]), frq)), c=colors[2], label="push-fwd of ref cloud by matrix map")
#
# # Gaussian approximation of that sample
# mean1d = np.sum(v * meanList[j])
# var1d = np.einsum(v, [0], v, [1], matList[j], [0, 1], [])
# val = drawGaussian1D(mean1d, var1d, xspace) * pts.shape[0] * (xmax - xmin) / nBins
# plt.plot(xspace, val, c=colors[3], label="Gaussian approx of sample")
#
# plt.legend(loc=1)
#
# plt.tight_layout()
# plt.show()


### Compare the distances to the reference measure

# distance of point cloud sample to reference point cloud
dRefLin = [np.einsum((linEmb[i]) ** 2, [0, 1], muRef, [0], []) for i in range(nSamples)]
# distance of Gaussian sample to reference Gaussian
dRefLinMat = np.sum(linEmbMat ** 2, axis=1)

# plot distances and compare results
fig = plt.figure()
plt.plot(dRefLin, label="Point clouds")
plt.plot(dRefLinMat, label="Gaussians")
plt.legend()
plt.tight_layout()
plt.show()

# the distances are clearly distinct, but they follow the same trends
# this probably explains, why the analysis of both looks so similar


## compute linearized Wasserstein distance matrices for point cloud and Gaussian approximation
dLin = np.zeros((nSamples, nSamples))
dLinMat = np.zeros((nSamples, nSamples))

for i in range(nSamples):
    for j in range(nSamples):
        if i != j:
            dLin[i, j] = np.einsum((linEmb[i] - linEmb[j]) ** 2, [0, 1], muRef, [0], []) ** 0.5
            dLinMat[i, j] = np.sum((linEmbMat[i] - linEmbMat[j]) ** 2) ** 0.5

## compare total norms of dLin,dLinMat and difference
for x in [dLin, dLinMat, dLin - dLinMat]:
    print(np.sum(x ** 2))


## own colormap for better contrast in W-distance charts
from mpl_toolkits.axes_grid1 import AxesGrid

def shiftedColorMap(cmap, start=0, midpoint=0.5, stop=1.0, name='shiftedcmap'):
    cdict = {
        'red': [],
        'green': [],
        'blue': [],
        'alpha': []
    }

    # regular index to compute the colors
    reg_index = np.linspace(start, stop, 257)

    # shifted index to match the data
    shift_index = np.hstack([
        np.linspace(0.0, midpoint, 128, endpoint=False),
        np.linspace(midpoint, 1.0, 129, endpoint=True)
    ])

    for ri, si in zip(reg_index, shift_index):
        r, g, b, a = cmap(ri)

        cdict['red'].append((si, r, r))
        cdict['green'].append((si, g, g))
        cdict['blue'].append((si, b, b))
        cdict['alpha'].append((si, a, a))

    newcmap = matplotlib.colors.LinearSegmentedColormap(name, cdict)

    return newcmap

blues = cm.get_cmap('Blues')
mycmap = shiftedColorMap(blues, start=0, midpoint=0.35, stop=1.0, name='shifted')


## plot Wasserstein distance chart
from mpl_toolkits.axes_grid1 import make_axes_locatable

plt.figure(figsize=(8, 8))
ax = plt.gca()

plt.title('Bures-Wasserstein Distance', fontsize=20)

# choose one
#im = ax.imshow(dLin, cmap=mycmap)     # pointclouds
im = ax.imshow(dLinMat, cmap=mycmap)   # Gaussians


plt.axvline(x=5.5, ls='--', c='black', lw=2)
plt.axhline(y=5.5, ls='--', c='black', lw=2)

locs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
lbl = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
plt.xticks(ticks=locs, labels=lbl, fontsize=15)
plt.yticks(ticks=locs, labels=lbl, fontsize=15)

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.09)

cb = plt.colorbar(im, cax=cax)
cb.ax.tick_params(size=4, width=2, labelsize=15)

ax.set_xlabel('Subject number', fontsize=18)
ax.set_ylabel('Subject number', fontsize=18)

ax.xaxis.set_tick_params(which='major', size=4, width=2, direction='out')
ax.yaxis.set_tick_params(which='major', size=4, width=2, direction='out')

os.makedirs(resultpath + 'W-full-charts', exist_ok=True)
plt.savefig(resultpath + 'W-full-charts/Wfull-Gaussians-Chart.svg', bbox_inches='tight', pad_inches=0.02, transparent = True)
plt.show()


## compute averaged distances in quadrants:
print('Intra Group CTRL:')
print(mean(dLinMat[0:6,0:6]))
print('Inter group:')
print(mean(dLinMat[0:6,6:12]))
print('Intra Group MS:')
print(mean(dLinMat[6:12,6:12]))



## PCA on linear embedding

# how many dimensions in tangent space do we keep?
keep = 3

# point cloud PCA
eigvalFull, eigvec = PCA(linEmbEucl, keep=None)
eigval = eigvalFull[:keep]
eigvec = eigvec[:keep]
# Gaussians PCA
eigvalFullMat, eigvecMat = PCA(linEmbMat, keep=None)
eigvalMat = eigvalFullMat[:keep]
eigvecMat = eigvecMat[:keep]

## fraction of captured variance
print("fraction of captured variance:")
print("point cloud:\t {:.3f}".format(np.sum(eigval) / np.sum(eigvalFull)))
print("Gaussians:\t {:.3f}".format(np.sum(eigvalMat) / np.sum(eigvalFullMat)))

fig = plt.figure(facecolor="w")
plt.title("PCA spectrum")
plt.plot(eigvalFull + 1E-100, marker="x", label="point cloud")
plt.plot(eigvalFullMat + 1E-100, marker="x", label="Gaussians")

plt.yscale("log")
plt.ylim([1E-4, 1E0])
plt.xlim([0, 11])
plt.legend()
plt.tight_layout()

plt.show()


##  project samples into eigenbasis
coords = np.einsum(eigvec, [0, 1], linEmbEucl, [2, 1], [2, 0])
coordsMat = np.einsum(eigvecMat, [0, 1], linEmbMat, [2, 1], [2, 0])

## compute mean values of both group in PCA eigenbasis :

mean_CTRL = []
mean_MS = []
mean_clouds_CTRL = []
mean_clouds_MS = []
for i in range(3):
    mean_CTRL.append(mean(coordsMat[0:6,i]))
    mean_MS.append(mean(coordsMat[6:12, i]))
    mean_clouds_CTRL.append(mean(coords[0:6,i]))
    mean_clouds_MS.append(mean(coords[6:12, i]))

## Gaussians
## plot samples in 3d eigenbasis of subject space

# select axis: 0,1,2
axsel = [1, 2]

fig = plt.figure( figsize = (6,6))
ax = plt.gca()

for i in range(2):
    sel = (classes == i)
    plt.scatter(coordsMat[sel, axsel[0]], coordsMat[sel, axsel[1]], c=colors[i], label=classlabel[i], s =75)
for j in range(nSamples):
    if j == 2:
        plt.annotate(j + 1, xy=(coordsMat[j, axsel[0]] + 0.05, coordsMat[j, axsel[1]] - 0.01), fontsize=13)
    else:
        plt.annotate(j+1, xy=(coordsMat[j, axsel[0]] +0.04, coordsMat[j, axsel[1]] - 0.09), fontsize = 13)

plt.title("Coordinates in PCA-eigenbasis", fontsize=16)
plt.xlabel('PCA-mode ' + str(axsel[0] + 1) , fontsize=16)
plt.ylabel('PCA-mode ' + str(axsel[1] + 1) , fontsize=16)
plt.xticks([-1,0,1] )
plt.yticks([-1,0,1])
plt.axis('scaled')
plt.xlim(-1.5,1.2)
plt.ylim(-1.5,1.2)
ax.xaxis.set_tick_params(which='major', size=8, width=2, direction='in')
ax.yaxis.set_tick_params(which='major', size=8, width=2, direction='in')
#plt.legend( fontsize=13, handletextpad = -0.2, edgecolor = 'black', frameon = True, fancybox = False, transparent = True)
plt.tight_layout()

os.makedirs(resultpath + 'PCA-and-SVM', exist_ok=True)
plt.savefig(resultpath + 'PCA-and-SVM/PCA-Gaussians-'+str(axsel[0]+1)+'-'+str(axsel[1]+1)+'.svg',  bbox_inches='tight', pad_inches=0.02, transparent = True)
plt.show()
##

axsel = [0, 2]
fig = plt.figure(figsize = (7,6))
ax = plt.gca()

for i in range(2):
    sel = (classes == i)
    plt.scatter(coordsMat[sel, axsel[0]], coordsMat[sel, axsel[1]], c=colors[i], label=classlabel[i], s =75)
for j in range(nSamples):
    plt.annotate(j+1, xy=(coordsMat[j, axsel[0]] +0.06, coordsMat[j, axsel[1]] - 0.11), fontsize = 13)

plt.title("Coordinates in PCA-eigenbasis", fontsize=16)
plt.xlabel('PCA-mode ' + str(axsel[0] + 1) , fontsize=16)
plt.ylabel('PCA-mode ' + str(axsel[1] + 1) , fontsize=16)
plt.yticks([-1,0,1])
plt.xlim(-2.5,2.4)
plt.ylim(-1.5,1.2)
ax.xaxis.set_tick_params(which='major', size=8, width=2, direction='in')
ax.yaxis.set_tick_params(which='major', size=8, width=2, direction='in')
plt.legend( bbox_to_anchor=(0.26,1), fontsize=13, handletextpad = -0.2, edgecolor = 'black', frameon = True, fancybox = False)

plt.savefig(resultpath + 'PCA-and-SVM/PCA-Gaussians-'+str(axsel[0]+1)+'-'+str(axsel[1]+1)+'.svg', bbox_inches='tight', pad_inches=0.02, transparent = True)
plt.show()

## the same for point clouds
##
axsel = [1, 2]
fig = plt.figure( figsize = (6,6))
ax = plt.gca()

for i in range(2):
    sel = (classes == i)
    plt.scatter(coords[sel, axsel[0]], coords[sel, axsel[1]], c=colors[i], label=classlabel[i], s =75)
for j in range(nSamples):
    if j == 2:
        plt.annotate(j + 1, xy=(coords[j, axsel[0]] + 0.05, coords[j, axsel[1]] - 0.01), fontsize=13)
    else:
        plt.annotate(j+1, xy=(coords[j, axsel[0]] +0.04, coords[j, axsel[1]] - 0.09), fontsize = 13)

plt.title("Coordinates in PCA-eigenbasis", fontsize=16)
plt.xlabel('PCA-mode ' + str(axsel[0] + 1) , fontsize=16)
plt.ylabel('PCA-mode ' + str(axsel[1] + 1) , fontsize=16)
plt.xticks([-1,0,1])
plt.yticks([-1,0,1])
plt.axis('scaled')
plt.xlim(-1.5,1.2)
plt.ylim(-1.5,1.2)
ax.xaxis.set_tick_params(which='major', size=8, width=2, direction='in')
ax.yaxis.set_tick_params(which='major', size=8, width=2, direction='in')
#plt.legend(fontsize=13, handletextpad = -0.2, edgecolor = 'black', frameon = True, fancybox = False)
plt.tight_layout()

plt.savefig(resultpath + 'PCA-and-SVM/PCA-Pointclouds-'+str(axsel[0]+1)+'-'+str(axsel[1]+1)+'.svg',  bbox_inches='tight', pad_inches=0.02, transparent = True)
plt.show()
##

axsel = [0, 2]
fig = plt.figure(figsize = (7,6))
ax = plt.gca()

for i in range(2):
    sel = (classes == i)
    plt.scatter(coords[sel, axsel[0]], coords[sel, axsel[1]], c=colors[i], label=classlabel[i], s =75)
for j in range(nSamples):
    plt.annotate(j+1, xy=(coords[j, axsel[0]] +0.06, coords[j, axsel[1]] - 0.11), fontsize = 13)

plt.title("Coordinates in PCA-eigenbasis", fontsize=16)
plt.xlabel('PCA-mode ' + str(axsel[0] + 1) , fontsize=16)
plt.ylabel('PCA-mode ' + str(axsel[1] + 1) , fontsize=16)
plt.yticks([-1,0,1])
plt.xlim(-2.5,2.4)
plt.ylim(-1.5,1.2)
ax.xaxis.set_tick_params(which='major', size=8, width=2, direction='in')
ax.xaxis.set_tick_params(which='minor', size=4, width=2, direction='in', top='on')
ax.yaxis.set_tick_params(which='major', size=8, width=2, direction='in')
plt.legend( bbox_to_anchor=(0.26,1), fontsize=13, handletextpad = -0.2, edgecolor = 'black', frameon = True, fancybox = False)

plt.savefig(resultpath + 'PCA-and-SVM/PCA-Pointclouds-'+str(axsel[0]+1)+'-'+str(axsel[1]+1)+'.svg', bbox_inches='tight', pad_inches=0.02, transparent = True)
plt.show()




##  Video of subject coordinates in 3d PCA-eigenbasis (rotating animation)

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation

# plot coordinates
fig = plt.figure()
ax = fig.add_subplot(1,1,1, projection='3d')
ax.scatter(coordsMat[0:6, 0], coordsMat[0:6, 1], coordsMat[0:6, 2], s=25., color = colors[0])
ax.scatter(coordsMat[6:12, 0], coordsMat[6:12, 1], coordsMat[6:12, 2], s=25., color = colors[1])
for j in range(nSamples):
    ax.text(coordsMat[j, 0], coordsMat[j, 1], coordsMat[j, 2], j+1, fontsize = 12, color='black')
ax.set_yticks([-1,0,1])
ax.set_zticks([-1,0,1])
ax.set_title('3d PCA Eigenbasis - Subject Space', fontsize = 18)
ax.set_xlabel('PCA-mode 1', labelpad= 10)
ax.set_ylabel('PCA-mode 2', labelpad= 10)
ax.set_zlabel('PCA-mode 3', labelpad= 10)
plt.show()

def rotate(angle):
    ax.view_init(azim=angle,)

rot_animation = animation.FuncAnimation(fig, rotate, frames=np.arange(0,362,2),interval=100)
writervideo = animation.FFMpegWriter(fps=20)
#rot_animation.save(resultpath + '3d-Subject-space.mp4', dpi=300, writer=writervideo)



## Simple classification approach via Support vector machine (SVM)
# apply basic linear SVM to samples in PCA space
# only on the reduced dimensions, specified by "keep" above

from sklearn.svm import LinearSVC

clf = LinearSVC(loss="squared_hinge", C = 5)
clf.fit(coords, classes);
vecSVM = clf.coef_[0]
vecSVMNorm = np.linalg.norm(vecSVM)
vecSVM = vecSVM / vecSVMNorm
print("normalized separating hyperplane normal:")
print(vecSVM)

## plot distances from samples to hyperplane

coordsSVM = np.einsum(vecSVM, [1], coords, [0, 1], [0])
fig = plt.figure(facecolor="w", figsize = (7,4.8))
xcoords = np.arange(nSamples)
for i in range(2):
    sel = (classes == i)
    plt.scatter(xcoords[sel], coordsSVM[sel] + clf.intercept_ / vecSVMNorm, \
                marker="o", label=classlabel[i], c=colors[i])
plt.plot(np.zeros_like(coordsSVM), c = 'grey')

plt.legend(loc = 2, fontsize=13, handletextpad = -0.2, edgecolor = 'black', frameon = True, fancybox = False)
plt.title("SVM Classification in PCA-basis", fontsize=16)
plt.xticks(ticks = np.arange(nSamples, dtype=int), labels = np.arange(nSamples, dtype=int) + 1,fontsize = 13)
plt.yticks(fontsize = 13)
plt.xlabel("Subject number", fontsize=16)
plt.ylabel("SVM decision function value" , fontsize=16)
plt.tight_layout()

ax = plt.gca()
ax.xaxis.set_tick_params(which='major', size=4, width=2, direction='out')
ax.xaxis.set_tick_params(which='minor', size=4, width=2, direction='in', top='on')
ax.yaxis.set_tick_params(which='major', size=4, width=2, direction='out')


plt.savefig(resultpath + 'PCA-and-SVM/SVM-PointClouds.svg', bbox_inches='tight',pad_inches = 0.02, transparent = True )
plt.show()



## SVM on Gaussians

clfMat = LinearSVC(loss="squared_hinge", C= 5)
clfMat.fit(coordsMat, classes);
vecSVMMat = clfMat.coef_[0]
vecSVMMatNorm = np.linalg.norm(vecSVMMat)
vecSVMMat = vecSVMMat / vecSVMMatNorm
print("normalized separating hyperplane normal:")
print(vecSVMMat)

## plot distances to hyperplane

coordsSVMMat = np.einsum(vecSVMMat, [1], coordsMat, [0, 1], [0])
fig = plt.figure(facecolor="w", figsize = (7,4.8))
xcoords = np.arange(nSamples)
for i in range(2):
    sel = (classes == i)
    plt.scatter(xcoords[sel], coordsSVMMat[sel] + clfMat.intercept_ / vecSVMMatNorm, \
                marker="o", label=classlabel[i], c=colors[i])
plt.plot(np.zeros_like(coordsSVMMat), c = 'grey')

plt.legend(loc = 2, fontsize=13, handletextpad = -0.2, edgecolor = 'black', frameon = True, fancybox = False)
plt.title("SVM Classification in PCA-basis", fontsize=16)
plt.xticks(ticks = np.arange(nSamples, dtype=int), labels = np.arange(nSamples, dtype=int) + 1
           ,fontsize = 13)
plt.yticks(fontsize = 13)
plt.xlabel("Subject number", fontsize=16)
plt.ylabel("SVM decision function value" , fontsize=16)
plt.tight_layout()

ax = plt.gca()
ax.xaxis.set_tick_params(which='major', size=4, width=2, direction='out')
ax.yaxis.set_tick_params(which='major', size=4, width=2, direction='out')

plt.savefig(resultpath + 'PCA-and-SVM/SVM-Gaussians.svg', bbox_inches='tight',pad_inches = 0.02, transparent = True)
plt.show()



## Backprojection from PCA eigenbasis in subject space to tangent space

# Shoot along selected mode and visualize evolution of histograms

# unit vectors in feature space
vList = [
    np.array([1., 0., 0., 0., 0., 0.]),
    np.array([0., 1., 0., 0., 0., 0.]),
    np.array([0., 0., 1., 0., 0., 0.]),
    np.array([0., 0., 0., 1., 0., 0.]),
    np.array([0., 0., 0., 0., 1., 0.]),
    np.array([0., 0., 0., 0., 0., 1.]),
]

# histogram settings
xmin, xmax = [-4, 4]
nBins = 80
xspace = np.linspace(xmin, xmax, num=100)

##
numS = 2
sList = np.linspace(-0.5, 0.5, num=numS)
lList = ["– –", "++"]


## coordinate of SVM mode in Euclideanized-centrized tangent space

def point_to_tanspace(h):
    # backprojection from PCA eigenbasis to highdimensional subject space
    tan = np.einsum(h, [0], eigvec, [0, 1], [1])
    # mode as relative transport map  (subject space -> tangent space)
    tan = np.einsum(tan.reshape((-1, nFeatures)), [0, 1], 1 / np.sqrt(muRef), [0], [0, 1])
    return tan

def point_to_tanspace_gauss(h):
    return np.einsum(h, [0], eigvecMat, [0, 1], [1])

##
tan_groupmeans_clouds = [point_to_tanspace(mean_clouds_CTRL), point_to_tanspace(mean_clouds_MS)]
tan_groupmeans = [point_to_tanspace_gauss(mean_CTRL), point_to_tanspace_gauss(mean_MS)]

## Pushforward of reference sample to the groupmeans: Gaussians

fig = plt.figure(facecolor="w", figsize=(8, 10))

ax = fig.add_subplot(111)
#ax.set_ylabel('Frequency', fontsize=17)
#ax.set_xlabel('Feature Space', fontsize=17)
ax.set_title('Push forward to group means ', fontsize=26, pad=45)

# Turn off axis lines and ticks of the big subplot
ax.spines['top'].set_color('none')
ax.spines['bottom'].set_color('none')
ax.spines['left'].set_color('none')
ax.spines['right'].set_color('none')
ax.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)

for iFeat, v in enumerate(vList):
    fig.add_subplot(3, 2, iFeat + 1)
    for i, tan_mean in enumerate(tan_groupmeans):
        # reference measure push forward
        S, mS = LinW2ExpMean(linEmbMatMean + tan_mean, matBar, meanBar, matBarSqrt, matBarSqrtInv)
        # gaussian plot
        mean1d = np.sum(v * mS)
        var1d = np.einsum(v, [0], v, [1], S, [0, 1], [])
        val = drawGaussian1D(mean1d, var1d, xspace)

        plt.plot(xspace, val, c=cm.bwr(i / (numS - 1)), label=lList[i], lw = 2.5)
        plt.xticks([-3, 0, 3])
        plt.xlim(-4,4)

        ax = plt.gca()
        ax.xaxis.set_tick_params(which='major', size=4, width=2, direction='out')
        ax.yaxis.set_tick_params(which='major', size=4, width=2, direction='out')
        if iFeat == 0:
            plt.legend(fontsize=12, handletextpad = 0.5, edgecolor = 'black', frameon = True, fancybox = False)
        if iFeat == 1:
            plt.yticks([0, 0.3, 0.6])
        else:
            plt.yticks([0, 0.2, 0.4])

    plt.title(featureNames[iFeat] )

plt.tight_layout()
plt.subplots_adjust( hspace = 0.45)

os.makedirs(resultpath + 'Pushforwards', exist_ok=True)
plt.savefig(resultpath + 'Pushforwards/Pushforward-GroupMeans-Gauss.svg' , bbox_inches='tight',pad_inches = 0.02, transparent = True)
plt.show()




## Pushforward of reference sample to both groupmeans: point clouds

fig = plt.figure(facecolor="w", figsize=(8, 10))

ax = fig.add_subplot(111)
#ax.set_ylabel('Frequency', fontsize=17)
#ax.set_xlabel('Feature Space', fontsize=17)
ax.set_title('Push forward to group means ', fontsize=26, pad=45)
# Turn off axis lines and ticks of the big subplot
ax.spines['top'].set_color('none')
ax.spines['bottom'].set_color('none')
ax.spines['left'].set_color('none')
ax.spines['right'].set_color('none')
ax.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)

for iFeat, v in enumerate(vList):
    fig.add_subplot(3, 2, iFeat + 1)
    for i, tan_mean in enumerate(tan_groupmeans_clouds):
        # reference measure push forward
        ptsRefPush = ptsRef + linEmbMean + tan_mean
        pts1d = np.einsum(v, [0], ptsRefPush, [1, 0], [1])
        frq, edges = np.histogram(pts1d, bins=nBins, range=[xmin, xmax])
        # renormalize
        frq = frq / (nPtsRef * (xmax - xmin) / nBins)
        # histogram plot
        plt.step(edges, np.concatenate((np.array([0.]), frq)), c=cm.bwr(i / (numS - 1)), label=lList[i], lw = 2.5)
    plt.title(featureNames[iFeat])
    plt.xticks([-3, 0, 3])
    plt.xlim(-4,4)
    ax = plt.gca()
    ax.xaxis.set_tick_params(which='major', size=4, width=2, direction='out')
    ax.yaxis.set_tick_params(which='major', size=4, width=2, direction='out')
    if iFeat == 0:
        plt.legend(fontsize=12, handletextpad = 0.5, edgecolor = 'black', frameon = True, fancybox = False)
    if iFeat == 1:
        plt.yticks([0, 0.3, 0.6])
    else:
        plt.yticks([0, 0.2, 0.4])

plt.tight_layout()
plt.subplots_adjust(hspace = 0.45)

plt.savefig(resultpath + 'Pushforwards/Pushforward-GroupMeans-PointClouds.svg' , bbox_inches='tight',pad_inches = 0.02, transparent = True)
plt.show()



##

