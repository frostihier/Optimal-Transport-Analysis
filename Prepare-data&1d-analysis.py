import numpy as np
from numpy import mean, std, median, amax, pi, exp, append, argmax, save, savetxt, loadtxt
from numpy import array, column_stack, zeros, sin, cos, sqrt, load, ones, concatenate
import ot
import matplotlib.pyplot as plt
from numpy.random import multivariate_normal
from scipy.spatial import cKDTree
from time import time
import os

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


### Overview over Feature-Columns

# 0,1,2: X, Center of Geometry (µm)	Y, Center of Geometry (µm)	Z, Center of Geometry (µm)
# 3: Mean, Intensities
# 4: Standard Dev., Intensities
# 5: Volume, Volume (µm³)
# 6: Sphericity
# 7: Number of neighbors
# 8: Distance to nearest neighbor

## define datapaths
txt_data = 'Data/txt-data/'
fileroot = 'Data/Data-for-TransportAnalysis/'
resultpath = 'Results/'

if not os.path.isdir(resultpath):
    os.makedirs(resultpath)

##
# load in txt tables. The columns contain the features, rows correspond to single cells
# every file is one patient

#Group1 - CTRL
data1 = loadtxt(txt_data + '6716.txt', skiprows = 1)
data2 = loadtxt(txt_data + '6916.txt', skiprows = 1)
data3 = loadtxt(txt_data + '2616.txt', skiprows = 1)
data4 = loadtxt(txt_data + '5416.txt', skiprows = 1)
data5 = loadtxt(txt_data + '12816.txt', skiprows = 1)
data6 = loadtxt(txt_data + '6515.txt', skiprows = 1)

#Group2 - MS
data7 = loadtxt(txt_data + '12006.txt', skiprows = 1)
data8 = loadtxt(txt_data + 'T800.txt', skiprows = 1)
data9 = loadtxt(txt_data + '11313.txt', skiprows = 1)
data10 = loadtxt(txt_data + '13609.txt', skiprows = 1)
data11 = loadtxt(txt_data + '22091.txt', skiprows = 1)
data12 = loadtxt(txt_data + '3614.txt', skiprows = 1)

data = [data1, data2, data3, data4, data5, data6, data7, data8, data9, data10, data11, data12]

##
labels = ['X-Position', 'Y-Position', 'Z-Position', 'Volume', 'Electron density', 'Heterogeneity', 'Sphericity', 'Num-Neighbors', 'NN-Distance']
sampleTags = ['6716', '6916', '2616', '5416', '12816', '6515', '12006', 'T800', '11313', '13609', '22091', '3614']
classes = [0,0,0,0,0,0,1,1,1,1,1,1]

## list of features used in OT analysis
list_f = [3,4,5,6,7,8]

## determine two features out of the nuclei positions:
# number of nearest neighbors within a radius (7.45µm) and distances to nearest neighbor

posdata = []
for i in range(len(data)):
    posdata.append(data[i][:,0:3])

# done with cKDtree
def nearest_neighbors_stats(x, r):
    # creates the tree
    tree = cKDTree(x)

    # number of neighbors in shell
    # iterates over all lists of pairs and returns the length of this list
    distances = tree.query_ball_point(x, r)
    d_counts = array([len(d) - 1 for d in distances])

    # calculates distance to nearest neighbor
    nn, ii = tree.query(x, k=[2])

    return d_counts, nn

# calculate number of neighbors in shell NN
for d, i in zip(posdata, range(len(posdata))):
    d_counts, nn = nearest_neighbors_stats(d, 7.45)    # value of 7.45 microns determined by first local minimum of radial distributions function

    # appending both features to data
    data[i] = column_stack((data[i], d_counts))
    data[i] = column_stack((data[i], nn))





## perform t-test for all features, print p values and medians of unstandartisized features for both groups in console
from scipy.stats import ttest_ind

# Note: since the feature number of neighbors always takes on integer values,
# the mean of the population is more expressive than the median of it.

for s in list_f:

    medians = []
    means = []

    for d in data:
        medians.append(median(d[:, s]))
        means.append(mean(d[:, s]))

    # apply Welch's two-sided t-test
    if s == 7:
        res = ttest_ind(means[0:6], means[6:12], equal_var=False)
    else:
        res = ttest_ind(medians[0:6], medians[6:12], equal_var=False)

    # print p-values
    print('p-Value of ' + labels[s] + ':', res[1])

   # print population medians and standard-deviation
    if s == 7:
        print('--ALL--')
        print(mean(means[0:12]))
        print(std(means[0:12]))

        print('--CTRL--')
        print(mean(means[0:6]))
        print(std(means[0:6]))

        print('--MS--')
        print(mean(means[6:12]))
        print(std(means[6:12]))

        print('___________')

    else:
        print('--ALL--')
        print(mean(medians[0:12]))
        print(std(medians[0:12]))

        print('--CTRL--')
        print(mean(medians[0:6]))
        print(std(medians[0:6]))

        print('--MS--')
        print(mean(medians[6:12]))
        print(std(medians[6:12]))

        print('___________')



## Standartisation with z-score

from scipy.stats import zmap


# Index of features to standartisize:
for i in list_f:

    #creates the total population (all nuclei of all patients) for each feature i
    ges = []
    for d in data:
        ges = concatenate((ges,d[:,i]))

    #standartisize the patient-population with total population
    for d in data:
        d[:,i] = zmap(d[:,i], ges)


## plotting histograms of single features from one patient exemplarily


for s in list_f:
    plt.figure()
    plt.grid(zorder=0)
    plt.hist(data[0][:,s], bins = 70, density=True, zorder =3, ec = 'black', fc = 'blue')
    plt.xlabel(labels[s], fontsize = 20)
    plt.ylabel('Frequency',fontsize = 20 )
    plt.xticks(fontsize = 15)
    plt.yticks(fontsize = 15)
    plt.xlim(-2.7,5.2)

    ax = plt.gca()
    ax.xaxis.set_tick_params(which='major', size=6, width=2, direction='out')
    ax.yaxis.set_tick_params(which='major', size=6, width=2, direction='out')

    os.makedirs(resultpath + 'Histograms', exist_ok = True)
    plt.savefig(resultpath + 'Histograms/' + labels[s] + '.svg', bbox_inches='tight',pad_inches = 0.02, transparent = True)
    plt.show()




## draw point cloud and ellipses in 2d projection of feature space
fig = plt.figure(2)

# select features for 2d plane
i = 3
j = 6

for x in range(1):

    plt.title('Feature Space', fontsize=20)
    plt.xlabel(labels[i], fontsize=20)
    plt.ylabel(labels[j], fontsize=20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)

    #for i=3, j=6
    plt.xlim(-1.8, 3.6)
    plt.ylim(-2.5, 2.1)

    #for i=4-, j=5
    #plt.xlim(-2.6, 1.4)
    #plt.ylim(-3.2, 3.3)
    #plt.yticks([-2,0,2])

    ax = plt.gca()
    ax.xaxis.set_tick_params(which='major', size=6, width=2, direction='in')
    ax.yaxis.set_tick_params(which='major', size=6, width=2, direction='in')
    colors = ['blue', 'cyan', 'dodgerblue', 'royalblue', 'slateblue', 'lightskyblue',
                        'red', 'tomato', 'crimson', 'darkorange', 'darkred', 'firebrick']

    # pick patient
    k = 0

    # draw ellipse by computing empirical covariance and solving eigen-problem
    cov = np.cov(data[k].T)
    # pick covariance entries of 2d Subspace
    cov2x2 = np.reshape([cov[i,i], cov[i,j], cov[j,i], cov[j,j]], (2,2))

    eigenvalues, eigenvectors = np.linalg.eig(cov2x2)

    theta = np.linspace(0, 2*pi, 1000)
    ellipsis = (np.sqrt(eigenvalues[None,:]) * eigenvectors) @ [np.sin(theta), np.cos(theta)]

    # draw point cloud
    plt.plot(data[k][:,i], data[k][:,j] , '.', ms=0.22,  c=colors[0])
    # draw ellipse
    plt.plot(ellipsis[0, :] + mean(data[k][:, i]), ellipsis[1, :] + mean(data[k][:, j]), label=k, c=colors[0], lw = 2.5)

    os.makedirs(resultpath + 'Feature Space', exist_ok=True)
    plt.savefig(resultpath + 'Feature Space/FeatureSpace-'+labels[i]+'-'+labels[j]+'.png', bbox_inches='tight', pad_inches=0.02, transparent=True, dpi=300)
    plt.show()



## draw ellipses - as representation of the gaussian approximation - of all patient in 2d subspace

plt.figure(figsize=(7,6))

# select features for 2d plane
i = 3
j = 5

for x in range(1):

    plt.title('Feature Space', fontsize=16)
    plt.xlabel(labels[i], fontsize=16)
    plt.ylabel(labels[j], fontsize=16)

    #for i=4, j=5
    #plt.xlim(-2.4, 2.8)
    #plt.ylim(-2.6, 2.6)

    #for i=3, j=5
    plt.xlim(-1.8, 3.5)
    plt.ylim(-2.6, 2.6)
    #plt.yticks([-2,0,2])

    ax = plt.gca()
    ax.xaxis.set_tick_params(which='major', size=6, width=2, direction='in')
    ax.yaxis.set_tick_params(which='major', size=6, width=2, direction='in')
    ax.grid(ls='--')
    colors = ['blue', 'cyan', 'dodgerblue', 'royalblue', 'slateblue', 'lightskyblue',
                        'red', 'tomato', 'crimson', 'darkorange', 'darkred', 'firebrick']

    # for all patient draw an ellipse
    for k in range(0,12):

        # compute empirical covariance matrix
        cov = np.cov(data[k].T)

        # pick covariance entries of 2d Subspace
        cov2x2 = np.reshape([cov[i,i], cov[i,j], cov[j,i], cov[j,j]], (2,2))

        eigenvalues, eigenvectors = np.linalg.eig(cov2x2)

        theta = np.linspace(0, 2*pi, 1000)
        ellipsis = (np.sqrt(eigenvalues[None,:]) * eigenvectors) @ [np.sin(theta), np.cos(theta)]

        plt.plot(ellipsis[0,:] + mean(data[k][:,i]), ellipsis[1,:] + mean(data[k][:,j]), label = k+1, c=colors[k], lw = 2.5 )


    plt.legend(bbox_to_anchor=(1,1), loc=1, frameon=False, fontsize=12, handletextpad = 0.4)

    os.makedirs(resultpath + 'Ellipses', exist_ok=True)
    plt.savefig(resultpath + 'Ellipses/Ellipses-'+labels[i]+'-'+labels[j]+'.svg', bbox_inches='tight', pad_inches=0.02, transparent=True, dpi=400)
    plt.show()



##  plot median values

fig = plt.figure(figsize = (12,7))

X1 = ones(6) * 0.5
X2 = ones(6) * 1

ax = fig.add_subplot(111)
ax.set_ylabel('Median over population ', fontsize=26)
# Turn off axis lines and ticks of the big subplot
ax.spines['top'].set_color('none')
ax.spines['bottom'].set_color('none')
ax.spines['left'].set_color('none')
ax.spines['right'].set_color('none')
ax.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)


# for each of the six feature
for k, s in enumerate(list_f):

    # plot median over population from each patients
    medians = []
    means = []              # only for features NN -> mean instead of median

    for d in data:
        medians.append(median(d[:,s]))
        means.append(mean(d[:,s]))

    fig.add_subplot(2, 3, k + 1)
    if k == 4:
        plt.plot(X1, means[0:6], '_', ms=30, lw=100, c='blue', label='CTRL')
        plt.plot(X2, means[6:12], '_', ms=30, c='red', label='MS')
        plt.plot(X1[0], mean(means[0:6]), '_', ms=50, c='black')
        plt.plot(X2[0], mean(means[6:12]), '_', ms=50, c='black')
    else:
        plt.plot(X1, medians[0:6], '_', ms=30, lw= 100, c = 'blue', label='CTRL')
        plt.plot(X2, medians[6:12], '_', ms=30, c='red', label='MS')
        plt.plot(X1[0], mean(medians[0:6]), '_', ms=50, c = 'black' )
        plt.plot(X2[0], mean(medians[6:12]), '_', ms=50, c = 'black' )
    plt.axvline(x=0.5, ls='--', c='grey')
    plt.axvline(x=1, ls='--', c='grey')
    ax = plt.gca()
    ax.xaxis.set_tick_params(which='major', size=6, width=2, direction='in')
    ax.yaxis.set_tick_params(which='major', size=6, width=2, direction='in')
    plt.title(labels[s])
    plt.xlim(0,1.5)
    plt.xticks(ticks = [0.5,1], labels = ['CTRL', 'MS'])

fig.tight_layout()

os.makedirs(resultpath + 'Median-Values', exist_ok=True)
plt.savefig(resultpath + 'Median-Values/Median-Values.svg',  bbox_inches='tight', pad_inches=0.02, transparent = True)
plt.show()


## create Violinplots

# for each feature:
for j,s in enumerate(list_f):
    v = []
    meds = []
    cb = []
    ct = []
    q1 = []
    q3 = []

    # since data are very scattered, an IQR filter is applied which cuts off the outermost 5% of the edges
    # from filterd data first and third quantile is calculated, shown as black bars in the plot
    for d in data:
        cut_bottom, med, cut_top = np.percentile(d[:, s], [5, 50, 95])

        # filter data
        filter = d[:,s].copy()
        filter = filter[filter > cut_bottom]
        filter = filter[filter < cut_top]

        #filtered data and mix/max of it
        v.append(filter)
        cb.append(cut_bottom)
        ct.append(cut_top)
        #save medians
        meds.append(med)
        # from filtered data compute first and third quantile
        q1.append(np.percentile(filter, [25]))
        q3.append(np.percentile(filter, [75]))


    plt.figure(figsize=(12, 5))

    ax = plt.gca()
    ax.set_axisbelow(True)
    ax.grid(ls='--')
    vplots = plt.violinplot(v, showmedians=False, showextrema=False, points = 200)

    inds = np.arange(1, len(meds) + 1)
    # plot medians as white dots
    plt.scatter(inds, meds, marker='o', color='white', s=30, zorder=3)
    plt.vlines(inds, array(cb), array(ct), color='Black', linestyle='-', lw=1)
    #plot quantiles
    plt.vlines(inds, array(q1), array(q3), color='Black', linestyle='-', lw=5)

    plt.xticks(ticks = inds, labels = inds, fontsize = 15 )
    plt.yticks(fontsize=15)
    plt.xlabel('Subject Number', fontsize = 26)
    plt.ylabel(labels[s], fontsize = 26)

    # set colors
    for pc,i in zip(vplots['bodies'],range(12)):
        if i < 6:
            pc.set_facecolor('blue')
            pc.set_edgecolor('Darkblue')
            pc.set_alpha(0.75)
        else:
            pc.set_facecolor('red')
            pc.set_edgecolor('Darkred')
            pc.set_alpha(0.75)

    # customize legend
    import matplotlib.patches as mpatches
    red_patch = mpatches.Patch(color='red')
    blue_patch = mpatches.Patch(color='blue')
    l = ['Control', 'MS']
    legend_pos = [2,2,2,3,2,1]
    plt.legend(loc=legend_pos[j], handles=[blue_patch,red_patch], labels=l, fontsize=13, handletextpad=0.35, edgecolor='black', frameon=True, fancybox=False)

    os.makedirs(resultpath + 'Violin-Plots', exist_ok=True)
    plt.savefig(resultpath + 'Violin-Plots/' + labels[s] + '-Violinplot.svg', bbox_inches='tight', pad_inches=0.02, transparent = True)
    plt.show()


## creating a new variable which contains only feature columns used for OT-analysis
# name it cdata

cdata = []
for i in range(len(data)):
    chosen_data = data[i][:,[3,4,5,6,7,8]]
    cdata.append(chosen_data)

    # save chosen data
    savetxt(fileroot + 'pts_{:03d}'.format(i), chosen_data)

print(cdata[0])

# Features Columns in 'cdata'
# 0 = Volume
# 1 = Electron Density
# 2 = Heterogenity
# 3 = Sphericity
# 4 = Number of neighbors
# 5 = Distance to nearest neighbors



## OT ANALYSIS 1D FEATURE HISTOGRAMS

## Enter number of samples and number of features here.
nSamples = 12
nFeat = 6


#### Wassserstein Distance chart: feature histograms

# calculate pairwise W2 distance between feature histograms of patients
# function gets 1d feature histograms of samples as input and return matrix containg the distances
def W2_distance_charts(data, f):

    l = len(data)
    chart = zeros((l,l))

    for i in range(l):
        for j in range(l):

            x = cdata[i][:,f]
            y = cdata[j][:,f]

            dist = sqrt(ot.emd2_1d(x, y))   # Wasserstein-2 distance

            chart[i,j] = dist

    return chart


## compute charts for each feature separately and store it
feature_charts = []

for i in range(nFeat):
    print(i)
    chart = W2_distance_charts(cdata, i)
    feature_charts.append(chart)


## create own colorbar out of the 'Blues' cmap, just for better contrast in charts

from matplotlib.colors import LinearSegmentedColormap
from matplotlib import cm

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

    newcmap = LinearSegmentedColormap(name, cdict)

    return newcmap

blues = cm.get_cmap('Blues')
mycmap = shiftedColorMap(blues, start=0, midpoint=0.35, stop=1.0, name='shifted')



## plot W-matrix-charts

from mpl_toolkits.axes_grid1 import make_axes_locatable



for i,j in zip(list_f,range(6)):

    plt.figure(figsize=(8,8))
    plt.title('Wasserstein Distance - ' + labels[i], fontsize=20)

    ax = plt.gca()
    im = ax.imshow(feature_charts[j], cmap=mycmap)    # own cmap
    plt.axvline(x=5.5, ls='--', c='black', lw=2)
    plt.axhline(y=5.5, ls='--', c='black', lw=2)
    locs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    lbl = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    plt.xticks(ticks=locs, labels=lbl, fontsize=15)
    plt.yticks(ticks=locs, labels=lbl, fontsize=15)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.09)

    if j == 1:
        cb = plt.colorbar(im, cax=cax, ticks=[0.0,1.0,2.0,3.0])
    else:
        cb = plt.colorbar(im, cax=cax, ticks=[0,0.5,1,1.5,2])
    cb.ax.tick_params(size=4, width=2, labelsize=15)
    ax.xaxis.set_tick_params(which='major', size=4, width=2, direction='out')
    ax.yaxis.set_tick_params(which='major', size=4, width=2, direction='out')

    os.makedirs(resultpath + 'W-feature-charts', exist_ok=True)
    plt.savefig(resultpath + 'W-feature-charts/W-' + labels[i] + '-chart.svg', bbox_inches='tight', pad_inches=0.02, transparent = True)
    plt.show()


## PREPARATION FOR MULTIDIMENSIONAL OT-ANALYSIS

## import OT library written by Bernhard Schmitzer (unpublished)
import sys
sys.path.append("../../../python")
from lib.misc import *
import lib.SinkhornNP as Sinkhorn
import lib.LinW2 as LinW2


## for gaussian approxiamtion of point clouds: compute empirical means and cov matrices
# since analysis requires only cov and mean, no actual pointcloud is sampled from that

covMatrices = zeros((nSamples, nFeat, nFeat))
means = zeros((nSamples, nFeat))

for i in range(nSamples):
    covMatrices[i, :, :] = np.cov(cdata[i], rowvar=False, bias=True)

    # Subjects in columns und features in rows
    for j in range(nFeat):
        means[i, j] = mean(cdata[i][:, j])


## save covs, means, sample tags and classlabels
save(fileroot + 'matList', covMatrices)
savetxt(fileroot + 'meanList' , means)

sampleTags = ['6716', '6916', '2616', '5416', '12816', '6515', '12006', 'T800', '11313', '13609', '22091', '3614']  #Subject numbers
savetxt(fileroot + 'sampleTags', sampleTags, fmt = '%s')

classes = [0,0,0,0,0,0,1,1,1,1,1,1]
savetxt(fileroot + 'classes' , classes)


## compute Wasserstein barycenter, sample from that and also store it

weights=np.full(nSamples,1./nSamples,dtype=np.double)

# compute barycenter with 'fixed point algorithm' (Alvarez-Esteban, 2016)
matBar = barycenter(covMatrices,weights)
meanBar = np.mean(means,axis=0)

# sample from center (only for point clouds analysis needed )
rng = np.random.default_rng()
nPts = 10000
ref_sample = sampleFromGaussian(matBar,meanBar,nPts,rng)

save(fileroot + 'pts_ref', ref_sample)


## put this file manually in folder: contains chosen paramters for entropic regularization
with open("Data/params.txt", "r") as f:
    params = json.load(f)
    f.close()

nSamples = params["nSamples"]
epsTarget = params["epsTarget"]
epsInit = params["epsInit"]
SinkhornErrorGoal = params["SinkhornErrorGoal"]
keep = params["keep"]
epsStep = .667
verbose = True


##
# number of particles in barycenter
num_ref_sample = ref_sample.shape[0]
# weights for particles of barycenter (here uniform)
mu_ref_sample = np.full(num_ref_sample,1./num_ref_sample)


## solve transport problems from reference sample (barycenter) to samples
# returns the optimal couplings pi

# Computation time: about 10 hours, ca 40 min for one sample

start = time()

for i in range(len(cdata)):
    print(i)

    ncdata = cdata[i].shape[0]
    mucdata = np.full(ncdata, 1. / ncdata)

    cost = getCostEuclidean(ref_sample, cdata[i])

    # Solve transport problem with entropic regularization
    solverDat = Sinkhorn.SolveW2(mu_ref_sample, ref_sample, mucdata, cdata[i], SinkhornErrorGoal, epsTarget, epsInit, returnSolver=True,
                             epsStep=epsStep, verbose=verbose)
    solver = solverDat[2]

    save(fileroot + 'sol_{:03d}_alpha'.format(i), solver.alpha)
    save(fileroot + 'sol_{:03d}_beta'.format(i), solver.beta)

    print('Nummer ' + str(i+1) + ' fertig')
    print('____________________________')

end = time()
print('Rechenzeit in sek: ',end - start)


## Linear Embedding

# create a Monge map from optimal couplings by averaging over the mass transport.
# Monge map contains tangent vectors which are simultaneously the linear embedded samples.

# Computation time: 5 min

tanList=[]

for i in range(0,len(cdata)):

    print(i+1)

    ncdata = cdata[i].shape[0]
    mucdata = np.full(ncdata, 1. / ncdata)
    cost = getCostEuclidean(ref_sample, cdata[i])

    alpha = load(fileroot + 'sol_{:03d}_alpha'.format(i)+'.npy')
    beta = load(fileroot + 'sol_{:03d}_beta'.format(i)+'.npy')

    pi = np.einsum(
        np.exp((-cost + alpha.reshape((-1, 1)) + beta.reshape((1, -1))) / epsTarget), [0, 1],
        mu_ref_sample, [0], mucdata, [1], [0, 1])
    pi = scipy.sparse.csr_matrix(pi)

    T = LinW2.extractMongeData(pi, cdata[i])
    tanList.append(T - ref_sample)

## save linear embedded samples
tanList=np.array(tanList)
save(fileroot + 'linEmb', tanList)     #should have format 12 x 10000 x 6 haben

##
# ready
# continue with transport-analysis Notebook