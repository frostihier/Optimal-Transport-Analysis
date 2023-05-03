# Calculation of structure factors and cell densities

from numpy import array, ravel, column_stack, zeros, arange, sqrt, delete, \
    pi, exp, meshgrid, single, savetxt, save, load, loadtxt, ones, mean, median, var, zeros_like, std
import matplotlib.pyplot as plt
import torch
from torch import Tensor
import os

### settings for figures

import matplotlib.font_manager as fm
import matplotlib as mpl

# Collect all the font names available to matplotlib
font_names = [f.name for f in fm.fontManager.ttflist]
print(font_names)

# Edit the font, font size, and axes width
mpl.rcParams['font.family'] = 'Montserrat'
plt.rcParams['font.size'] = 18
plt.rcParams['axes.linewidth'] = 2


## determine paths
# from where to read in the data:
datapath = 'Data/txt-data/'

# where to save the results
resultpath = 'Results/'

##
os.makedirs(resultpath + 'SF-powder', exist_ok = True)
os.makedirs(resultpath + 'SF-3d', exist_ok = True)

## the following code cells contain the calculation of the structure factors out of the nuclei positions.
# for this purpose a CUDA capable graphics card is required
# you can skip this part - below, the calculated structure factors are provided

### proofs if a CUDA-GPU is available
torch.cuda.is_available()

## define function for powder averaging:
from scipy.stats import binned_statistic

def powder_average(S, q_betrag, bin_number):
    q_betrag = ravel(q_betrag)
    S = ravel(S)

    # binned_statistics averages all entries from S which have the same underlying q-values
    # the averaging is done for all q-bins respectively
    bin_means, bin_edges, number = binned_statistic(q_betrag, S, bins=bin_number)

    bin_edges = delete(bin_edges, len(bin_edges) - 1)  # one edge too much, kill last edge
    bin_width = bin_edges[1] - bin_edges[0]  # shift to the middle of the bins
    bin_edges = bin_edges + 0.5 * bin_width

    print(bin_edges.shape)

    return bin_means, bin_edges

## Calculation of the structure factor of each sample

#sampleTags = ['6716']    # try only one sample

# Sequence of samples
sampleTags = ['6716', '6916', '2616', '5416', '12816', '6515', '12006', 'T800', '11313', '13609', '22091', '3614']

# compute 3d structure factor for each sample
for sample in sampleTags:

    # read in the first three columns from txt-data, which contain positions of the nuclei
    pos = loadtxt(datapath + sample + '.txt', usecols= (0,1,2), skiprows=1)
    print('Subject ' + sample)

    # creating the wave vectors
    lim = 5
    step = 0.02
    qx = Tensor(arange(-lim, lim, step, dtype = single))
    qy = Tensor(arange(-lim, lim, step, dtype = single))
    qz = Tensor(arange(-lim, lim, step, dtype = single))

    L = len(qx)
    print('Number of wave vectors in qx: ' ,L)
    print('Number of nuclei: ' ,len(pos))


    #from that create 3d wave vector grid
    Qx, Qy, Qz = meshgrid(qx,qy,qz)
    # Array to compute the structure factor
    GR = torch.zeros((len(qx), len(qy), len(qz)), dtype=torch.cfloat)

    #send all arrays to the GPU
    cuda = torch.device('cuda')
    GQx = Tensor(Qx).to(device = cuda)
    GQy = Tensor(Qy).to(device = cuda)
    GQz = Tensor(Qz).to(device = cuda)
    GR =  GR.to(device = cuda)
    gpos = Tensor(pos).to(device = cuda)


    # calculate structure factor via summation over particle posiitons
    # parallel computation with torch
    for i in range(len(pos)):
        GR += torch.exp(1j * (gpos[i, 0] * GQx + gpos[i, 1] * GQy + gpos[i, 2] * GQz))

    # multiply with complex conjugate
    S = torch.real(GR * torch.conj(GR))

    # finally, normalize and send result back to CPU
    num = len(pos)
    S_norm = S / num
    S_norm = S_norm.cpu()


    # compute 3d array whose entries contain the maginutde of the wavevector
    # has the same shape as S
    B = sqrt(Qx * Qx + Qy * Qy + Qz * Qz)

    # powder averaging, number of bins important
    num_bin = 100
    bin_means, bin_edges = powder_average(S_norm, B, num_bin)
    struct_powderaverage = column_stack((bin_edges,bin_means))

    # save the results
    savetxt(resultpath + 'SF-powder/' + sample + '-powder', struct_powderaverage)
    save(resultpath + 'SF-3d/' + sample + '-3d' ,S_norm)
    #savetxt(resultpath + 'Sampling_data/' + sample + '-sampling-data' , [lim,step])


    print('ready')




## Plotting

## read in all powder averaged structure factors

#CTRL-group
SF1 = loadtxt(resultpath + 'SF-powder/6716-powder')
SF2 = loadtxt(resultpath + 'SF-powder/6916-powder')
SF3 = loadtxt(resultpath + 'SF-powder/2616-powder')
SF4 = loadtxt(resultpath + 'SF-powder/5416-powder')
SF5 = loadtxt(resultpath + 'SF-powder/12816-powder')
SF6 = loadtxt(resultpath + 'SF-powder/6515-powder')

#MS-group
SF7 = loadtxt(resultpath + 'SF-powder/12006-powder')
SF8 = loadtxt(resultpath + 'SF-powder/T800-powder')
SF9 = loadtxt(resultpath + 'SF-powder/11313-powder')
SF10 = loadtxt(resultpath + 'SF-powder/13609-powder')
SF11 = loadtxt(resultpath + 'SF-powder/22091-powder')
SF12 = loadtxt(resultpath + 'SF-powder/3614-powder')


## plotting all powder averaged SF together


plt.figure(figsize=(8, 8))
plt.xlim(0, 4.2)
plt.ylim(0.3, 2.2)
plt.xlabel("Wavenumber q [1/μm]", fontsize=20)
plt.ylabel("Amplitude of S(q)", fontsize=20)
plt.axhline(y=1, c='black', ls = '--', lw = 1.7)
plt.yticks([0.5,1,1.5,2])
ax = plt.gca()
ax.xaxis.set_tick_params(which='major', size=6, width=2, direction='out')
ax.yaxis.set_tick_params(which='major', size=6, width=2, direction='out')
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.title('Powder averaged structure factors', fontsize = 20)

# CTRL
plt.plot(SF1[:,0] , SF1[:,1] , '-', c = 'blue', label = 'Control') #,  label = '6716')
plt.plot(SF2[:,0] , SF2[:,1] , '-', c = 'blue') #, label = '6916')
plt.plot(SF3[:,0] , SF3[:,1] , '-', c = 'blue') #, label = '2616')
plt.plot(SF4[:,0] , SF4[:,1] , '-', c = 'blue') #, label = '5416')
plt.plot(SF5[:,0] , SF5[:,1] , '-', c = 'blue') #, label = '12816')
plt.plot(SF6[:,0] , SF6[:,1] , '-', c = 'blue') #, label = '6515')

# MS
plt.plot(SF7[:,0] , SF7[:,1] , '-', c = 'red', label = 'MS') #, label = '12006')
plt.plot(SF8[:,0] , SF8[:,1] , '-', c = 'red')   #, label = 'T800')
plt.plot(SF9[:,0] , SF9[:,1] , '-', c = 'red')   #, label = '11313')
plt.plot(SF10[:,0] , SF10[:,1] , '-', c = 'red') #, label = '13609')
plt.plot(SF11[:,0] , SF11[:,1] , '-', c = 'red') #, label = '22091')
plt.plot(SF12[:,0] , SF12[:,1] , '-', c = 'red') #, label = 'MS6')



plt.legend(fontsize=17, handletextpad = 0.5, edgecolor = 'black', frameon = True, fancybox = False, facecolor = None)

os.makedirs(resultpath + 'SF-plots', exist_ok = True)
plt.savefig(resultpath + 'SF-plots/All-SF-together.svg', bbox_inches='tight', pad_inches=0.02, transparent = True )
plt.show()



### creating an averaged structure factor for both groups:

# create arrays
ctrls = array([ SF1[:,1], SF2[:,1], SF3[:,1], SF4[:,1], SF5[:,1], SF6[:,1] ])
ms = array([ SF7[:,1], SF8[:,1], SF9[:,1], SF10[:,1], SF11[:,1], SF12[:,1] ])

# averaging
SF_mean_MS = mean(ms, axis = 0)
SF_mean_CTRL = mean(ctrls, axis = 0)

##  create an array which contains the standard error of the mean of the averaged structure factors

std_CTRL = zeros_like(SF_mean_MS)
std_MS = zeros_like(SF_mean_MS)

# calculates standard error of the mean at every sampling point q
for i in range(len(SF_mean_MS)):
    std_MS[i] = std([SF7[i, 1], SF8[i, 1], SF9[i, 1], SF10[i, 1], SF11[i, 1], SF12[i, 1]])  / sqrt(6)
    std_CTRL[i] = std([SF1[i,1], SF2[i,1] , SF3[i,1] , SF4[i,1] , SF5[i,1] , SF6[i,1]] )    / sqrt(6)


## plotting the averaged structure factors together with their errors
# check if the curves each lying in the error interval of the other

plt.figure(figsize=(7, 7))
plt.xlabel("Wavenumber q", fontsize=20)
plt.ylabel("Amplitude of S(q)", fontsize=20)
plt.axhline(y=1, c='black', ls = '--', lw = 1.7)
plt.xticks( fontsize=20)
plt.yticks( fontsize=20)
ax = plt.gca()
ax.xaxis.set_tick_params(which='major', size=6, width=2, direction='in')
ax.yaxis.set_tick_params(which='major', size=6, width=2, direction='in')

plt.title('Averaged structure factors and errors', fontsize = 20)
plt.xlim(0, 3.5)
plt.ylim(0.35, 1.7)
plt.xticks([0,1,2,3])
plt.yticks([0.4,0.8,1.2,1.6])

# plotting the standard error of the mean as area, choose either MS or CTRL
plt.fill_between(SF1[:,0], SF_mean_MS + std_MS, SF_mean_MS - std_MS, color = 'red', alpha = 0.3, label = 'MS Error')
#plt.fill_between(SF1[:,0], SF_mean_CTRL + std_CTRL, SF_mean_CTRL - std_CTRL, color = 'blue', alpha = 0.3, label = 'Control Error')

# plotting structure factor curves
plt.errorbar(SF1[:,0], SF_mean_MS, yerr = None, barsabove=True, capsize = 1, c = 'red', label = 'MS', lw = 1.7)  #yerr = std_MS
plt.errorbar(SF1[:,0], SF_mean_CTRL, yerr = None, c = 'blue', label = 'Control', lw = 1.7)


# plot rectangle
from matplotlib.patches import Rectangle
ax.add_patch(Rectangle((1,0.9),1,0.4, fill = None))

# reordering the labels of the legend
handles, labels = plt.gca().get_legend_handles_labels()
# specify order
order = [2, 1, 0]
plt.legend([handles[i] for i in order], [labels[i] for i in order], fontsize=14, handletextpad = 0.5, edgecolor = 'black', frameon = True, fancybox = False)

plt.savefig(resultpath + 'SF-plots/SF+MSError.svg', bbox_inches='tight', pad_inches=0.02, transparent = True )
plt.show()


## plot structure factors with MS-error range (left) and CTRL-error range (right)

fig = plt.figure(figsize=(15, 7), facecolor = 'white')

plt.title('Averaged structure factors and error ranges', fontsize = 27, pad = 20)
plt.axis('off')

# left plot
ax1 = fig.add_subplot(1, 2, 1)
plt.xlabel("Wavenumber q  [1/μm]", fontsize=20)
plt.ylabel("Amplitude of S(q)", fontsize=20)
plt.axhline(y=1, c='black', ls = '--', lw = 1.7)
plt.xticks( fontsize=20)
plt.yticks( fontsize=20)
ax = plt.gca()
ax.xaxis.set_tick_params(which='major', size=6, width=2, direction='in')
ax.yaxis.set_tick_params(which='major', size=6, width=2, direction='in')
plt.xlim(0, 3.5)
plt.ylim(0.35, 1.7)
plt.xticks([0,1,2,3])
plt.yticks([0.4,0.7,1,1.3,1.6])

# Left plot: MS error
# plotting the std error of mean as area, and both graphs
plt.fill_between(SF1[:,0], SF_mean_MS + std_MS, SF_mean_MS - std_MS, color = 'red', alpha = 0.3, label = 'MS-Error')
plt.errorbar(SF1[:,0], SF_mean_MS, yerr = None, barsabove=True, capsize = 1, c = 'red', label = 'MS', lw = 1.7)
plt.errorbar(SF1[:,0], SF_mean_CTRL, yerr = None, c = 'blue', label = 'CTRL', lw = 1.7)

# draw rectangle
from matplotlib.patches import Rectangle
ax.add_patch(Rectangle((1,0.9),1,0.4, fill = None))

# legend
handles, labels = plt.gca().get_legend_handles_labels()
order = [2, 1, 0]
ax1.legend([handles[i] for i in order], [labels[i] for i in order], fontsize=14, handletextpad = 0.5, edgecolor = 'black', frameon = True, fancybox = False)



# Right plot: CTRL error
ax2 = fig.add_subplot(1, 2, 2)

plt.xlabel("Wavenumber q  [1/μm]", fontsize=20)
#plt.ylabel("Amplitude of S(q)", fontsize=20)
plt.axhline(y=1, c='black', ls = '--', lw = 1.7)
plt.xticks( fontsize=20)
plt.yticks( fontsize=20)
ax = plt.gca()
ax.xaxis.set_tick_params(which='major', size=6, width=2, direction='in')
ax.yaxis.set_tick_params(which='major', size=6, width=2, direction='in')
plt.xlim(0, 3.5)
plt.ylim(0.35, 1.7)
plt.xticks([0,1,2,3])
plt.yticks([0.4,0.7,1,1.3,1.6])

# plotting the std error of mean as area and both structure factors
plt.fill_between(SF1[:,0], SF_mean_CTRL + std_CTRL, SF_mean_CTRL - std_CTRL, color = 'blue', alpha = 0.3, label = 'CTRL-Error')
plt.errorbar(SF1[:,0], SF_mean_MS, yerr = None, barsabove=True, capsize = 1, c = 'red', label = 'MS', lw = 1.7)
plt.errorbar(SF1[:,0], SF_mean_CTRL, yerr = None, c = 'blue', label = 'CTRL', lw = 1.7)

handles, labels = plt.gca().get_legend_handles_labels()
ax2.legend([handles[i] for i in order], [labels[i] for i in order], fontsize=14, handletextpad = 0.5, edgecolor = 'black', frameon = True, fancybox = False)

plt.savefig(resultpath + 'SF-plots/SF+erros.svg', bbox_inches='tight', pad_inches=0.02, transparent = True )
plt.show()

## create a zoom figure

plt.figure(figsize=(7, 5.1))
#plt.xlabel("Wavenumber q", fontsize=20)
#plt.ylabel("Amplitude of S(q)", fontsize=20)
plt.axhline(y=1, c='black', ls = '--', lw = 1.7)
plt.xticks( fontsize=20)
plt.yticks( fontsize=20)
ax = plt.gca()
ax.xaxis.set_tick_params(which='major', size=6, width=2, direction='in')
ax.yaxis.set_tick_params(which='major', size=6, width=2, direction='in')
plt.xticks([1,1.5,2,2.5])
plt.yticks([1,1.1,1.2,1.3])
plt.xlim(1, 2.5)
plt.ylim(0.9, 1.3)


# plotting the std error of means as area, choose either MS or CTRL
plt.fill_between(SF1[:,0], SF_mean_MS + std_MS, SF_mean_MS - std_MS, color = 'red', alpha = 0.3, label = 'MS Error')
#plt.fill_between(SF1[:,0], SF_mean_CTRL + std_CTRL, SF_mean_CTRL - std_CTRL, color = 'blue', alpha = 0.3, label = 'Control Error')

plt.errorbar(SF1[:,0], SF_mean_MS, yerr = None, barsabove=True, capsize = 1, c = 'red', label = 'MS', lw = 1.7)  #yerr = std_MS
plt.errorbar(SF1[:,0], SF_mean_CTRL, yerr = None, c = 'blue', label = 'Control', lw = 1.7)


# reordering the labels of the legend
handles, labels = plt.gca().get_legend_handles_labels()
# specify order
order = [2, 1, 0]
#plt.legend([handles[i] for i in order], [labels[i] for i in order], fontsize=14, handletextpad = 0.5, edgecolor = 'black', frameon = True, fancybox = False)

plt.savefig(resultpath + 'SF-plots/SF+MSError-Zoom.svg', bbox_inches='tight', pad_inches=0.02, transparent = True )
plt.show()




## evaluate differences between the structure factor graphs quantitatively with chi-squared method

## determine boundaries

# the powder-averaged structure factors are noisy for very high and very small $q$ values
# 3d structure factors was sampled to limit 5, so powder averaged SF only valid to the this value.

print(SF1[:,0])

# q = SF[58,0] = 5
print(SF1[58,0])  # upper bound
# choose
print(SF1[3,0])   # lower bound

# -> 3,58 bounds

plt.figure()
plt.plot(SF1[3:58,0], SF_mean_MS[3:58])
plt.show()


### calculate chi-squared value by ratio between std error and distance between SFs
def calculate_chi(SF_1, SF_2, sigma):
    chi = 0
    count = 0
    for i in range(3,58):    # chosen limits
        count += 1
        chi += ((SF_1[i] - SF_2[i])/ sigma[i] )**2

    chi = chi / count

    return(chi)

## Chi-squared values - results

# chi >> 1: SF graph lies outside the error interval of the other -> can be considered as different
# chi < 1: SF graphs close.
print(calculate_chi(SF_mean_MS, SF_mean_CTRL, std_MS))
print(calculate_chi(SF_mean_MS, SF_mean_CTRL, std_CTRL))


## copmute and plot cell density of each sample

## load in positions of the granules

data1 = loadtxt(datapath + '6716.txt',  usecols=(0,1,2), skiprows = 1)
data2 = loadtxt(datapath + '6916.txt',  usecols=(0,1,2), skiprows = 1)
data3 = loadtxt(datapath + '2616.txt',  usecols=(0,1,2), skiprows = 1)
data4 = loadtxt(datapath + '5416.txt',  usecols=(0,1,2), skiprows = 1)
data5 = loadtxt(datapath + '12816.txt', usecols=(0,1,2),  skiprows = 1)
data6 = loadtxt(datapath + '6515.txt',  usecols=(0,1,2), skiprows = 1)

#Group2 - MS
data7 = loadtxt(datapath + '12006.txt', usecols=(0,1,2),  skiprows = 1)
data8 = loadtxt(datapath + 'T800.txt',  usecols=(0,1,2), skiprows = 1)
data9 = loadtxt(datapath + '11313.txt', usecols=(0,1,2),  skiprows = 1)
data10 = loadtxt(datapath + '13609.txt', usecols=(0,1,2),  skiprows = 1)
data11 = loadtxt(datapath + '22091.txt', usecols=(0,1,2),  skiprows = 1)
data12 = loadtxt(datapath + '3614.txt',  usecols=(0,1,2), skiprows = 1)

data = [data1, data2, data3, data4, data5, data6, data7, data8, data9, data10, data11, data12]

## calculate cell densities with ConvexHull

from scipy.spatial import ConvexHull
density_list = []

# use ConvexHull for volume-estimation
for d in data:
    hull = ConvexHull(d[:, 0:3])

    vol = hull.volume * 1e-9
    #volume_list.append(vol)

    density = len(d) / vol
    print(len(d))
    density_list.append(density)
    print('Cell density in 10^-6 mm^-3: ', density)


## plot all density values

fig = plt.figure()
X1 = ones(6) * 0.5
X2 = ones(6) * 1
plt.plot(X1, density_list[0:6], '_', ms=30, lw= 100, c = 'blue', label='CTRL')
plt.plot(X2, density_list[6:12], '_', ms=30, c='red', label='MS')
plt.plot(X1[0], mean(density_list[0:6]), '_', ms=50, c = 'black' )
plt.plot(X2[0], mean(density_list[6:12]), '_', ms=50, c = 'black' )
plt.axvline(x=0.5, ls='--', c='grey')
plt.axvline(x=1, ls='--', c='grey')
ax = plt.gca()
ax.xaxis.set_tick_params(which='major', size=6, width=2, direction='in')
ax.yaxis.set_tick_params(which='major', size=6, width=2, direction='in')
plt.title('Cell densities')
plt.xlim(0,1.5)
#plt.yticks([-1,0,1])
plt.xticks(ticks = [0.5,1], labels = ['CTRL', 'MS'])
plt.ylabel('Density in mm$^{-3}$')

os.makedirs(resultpath + 'Cell-densities', exist_ok = True)
plt.savefig(resultpath + 'Cell-densities/Cell_densites.svg',  bbox_inches='tight', pad_inches=0.02, transparent = True)
plt.show()



## print median values, std deviation and p-values of density in console
print('ALL')
print(median(density_list[0:12]))
print(std(density_list[0:12]))

print('--CTRL--')
print(median(density_list[0:6]))
print(std(density_list[0:6]))

print('--MS--')
print(median(density_list[6:12]))
print(std(density_list[6:12]))

from scipy.stats import ttest_ind
print(ttest_ind(density_list[6:12], density_list[0:6], equal_var=False))


##

