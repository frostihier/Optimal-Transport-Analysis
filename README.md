
# RECOMMENDATIONS/REQUIREMENTS to run the scripts: 


The scripts were executed with Pycharm and in cell mode. The cell delimiters are marked by two consecutive number-signs ##. 

Version of Python interpreter: 3.9

Required Python-libraries: Numpy, Scipy, Matplotlib, Sklearn, POT (Python optimal transport), OS

For the own calculation of the structure factors, Pytorch and a CUDA capable graphics card is needed, 
but you can skip this part since already calculated structure facotrs are provide in the folders 
'Results/SF-powder and Results/SF-3d.  




# DESCRIPTION: 

### Scripts: 

The script Structure-factors.py contains the calculation of the structure factors and the cell density of all patients out of the positions of the nuclei.

The script Prepare-data&1d-analysis.py contains the creation of the feature space and the 1d histogram analysis (t-test, median values and 1dOT). 
Furthermore, the Gaussian approximations and the linear embedding are calculated for the multidimensional analysis with OT (see Transport-Analysis-script).

The script Transport-Analysis.py builds on the previous Prepare-data-script and contains the multidimensional analysis with Optimal Transport
with all results up to the final push forwards.



### Folders:

The folder 'Data/txt-data' contains txt-tables with the positions and properties of the nuclei obtained from the segmentation. Each file corresponds to one patient where 
the file-numbers indicate the patient numbers. Note this data is also provided in the excel-files.  

The folder 'Data/Data-for-Transport-Analysis' contains all the data created/intermediately stored in the script Prepare-data, and then used in the script Transport-Analysis 
(Gaussian approximations, linear embedding, reference sample ) 

lib-Folder: This folder contains a collection of Optimal Transport Algorithms (Barycenter computation, OT Solver) created by Bernhard Schmitzer and the 
Optimal Transport Group in Göttingen (unpublished). The documented library can be found also on Github: https://github.com/bernhard-schmitzer/UnbalancedLOT    

params_txt: This file contains chosen parameters for the entropic regularized solver of the transport problem.


DOI: [doi.org/10.1016/j.neuroscience.2023.04.002](https://www.sciencedirect.com/science/article/pii/S0306452223001616?via%3Dihub)

### Credits / Citation:

Cite the manuscript below, when using these scripts or data. 
 
3d virtual histology reveals pathological alterations of cerebellar granule cells in multiple sclerosis
Jakob Frost, Bernhard Schmitzer, Mareike Töpperwien, Marina Eckermann, Jonas Franz, Christine Stadelmann, Tim Salditt
NeuroScience (2023). 

In case of questions, please address authors of the corresponding manuscript.
 
