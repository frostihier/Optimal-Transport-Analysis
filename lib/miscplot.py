import numpy as np
import scipy

# plot covariance matrix as ellipse in 2d, with major axis given by std dev along eigenvector directions
from matplotlib.patches import Ellipse
def getEllipse(mat,mean=np.array([0.,0.])):
    eigdat=scipy.linalg.eigh(mat)
    v=eigdat[1][:,-1] # eigenvector of dominant eigenvalue
    rePhi=np.arctan2(v[1],v[0]) # orientation of this eigenvector
    eigdat[0][...]=np.maximum(eigdat[0],0.)
    result=Ellipse(xy=mean,width=2*np.sqrt(eigdat[0][1]),height=2*np.sqrt(eigdat[0][0]),angle=rePhi/np.pi*180)
    #result=Ellipse(xy=mean,width=2*(eigdat[0][1]),height=2*(eigdat[0][0]),angle=rePhi)
    return result

