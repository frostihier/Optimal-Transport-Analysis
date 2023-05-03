from .header_common import *

from numpy import linalg as lin
from scipy.linalg import sqrtm as sqrtm 

# W1 barycenter for (centered) Gaussian, implemented by Hugo Lavenant
def barycenter(matrices, weights, initialGuess = 1, tol = 10**(-5)) : 
    """
    matrices is an arry of symmetric matrices
    weight is an array of positive weights 
    initialGuess is an optional matrix specifying the starting point of the iterative procedure. If not specified, we start with the identity matrix. 
    tol is the tolerance for the iterative process to stop 
    """

    # Normalize the weights 
    weights = weights / np.sum(weights)

    # Get the dimensions
    # Number of matrices 
    n = matrices.shape[0] 
    # Dimension of each matrix 
    d = matrices.shape[1]

    # Initialize the output of the iterative algorithm
    if type(initialGuess) == int : 
        output = np.eye(d)
    else : 
        output = initialGuess

    error = tol + 1. 

    # Loop 
    while error > tol : 

        # Form the sum 
        auxSum = np.zeros((d,d))
        auxSqrt = sqrtm(output)
        for i in range(n) : 
            auxSum += weights[i] * sqrtm( np.dot( auxSqrt, np.dot(  matrices[i,:,:], auxSqrt  ) ) )
        # Take the square and bracket 
        auxSum = np.dot( auxSum, auxSum )
        auxSqrt = lin.inv( auxSqrt  )
        newOutput = np.dot(  auxSqrt, np.dot(  auxSum, auxSqrt  ) )

        error = lin.norm(  newOutput - output  )
        output = newOutput

    return output

# Bures metrik between pos. def. matrices
def BuresSqr(A,B):
    sqrtA=scipy.linalg.sqrtm(A)
    tmp=A+B-2*scipy.linalg.sqrtm(sqrtA.dot(B.dot(sqrtA)))
    return np.real(tmp.trace())

def Bures(A,B):
    return np.sqrt(BuresSqr(A,B))


def GaussW2Sqr(A,B,meanA,meanB):
    return np.sum((meanA-meanB)**2)+BuresSqr(A,B)

def GaussW2(A,B,meanA,meanB):
    return np.sqrt(GaussW2Sqr(A,B,meanA,meanB))

# rotation matrix, useful for creating examples to test ellipse plotting and bures metric
def R(phi):
    return np.array([[np.cos(phi),-np.sin(phi)],[np.sin(phi),np.cos(phi)]])



# linW2 functions for Gaussian measures
# note: convention of linear embedding is such that inner product is default Euclidean product
# formulas are given in notes/2020-11-30-BuresMetricIntegrals.xoj

# These are the centered versions, where mean=0 for the Gaussians is assumed
def LinW2Log(A,Q,QSqrt=None,QSqrtInv=None):
    """Compute standardized Euclideanized linear embedding of cov matrix A wrt Q"""
    if QSqrt is None:
        QSqrt=scipy.linalg.sqrtm(Q)
    if QSqrtInv is None:
        QSqrtInv=scipy.linalg.inv(QSqrt)
    result=scipy.linalg.sqrtm(QSqrt.dot(A.dot(QSqrt))).dot(QSqrtInv)-QSqrt
    return result

def LinW2Exp(S,Q,QSqrt=None,QSqrtInv=None):
    if QSqrt is None:
        QSqrt=scipy.linalg.sqrtm(Q)
    if QSqrtInv is None:
        QSqrtInv=scipy.linalg.inv(QSqrt)
    SEff=S+QSqrt
    result=QSqrtInv.dot(SEff.dot(QSqrt.dot(SEff)))
    return result

# general versions with non-zero mean, append mean coordinates at the end
def LinW2LogMean(A,mA,Q,mQ,QSqrt=None,QSqrtInv=None):
    """Compute standardized Euclideanized linear embedding of cov matrix A wrt Q"""
    if QSqrt is None:
        QSqrt=scipy.linalg.sqrtm(Q)
    if QSqrtInv is None:
        QSqrtInv=scipy.linalg.inv(QSqrt)
    result=scipy.linalg.sqrtm(QSqrt.dot(A.dot(QSqrt))).dot(QSqrtInv)-QSqrt
    # now flatten and append difference of means to the end
    result=result.ravel()
    result=np.concatenate((result,(mA-mQ).ravel()))
    return result

def LinW2ExpMean(S,Q,mQ,QSqrt=None,QSqrtInv=None):
    if QSqrt is None:
        QSqrt=scipy.linalg.sqrtm(Q)
    if QSqrtInv is None:
        QSqrtInv=scipy.linalg.inv(QSqrt)
        
    # first unravel S, obtain original dimension from shape of Q
    n=Q.shape[0]
    SEff=S[:n*n].reshape((n,n))+QSqrt
    mEff=S[n*n:]+mQ
        
    result=QSqrtInv.dot(SEff.dot(QSqrt.dot(SEff)))
    return result,mEff

def drawGaussian1D(mean,var,x):
    return np.exp(-0.5*(x-mean)**2/var)/np.sqrt(2*np.pi*var)

def drawGaussianND(mean,cov,x):
    n=mean.shape[0]
    covInv=np.linalg.inv(cov)
    return np.exp(-0.5*np.einsum((x-mean),[0,1],(x-mean),[0,2],covInv,[1,2],[0]))\
            /(2*np.pi)**(n/2.)/(np.linalg.det(cov))**0.5

#def sampleFromGaussian(Q,n):
#    """Sample n random points from Gaussian with covariance matrix Q"""
#    A=scipy.linalg.sqrtm(Q)
#    d=A.shape[0]
#    result=np.random.normal(size=(n,d))
#    result=np.einsum(A,[0,1],result,[2,1],[2,0])
#    return result

def sampleFromGaussian(cov,mean,n,rng):
    """Sample n random points from Gaussian with covariance matrix cov and mean mean"""
    A=scipy.linalg.sqrtm(cov)
    d=A.shape[0]
    result=rng.normal(size=(n,d))
    result=np.einsum(A,[0,1],result,[2,1],[2,0])+mean
    return result


def getCostEuclidean(x,y):
    dim=x.shape[1]
    nx=x.shape[0]
    ny=y.shape[0]
    diff=x.reshape((nx,1,dim))-y.reshape((1,ny,dim))
    result=np.sum(diff**2,axis=2)
    return result


def importPts(filename):
    dat=sciio.loadmat(filename)
    return np.array(dat["pts"],order="C",dtype=np.double)

def importSol(filename):
    dat=sciio.loadmat(filename)
    return [np.array(dat["alpha"],order="C",dtype=np.double).ravel(),
            np.array(dat["beta"],order="C",dtype=np.double).ravel()]

def PCA(dataMat,keep=None):
    nSamples,dim=dataMat.shape
    print('Samples' , nSamples)
    print('Dim' , dim)
    if dim<nSamples:
        if keep is None:
            keep=dim
        A=dataMat.transpose().dot(dataMat)/nSamples
        eigData=np.linalg.eigh(A)
        eigval=(eigData[0][-keep::])[::-1]
        eigvec=((eigData[1][:,-keep::]).transpose())[::-1]
    else:
        if keep is None:
            keep=nSamples
        A=dataMat.dot(dataMat.transpose())/nSamples
        eigData=np.linalg.eigh(A)
        eigval=(eigData[0][-keep::])[::-1]
        eigvec=((eigData[1][:,-keep::]).transpose())[::-1]

        eigvec=np.einsum(eigvec,[0,1],dataMat,[1,2],[0,2])
        # renormalize
        normList=np.linalg.norm(eigvec,axis=1)
        eigvec=np.einsum(eigvec,[0,1],1/normList,[0],[0,1])
    return eigval,eigvec

##

