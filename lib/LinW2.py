import numpy as np
import scipy
import scipy.sparse

def extractMongeData(pi,posNu):
    """Extract approximate Monge map from optimal coupling
    
    pi: optimal coupling in sparse.csr format
    posNu: positions of second marginal masses"""
    
    # compute second marginal of coupling pi
    pi1=np.array(pi.sum(axis=0)).flatten()
        
    # reserve empty array for Monge map
    T=np.zeros((pi.shape[0],posNu.shape[1]),dtype=np.double)

    # go through points in barycenter
    for j in range(T.shape[0]):
        # check if current row is empty
        if pi.indptr[j+1]==pi.indptr[j]:
            continue

        # extract masses in that row of the coupling (based on csr format)
        piRow=pi.data[pi.indptr[j]:pi.indptr[j+1]]
        # normalize masses
        piRow=piRow/np.sum(piRow)
        # extract indices non-zero entries (based on csr format)
        piIndRow=pi.indices[pi.indptr[j]:pi.indptr[j+1]]
        
        # need einsum for averaging along first ("zeroth") axis
        T[j,:]=np.einsum(posNu[piIndRow],[0,1],piRow,[0],[1])
        
    return T

