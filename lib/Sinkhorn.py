import numpy as np
import scipy
import scipy.sparse
import MultiScaleOT

def SolveHK(muX,posX,muY,posY,WFScale,
        SinkhornError,
        epsTarget, epsInit
        ):
    """Computes squared HK distance between two measures represented by weighted point clouds
    (muX,posX) and (muY,posY).
    All distances divided by WFScale, final result multiplied by WFScale**2.
    
    returns:
    value: approximate squared HK distance between two measures
    piCSR: approximate optimal coupling pi as scipy sparse CSR matrix
    """

    # spatial dimension, extract from point cloud array
    dim=posX.shape[1]
    
    # multi-scale setup parameters
    hierarchyDepth=0
    hierarchyCoarsest=0
    hierarchyFinest=hierarchyDepth


    # use WFmode (instead of standard W2)
    WFMode=True


    # setup OTToolbox solver
    
    # setup marginals
    MultiScaleSetupX=MultiScaleOT.TMultiScaleSetup(posX,muX,hierarchyDepth,\
            childMode=MultiScaleOT.childModeTree)

    MultiScaleSetupY=MultiScaleOT.TMultiScaleSetup(posY,muY,hierarchyDepth,\
            childMode=MultiScaleOT.childModeTree)

    # which cost function to use?
    costFunction=MultiScaleOT.THierarchicalCostFunctionProvider_SquaredEuclidean(
            MultiScaleSetupX,MultiScaleSetupY,WFMode,WFScale)

    # eps scaling
    epsScalingHandler=MultiScaleOT.TEpsScalingHandler()
    # use the following epsScaling schedule setup. these settings (with default parameters) should work on essentially all problems
    #epsScalingHandler.setupGeometricMultiLayerB(hierarchyDepth+1,epsFactor,4.,2,epsFinalSteps)

    # eps version for single layer, compute nr of steps such that reduction factor is larger than 0.5
    nSteps=int((np.log(epsInit)-np.log(epsTarget))/np.log(2)+1)
    epsScalingHandler.setupGeometricSingleLayer(1,epsInit,epsTarget,nSteps)


    # various other solver parameters
    cfg=MultiScaleOT.TSinkhornSolverParameters()

    # create the actual solver object
    SinkhornSolver=MultiScaleOT.TSinkhornSolverKLMarginals(epsScalingHandler,
            hierarchyCoarsest,hierarchyFinest,SinkhornError,
            MultiScaleSetupX,MultiScaleSetupY,costFunction,WFScale**2,
            cfg)
    
    # initialize and solve
    msg=SinkhornSolver.initialize()
    if msg!=0: raise ValueError("init: {:d}".format(msg))
    msg=SinkhornSolver.solve()
    if msg!=0: raise ValueError("solve: {:d}".format(msg))

    # extract unregularized primal score
    value=SinkhornSolver.getScorePrimalUnreg()
    
    # extract optimal coupling in scipy sparse CSR format:
    piCSRData=SinkhornSolver.getKernelCSRDataTuple()

    xres=muX.shape[0]
    yres=muY.shape[0]


    piCSR=scipy.sparse.csr_matrix(piCSRData,shape=(xres,yres))
    piCSR.sort_indices()
    
    # return optimal value and SinkhornSolver object in case one wants to do post processing
    return (value,piCSR)


def SolveW2(muX,posX,muY,posY,
        SinkhornError,
        epsTarget, epsInit
        ):
    """Computes squared W_2 distance between two measures represented by weighted point clouds
    (muX,posX) and (muY,posY).
    
    returns:
    value: approximate squared W_2 distance between two measures
    piCSR: approximate optimal coupling pi as scipy sparse CSR matrix
    """

    # spatial dimension, extract from point cloud array
    dim=posX.shape[1]
    
    # multi-scale setup parameters
    hierarchyDepth=0
    hierarchyCoarsest=0
    hierarchyFinest=hierarchyDepth


    # setup OTToolbox solver
    
    # setup marginals
    MultiScaleSetupX=MultiScaleOT.TMultiScaleSetup(posX,muX,hierarchyDepth,\
            childMode=MultiScaleOT.childModeTree)

    MultiScaleSetupY=MultiScaleOT.TMultiScaleSetup(posY,muY,hierarchyDepth,\
            childMode=MultiScaleOT.childModeTree)

    # which cost function to use?
    costFunction=MultiScaleOT.THierarchicalCostFunctionProvider_SquaredEuclidean(
            MultiScaleSetupX,MultiScaleSetupY,False)

    # eps scaling
    epsScalingHandler=MultiScaleOT.TEpsScalingHandler()
    # use the following epsScaling schedule setup. these settings (with default parameters) should work on essentially all problems
    #epsScalingHandler.setupGeometricMultiLayerB(hierarchyDepth+1,epsFactor,4.,2,epsFinalSteps)

    # eps version for single layer, compute nr of steps such that reduction factor is larger than 0.5
    nSteps=int((np.log(epsInit)-np.log(epsTarget))/np.log(2)+1)
    epsScalingHandler.setupGeometricSingleLayer(1,epsInit,epsTarget,nSteps)



    # various other solver parameters
    cfg=MultiScaleOT.TSinkhornSolverParameters()

    # create the actual solver object
    SinkhornSolver=MultiScaleOT.TSinkhornSolverStandard(epsScalingHandler,
            hierarchyCoarsest,hierarchyFinest,SinkhornError,
            MultiScaleSetupX,MultiScaleSetupY,costFunction,
            cfg)
    
    # initialize and solve
    msg=SinkhornSolver.initialize()
    if msg!=0: raise ValueError("init: {:d}".format(msg))
    msg=SinkhornSolver.solve()
    if msg!=0: raise ValueError("solve: {:d}".format(msg))

    # extract unregularized primal score
    value=SinkhornSolver.getScorePrimalUnreg()
    
    # extract optimal coupling in scipy sparse CSR format:
    piCSRData=SinkhornSolver.getKernelCSRDataTuple()

    xres=muX.shape[0]
    yres=muY.shape[0]


    piCSR=scipy.sparse.csr_matrix(piCSRData,shape=(xres,yres))
    piCSR.sort_indices()
    
    # return optimal value and SinkhornSolver object in case one wants to do post processing
    return (value,piCSR)

