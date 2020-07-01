
import PATnets as PAT
import h5py
import numpy as np
import matplotlib.pyplot as plt

# import matplotlib.pyplot as plt


''' Load system matrix generated with k-wave in Matlab '''
matrixName = 'forwMat20.mat'
fData = h5py.File(matrixName,'r')
inData = fData.get('Athresh')  #We use the thresholded version
rows = inData.shape[0]
cols = inData.shape[1]
print(rows, cols)
forwMat = np.float32(np.matrix(inData)) #Conversion to single precision

#Matrix dimensions defines the imaging geometry
imSize = int(np.sqrt(cols))
tLen   = int(rows/imSize)



#Create training set
trainSet = 'PhantomData/trainDataCombined.mat'
testSet  = 'PhantomData/testDataCombined.mat'

trainFlag = False


dataSet = PAT.read_data_sets(trainSet,testSet,tLen)

bSize               = int(4)
trainIter           = 50001           # Training iteration
useTensorboard      = False            # Use Tensorboard for tracking
filePath            = 'netData/'      # Where to save the network, can be absolute or relative
LGSiter             = 10


#####

lossFunc    = 'l2_loss'        #'l1_loss, l2_loss'
netType    = 'resUnet'         # 'Unet, resUnet, LGS, fullyLearned'

experimentName =  netType + 'test' + '_' + lossFunc   #Name for this experiment


noiseLev = 0.01

print('preparing input data')
''' Prepare input data '''
for idx in range(dataSet.test.true.shape[0]):


    p0_true=dataSet.test.true[idx]
    p0_true[:,0]=0.0
    dataSet.test.true[idx] = p0_true
    
    p0_vec=np.reshape(p0_true,[cols])
    pMeas=np.reshape(np.matmul(forwMat,p0_vec),[tLen,imSize])

    #Add noise
    pMeas += np.random.normal(size=[tLen,imSize])*np.max(pMeas)*noiseLev

    dataSet.test.data[idx,:,:,0]=np.reshape(pMeas,[tLen,imSize,1])
    '''Note, for LGS we only need pMeas'''    
    pMeas_vec=np.reshape(np.array(pMeas),[rows])
    p0_back = np.matmul(np.transpose(forwMat),pMeas_vec) 
        
    dataSet.test.reco[idx,:,:,0]=np.reshape(p0_back,[imSize,imSize,1])




rec = PAT.evaluation(dataSet,forwMat,netType,experimentName,filePath,
             lossFunc = lossFunc,
             bSize = bSize,
             trainIter = trainIter,
             LGSiter = LGSiter)
    


idx = 0    
plt.figure()
plt.imshow(rec[idx,:,:,0])
plt.title('Reconstruction')

plt.figure()
plt.imshow(dataSet.test.true[idx,:,:,0])
plt.title('Phantom')

plt.figure()
plt.imshow(dataSet.test.reco[idx,:,:,0])
plt.title('Input')