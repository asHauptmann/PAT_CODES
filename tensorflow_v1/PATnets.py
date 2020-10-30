''' Class file Learned image recontruction for PAT
https://doi.org/10.1117/1.JBO.25.11.112903
Written 2020 by Andreas Hauptmann, University of Oulu and UCL'''

import tensorflow.compat.v1 as tf
import numpy as np
import h5py
import os 
from os.path import exists
from skimage.measure import compare_ssim as ssim

FLAGS = None

''' !!!SET Directory for Tensorboard!!! '''
tensorboardDefaultDir =  '/Path4tensorboard/' #Here the logging data will be saved

tf.disable_v2_behavior()


#Loading part

def extract_images(filename,imageName):
  """Extract the images into a 4D uint8 numpy array [index, y, x, depth]."""
  fData = h5py.File(filename,'r')
  inData = fData.get(imageName)  
      
  
  num_images = inData.shape[0]
  rows = inData.shape[1]
  cols = inData.shape[2]
  print('Data size of: ' + imageName)
  print(num_images, rows, cols)
  data = np.array(inData)
    
  data = data.reshape(num_images, rows, cols)
  return data

class DataSet(object):

  def __init__(self, data, true,reco):
    """Construct a DataSet"""

    assert data.shape[0] == true.shape[0], (
        'images.shape: %s labels.shape: %s' % (data.shape,
                                                 true.shape))
    data = data.reshape(data.shape[0],
                            data.shape[1],data.shape[2],1)
    true = true.reshape(true.shape[0],
                            true.shape[1],true.shape[2],1)
    reco = reco.reshape(reco.shape[0],
                            reco.shape[1],reco.shape[2],1)
    
    self._num_examples = data.shape[0]

    self._data = data
    self._true = true
    self._reco = reco
    self._epochs_completed = 0
    self._index_in_epoch = 0

  @property
  def data(self):
    return self._data

  @property
  def true(self):
    return self._true
 
  @property
  def reco(self):
    return self._reco

  @property
  def num_examples(self):
    return self._num_examples

  @property
  def epochs_completed(self):
    return self._epochs_completed

  def next_batch(self, batch_size):
    """Return the next `batch_size` examples from this data set."""
    start = self._index_in_epoch
    self._index_in_epoch += batch_size
    if self._index_in_epoch > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1
      # Shuffle the data
      perm = np.arange(self._num_examples)
      np.random.shuffle(perm)
      self._data = self._data[perm]
      self._true = self._true[perm]
      self._reco = self._reco[perm]
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size
      assert batch_size <= self._num_examples
    end = self._index_in_epoch
    return self._data[start:end], self._true[start:end], self._reco[start:end]


def read_data_sets(FileNameTrain,FileNameTest,tLen):
  class DataSets(object):
    pass
  data_sets = DataSets()

  TRAIN_SET = FileNameTrain
  TEST_SET  = FileNameTest
  TRUE_NAME  = 'imagesTrue'
  
  print('Start loading data')  
  '''Placeholder for now with truth'''  
  train_true   = extract_images(TRAIN_SET,TRUE_NAME)
  test_true   = extract_images(TEST_SET,TRUE_NAME)
  
  #Create empty placeholder for data and reconstruction
  test_data   = np.zeros([test_true.shape[0],tLen,test_true.shape[2]])
  train_data  = np.zeros([train_true.shape[0],tLen,train_true.shape[2]])
  
  test_reco   = np.zeros(test_true.shape)
  train_reco  = np.zeros(train_true.shape)

  data_sets.train = DataSet(train_data, train_true,train_reco)
  data_sets.test = DataSet(test_data, test_true,test_reco)

  return data_sets


def Unet(x,imSize,bSize):

  x_image = tf.reshape(x, [-1, imSize[1],imSize[2], 1])
  
  # First convolutional layer - maps to 32 channels
  x_layer1=tf.layers.conv2d(x_image,32,3,padding='same',activation='relu')
  x_layer1=tf.layers.conv2d(x_layer1,32,3,padding='same',activation='relu')
  
  # First Maxpool layer
  x_layer2 = max_pool_2x2(x_layer1)
  
  # convolutional layer -- maps 32 channels 64.
  x_layer2=tf.layers.conv2d(x_layer2,64,3,padding='same',activation='relu')
  x_layer2=tf.layers.conv2d(x_layer2,64,3,padding='same',activation='relu')
    
  # Second maxpool layer
  x_layer3 = max_pool_2x2(x_layer2)
  
  # convolutional layer -- maps 64 to 128 channels
  x_layer3=tf.layers.conv2d(x_layer3,128,3,padding='same',activation='relu')
  x_layer3=tf.layers.conv2d(x_layer3,128,3,padding='same',activation='relu')
  

  # Prepare for Upsampling   
  layerSizeX=int(imSize[1]/2)
  layerSizeY=int(imSize[2]/2)
  
  #Upsample and concat ------ B2
  W_TconvPrioB3 = weight_variable([3, 3, 64, 128]) #Ouput, Input channels
  b_TconvPrioB3 = bias_variable([64])
  x_layer3 = tf.nn.relu(conv2d_trans(x_layer3, W_TconvPrioB3,[bSize,layerSizeX,layerSizeY,64]) + b_TconvPrioB3)
  
  x_layer2 = tf.concat([x_layer2,x_layer3],3)
  
  
  # convolutional layer -- maps 128 channels to 64.
  x_layer2=tf.layers.conv2d(x_layer2,64,3,padding='same',activation='relu')
  x_layer2=tf.layers.conv2d(x_layer2,64,3,padding='same',activation='relu')
  
  
  #Upsample and concat
  W_TconvPrioB2 = weight_variable([3, 3, 32, 64]) #Ouput, Input channels
  b_TconvPrioB2 = bias_variable([32])
  x_layer2 = tf.nn.relu(conv2d_trans(x_layer2, W_TconvPrioB2,[bSize,imSize[1],imSize[2],32]) + b_TconvPrioB2)
  
  x_layer1 = tf.concat([x_layer1,x_layer2],3)
  
  # First convolutional layer - maps 64 to 32 channels
  x_layer1=tf.layers.conv2d(x_layer1,32,3,padding='same',activation='relu')
  x_layer1=tf.layers.conv2d(x_layer1,32,3,padding='same',activation='relu')
  
  # Last convolutional layer to 1 channel - no ReLU!
  x_layer1=tf.layers.conv2d(x_layer1,1,3,padding='same',activation=None)
  
  x_update = tf.reshape(x_layer1, [-1, imSize[1],imSize[2],1])

  return x_update

def conv2d_trans(x, W, shape):
  '''conv2_trans for transpose convolution and upsampling'''  
  return tf.nn.conv2d_transpose(x, W, shape, strides=[1, 2, 2, 1], padding='SAME')

def max_pool_2x2(x):
  """max_pool_2x2 downsamples a feature map by 2X."""
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

def weight_variable(shape):
  """weight_variable generates a weight variable of a given shape."""
  initial = tf.truncated_normal(shape, stddev=0.025)
  return tf.Variable(initial)

def bias_variable(shape):
  """bias_variable generates a bias variable of a given shape."""
  initial = tf.constant(0.025, shape=shape)
  return tf.Variable(initial)






def denseBlock(x,imSize,imSizeOut,bSize,regParam):

  x_data = tf.reshape(x, [-1, imSize[1]*imSize[2]*imSize[3]])
  
  regularizer = tf.keras.regularizers.l1(regParam)
  x_1 = tf.layers.dense(x_data,imSize[1]*imSize[2]*imSize[3]*2,activation='elu',use_bias = True,kernel_regularizer = regularizer)
  x_1 = tf.layers.dense(x_1,imSizeOut[1]*imSizeOut[2],activation=None,use_bias = False,kernel_regularizer = regularizer)
    
  x_out = tf.reshape(x_1, [-1, imSizeOut[1],imSizeOut[2],1])
  
  return x_out

def preprocBlock(x,bSize):

#   First convolutional layer - maps to 32 channels
  x_layer1=tf.layers.conv2d(x,16,3,strides=[2, 2],padding='same',activation='relu')
  x_layer1=tf.layers.conv2d(x_layer1,32,3,strides=[2, 2],padding='same',activation='relu')
  x_update=tf.layers.conv2d(x_layer1,16,3,strides=[2, 1],padding='same',activation=None)

  return x_update


def resBlock(x,imSize,bSize):

#  x_image = tf.reshape(x, [-1, imSize[1],imSize[2], 1])
  
  # First convolutional layer - maps to 32 channels
  x_layer1=tf.layers.conv2d(x,32,3,padding='same',activation='relu')
  x_layer1=tf.layers.conv2d(x_layer1,32,3,padding='same',activation='relu')
  x_layer1=tf.layers.conv2d(x_layer1,32,3,padding='same',activation='relu')
  
  # Last convolutional layer to 1 channel - no ReLU!
  x_layer1=tf.layers.conv2d(x_layer1,1,3,padding='same',activation=None)
  
  x_update = tf.reshape(x_layer1, [bSize, imSize[1],imSize[2],1])

  return x_update


def psnr(x_result, x_true, name='psnr'):
    with tf.name_scope(name):
        maxval = tf.reduce_max(x_true) - tf.reduce_min(x_true)
        mse = tf.reduce_mean((x_result - x_true) ** 2)
        return 20 * log10(maxval) - 10 * log10(mse)

def log10(x):
    numerator = tf.log(x)
    denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator        
 

def applyMat(forwMat,vecs,outSize,bSize,transpose=False):
    vecOut = []
    for idx in range(bSize):
        vecCur = vecs[idx]
        vecCur = tf.reshape(vecCur,[vecs.shape[1]*vecs.shape[2]])      
        vecCur = forwMat.matvec(vecCur,adjoint=transpose)
        vecOut = tf.concat([vecOut,vecCur],0)
        
    vecOut = tf.reshape(vecOut,[bSize,outSize[1],outSize[2],1])
    return vecOut
    


def LGS(data,A,imSizeIn,imSizeOut,bSize,iterNum):
    
  forwMat = tf.linalg.LinearOperatorFullMatrix(A)
  x_out = applyMat(forwMat,data,imSizeOut,bSize,transpose=True)
  
  for iter in range(iterNum):
      #Compute gradient of data fidelity
      forw = applyMat(forwMat,x_out,imSizeIn,bSize)
      residual = forw - data
      grad = applyMat(forwMat,residual,imSizeOut,bSize,transpose=True)
      
      #Concat iterate with gradient
      x_in = tf.concat([x_out,grad],3)
      
      #Update iteration
      x_out = x_out + resBlock(x_in,imSizeOut,bSize)
  
  
  return x_out, grad, residual
  
  

def training(dataSet,A,netType,experimentName,filePath,
             lossFunc = 'l2_loss',
             bSize = 4,
             trainIter = 50001,
             LGSiter = 10,
             useTensorboard = True,
             lValInit=1e-3,
             regParam = 1e-3,
             finalRelu = True):
# Import data
  
   #Learning rate (could be changed)  
  
  sess = tf.InteractiveSession()    
  imSizeOut=dataSet.train.true.shape
  imSizeIn=dataSet.train.data.shape
  # Create the model
  data = tf.placeholder(tf.float32, [None, imSizeIn[1],imSizeIn[2],1])
  reco = tf.placeholder(tf.float32, [None, imSizeOut[1],imSizeOut[2],1])
  true = tf.placeholder(tf.float32, [None, imSizeOut[1],imSizeOut[2],1])
  
  
  
  # Build the graph for the network used
  if netType == 'Unet':
      print('Using 2D Unet')
      x_out = Unet(reco,imSizeOut,bSize)
  elif netType == 'resUnet':
      print('Using residual 2D Unet')
      x_out = reco + Unet(reco,imSizeOut,bSize)    
  elif netType == 'fullyLearned':
      print('Using a fully learned approach')
      x_1 = denseBlock(data,data.shape,imSizeOut,bSize,regParam)
      x_out = resBlock(x_1,imSizeOut,bSize)
  elif netType == 'LGS':
      print('Using LGS: ' + str(LGSiter) + ' iterations')
      x_out, grad, residual = LGS(data,A,imSizeIn,imSizeOut,bSize,LGSiter)
      
  else:
      print('Not supported network')
      return
  
  '''Final ReLU layer: TURN OFF IF NOT NEEDED'''
  if(finalRelu):
      print('Final maximum projection by ReLU')  
      x_out = tf.nn.relu(x_out)
      

  with tf.name_scope('optimizer'):
         
         if lossFunc == 'l2_loss':
             loss = tf.nn.l2_loss(x_out-true)/tf.nn.l2_loss(true)
             print('Using l2-loss')
         elif lossFunc == 'l1_loss':
             loss = tf.reduce_sum(tf.abs(x_out-true))/tf.reduce_sum(tf.abs(true))
             print('Using l1-loss')
         else: 
             print('Not supported loss function')
             return
         psnrEval = psnr(x_out, true)    
         lossComb = loss + tf.losses.get_regularization_loss()
         learningRate=tf.constant(1e-3) # This is an init, can be changed
         train_step = tf.train.AdamOptimizer(learningRate).minimize(lossComb)
         
    
  if(useTensorboard):
      with tf.name_scope('summaries'):
        tf.summary.scalar('loss', lossComb)
        tf.summary.scalar('psnr', psnr(x_out, true))
    
    
        tf.summary.image('result', tf.reshape(x_out[0],[1, imSizeOut[1], imSizeOut[2], 1]) )        
        tf.summary.image('true', tf.reshape(true[0],[1, imSizeOut[1], imSizeOut[2], 1]) )
        
        if netType == 'resUnet' or netType == 'Unet':
            tf.summary.image('recon', tf.reshape(reco[0],[1, imSizeOut[1], imSizeOut[2], 1]) )
        else:
            tf.summary.image('data', tf.reshape(data[0],[1, imSizeIn[1], imSizeIn[2], 1]) )
        
        if netType == 'LGS':
            tf.summary.image('grad', tf.reshape(grad[0],[1, imSizeOut[1], imSizeOut[2], 1]) ) 
            tf.summary.image('residual', tf.reshape(residual[0],[1, imSizeIn[1], imSizeIn[2], 1]) )
        if netType == 'fullyLearned':
            tf.summary.image('denseResult', tf.reshape(x_1[0],[1, imSizeOut[1], imSizeOut[2], 1]) ) 
        merged_summary = tf.summary.merge_all()
        test_summary_writer, train_summary_writer = summary_writers('default',experimentName)
          
    
  
  sess.run(tf.global_variables_initializer())
  saver = tf.train.Saver()
  
                 
  lVal = lValInit
  for i in range(trainIter):
          
        batch = dataSet.train.next_batch(bSize)

        feed_train={data: batch[0], true: batch[1], reco: batch[2], learningRate: lVal}
                 
        if(useTensorboard):
            _, merged_summary_result_train = sess.run([train_step, merged_summary],
                                          feed_dict=feed_train)
        else:
            sess.run([train_step],feed_dict=feed_train)
        
        
        if i % 10 == 0:
                  lVal= 1e-5 + 0.5*( np.cos( i *  np.pi/trainIter )+1)*lValInit
        if i % 50 == 0:
            
            batchTest = dataSet.test.next_batch(bSize) 
                        
            feed_test={data: batchTest[0], true: batchTest[1], reco: batchTest[2], learningRate: lVal}
            
            if(useTensorboard):                        
                loss_result, psnr_result, merged_summary_result = sess.run([loss, psnrEval, merged_summary],
                              feed_dict=feed_test)
        
                train_summary_writer.add_summary(merged_summary_result_train, i)
                test_summary_writer.add_summary(merged_summary_result, i)
                
            else:
                loss_result, psnr_result = sess.run([loss, psnrEval],
                              feed_dict=feed_test)
        


            print('iter={}, loss={}, psnr={}'.format(i, loss_result,psnr_result))  
            

        
        
  checkPointName = filePath + experimentName + '_final'
  save_path = saver.save(sess, checkPointName)
  print("Model saved in file: %s" % save_path)
  sess.close()
  return


   

def evaluation(dataSet,A,netType,experimentName,filePath,
             lossFunc = 'l2_loss',
             bSize = 4,
             trainIter = 50001,
             LGSiter = 10,
             useTensorboard = True,
             lValInit=1e-3,
             regParam = 1e-3,
             finalRelu = True,
             computeQuantMeas = False):
# Import data
  
   #Learning rate (could be changed)  
  
  sess = tf.InteractiveSession()    
  
  
  imSizeOut=dataSet.test.true.shape
  imSizeIn=dataSet.test.data.shape
  # Create the model
  data = tf.placeholder(tf.float32, [None, imSizeIn[1],imSizeIn[2],1])
  reco = tf.placeholder(tf.float32, [None, imSizeOut[1],imSizeOut[2],1])
  true = tf.placeholder(tf.float32, [None, imSizeOut[1],imSizeOut[2],1])
  
  
  
  # Build the graph for the network used
  if netType == 'Unet':
      print('Using 2D Unet')
      x_out = Unet(reco,imSizeOut,bSize)
  elif netType == 'resUnet':
      print('Using residual 2D Unet')
      x_out = reco + Unet(reco,imSizeOut,bSize)    
  elif netType == 'fullyLearned':
      x_1 = denseBlock(data,data.shape,imSizeOut,bSize,regParam)
      x_out = resBlock(x_1,imSizeOut,bSize)
  elif netType == 'LGS':
      print('Using LGS: ' + str(LGSiter) + ' iterations')
      x_out, grad, residual = LGS(data,A,imSizeIn,imSizeOut,bSize,LGSiter)
      
  else:
      print('Not supported network')
      return
  
  '''Final ReLU layer: TURN OFF IF NOT NEEDED'''
  if(finalRelu):
      print('Final maximum projection by ReLU')  
      x_out = tf.nn.relu(x_out)
      

  with tf.name_scope('optimizer'):
         
         if lossFunc == 'l2_loss':
             loss = tf.nn.l2_loss(x_out-true)/tf.nn.l2_loss(true)
             print('Using l2-loss')
         elif lossFunc == 'l1_loss':
             loss = tf.reduce_sum(tf.abs(x_out-true))/tf.reduce_sum(tf.abs(true))
             print('Using l1-loss')
         else: 
             print('Not supported loss function')
             return
         psnrEval = psnr(x_out, true)    

#         lossComb = loss + tf.losses.get_regularization_loss()
#         learningRate=tf.constant(1e-3) # This is an init, can be changed
#         train_step = tf.train.AdamOptimizer(learningRate).minimize(lossComb)


  
  sess.run(tf.global_variables_initializer())
  saver = tf.train.Saver()
  checkPointName = filePath + experimentName + '_final'
  saver.restore(sess,checkPointName)
  
  evalIter = int(np.floor(imSizeOut[0]/bSize))
  
  
  
  if(computeQuantMeas):
      recAll   = np.zeros([evalIter,imSizeOut[1],imSizeOut[2]])
      lossAll  = np.zeros(evalIter)
      psnrAll  = np.zeros(evalIter)
      ssimAll  = np.zeros(evalIter)
      for i in range(evalIter):
              
            bStart = int(i*bSize)
            bEnd   = int((i+1)*bSize)
            feed_eval={data: dataSet.test.data[bStart:bEnd], true: dataSet.test.true[bStart:bEnd], 
                        reco: dataSet.test.reco[bStart:bEnd]}
                             
                            
            loss_result, psnr_result, x_rec = sess.run([loss,psnrEval,x_out],feed_dict=feed_eval)
            
            recAll[i] = np.reshape(x_rec,[imSizeOut[1],imSizeOut[2]])
            lossAll[i] = loss_result
            psnrAll[i] = psnr_result
            ssimAll[i] = ssim(np.reshape(x_rec,[imSizeOut[1],imSizeOut[2]]), np.reshape(dataSet.test.true[bStart:bEnd] ,[imSizeOut[1],imSizeOut[2]])  )
            
    
      sess.close()
      return lossAll, psnrAll, ssimAll, recAll
  else:
            
      print('Evaluating first batch')
      bStart = int(0)
      bEnd   = int(bSize)
      feed_eval={data: dataSet.test.data[bStart:bEnd], true: dataSet.test.true[bStart:bEnd], 
                        reco: dataSet.test.reco[bStart:bEnd]}
                             
                            
      x_rec = sess.run(x_out,feed_dict=feed_eval)
            
      sess.close()
      return x_rec
      
      
      

def default_tensorboard_dir(name):
    tensorboard_dir = tensorboardDefaultDir 
    if not exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)
    return tensorboard_dir


def summary_writers(name, expName , session=None):
    if session is None:
        session = tf.get_default_session()
    
    dname = default_tensorboard_dir(name)
    

    
    test_summary_writer = tf.summary.FileWriter(dname + '/test_' + expName, session.graph)
    train_summary_writer = tf.summary.FileWriter(dname + '/train_' + expName)
    
    return test_summary_writer, train_summary_writer