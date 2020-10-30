''' Class file using pytorch Learned image recontruction for PAT
https://doi.org/10.1117/1.JBO.25.11.112903
Written 2020 by Andreas Hauptmann, University of Oulu and UCL'''

import numpy as np
import torch
from torch import nn
from torch import optim
import tensorboardX

import h5py
import os 
from os.path import exists
from skimage.measure import compare_ssim as ssim

FLAGS = None


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
    
  data = data.reshape(num_images,1, rows, cols)
  return data

class DataSet(object):

  def __init__(self, data, true,reco):
    """Construct a DataSet"""

    assert data.shape[0] == true.shape[0], (
        'images.shape: %s labels.shape: %s' % (data.shape,
                                                 true.shape))
    data = data.reshape(data.shape[0],
                            data.shape[1],data.shape[2],data.shape[3])
    true = true.reshape(true.shape[0],
                            true.shape[1],true.shape[2],true.shape[3])
    reco = reco.reshape(reco.shape[0],
                            reco.shape[1],reco.shape[2],reco.shape[3])
    
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
  test_data   = np.zeros([test_true.shape[0],1,tLen,test_true.shape[2]])
  train_data  = np.zeros([train_true.shape[0],1,tLen,train_true.shape[2]])
  
  test_reco   = np.zeros(test_true.shape)
  train_reco  = np.zeros(train_true.shape)

  data_sets.train = DataSet(train_data, train_true,train_reco)
  data_sets.test = DataSet(test_data, test_true,test_reco)

  return data_sets







   
def double_conv(in_channels, out_channels):
    return nn.Sequential(
       nn.Conv2d(in_channels, out_channels, 3, padding=1),
       nn.BatchNorm2d(out_channels),       
       nn.ReLU(inplace=True),
       nn.Conv2d(out_channels, out_channels, 3, padding=1),
       nn.BatchNorm2d(out_channels),
       nn.ReLU(inplace=True))

class UNet(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
               
        self.dconv_down1 = double_conv(n_in, 32)
        self.dconv_down2 = double_conv(32, 64)
        self.dconv_down3 = double_conv(64, 128)

       
        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)        
        self.xUp2  = nn.ConvTranspose2d(128,64,2,stride=2,padding=0)
        self.xUp1  = nn.ConvTranspose2d(64,32,2,stride=2,padding=0)
        

        self.dconv_up2 = double_conv(64 + 64, 64)
        self.dconv_up1 = double_conv(32 + 32, 32)
        self.conv_last = nn.Conv2d(32, n_out, 1)
        
        self.stepsize = nn.Parameter(torch.zeros(1, 1, 1, 1))

    def forward(self, x):
        inp = x
        
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)
        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)
        conv3 = self.dconv_down3(x)
        
        
        x = self.xUp2(conv3)                
        x = torch.cat([x, conv2], dim=1)      
        x = self.dconv_up2(x)
        x = self.xUp1(x)        
        
        x = torch.cat([x, conv1], dim=1)         
        x = self.dconv_up1(x)
        update = self.conv_last(x)

        return inp + self.stepsize * update
    

def summary_image_impl(writer, name, tensor, it):
    image = tensor[0, 0]
    image = (image - torch.min(image)) / (torch.max(image) - torch.min(image))
    writer.add_image(name, image, it, dataformats='HW')



def summary_image(writer, name, tensor, it, window=False):
    summary_image_impl(writer, name + '/full', tensor, it)
    if window:
        summary_image_impl(writer, name + '/window', (tensor), it)        
        
        

def summaries(writer, result, fbp, true, loss, it, do_print=False):
    residual = result - true
    squared_error = residual ** 2
    mse = torch.mean(squared_error)
    maxval = torch.max(true) - torch.min(true)
    psnr = 20 * torch.log10(maxval) - 10 * torch.log10(mse)
    
    
    relative = torch.mean((result - true) ** 2) / torch.mean((fbp - true) ** 2)

    
    if do_print:
        print(it, mse.item(), psnr.item(), relative.item())

    writer.add_scalar('loss', loss, it)
    writer.add_scalar('psnr', psnr, it)
    writer.add_scalar('relative', relative, it)

    summary_image(writer, 'result', result, it)
    summary_image(writer, 'true', true, it)
    summary_image(writer, 'fbp', fbp, it)
    summary_image(writer, 'squared_error', squared_error, it)
    summary_image(writer, 'residual', residual, it)
    summary_image(writer, 'diff', result - fbp, it)

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
    # bSize = 1
    device = 'cuda'
    if(useTensorboard):
        train_writer = tensorboardX.SummaryWriter(comment="/train")
        test_writer = tensorboardX.SummaryWriter(comment="/test")
    
    
    model = UNet(n_in=1, n_out=1).to(device)
    loss_train = nn.MSELoss()
    loss_test = nn.MSELoss()
    
    
    optimizer = optim.Adam(model.parameters(), lr=lValInit)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, trainIter)
    
    
    for it in range(trainIter):
        scheduler.step()    
        
        batch = dataSet.train.next_batch(bSize)
        # feed_train={data: batch[0], true: batch[1], reco: batch[2], learningRate: lVal}
         
         
        images = torch.from_numpy(batch[1]).float().to(device)
        # projs = torch.from_numpy(data).float().to(device)
        reco =  torch.from_numpy(batch[2]).float().to(device)
        
        model.train()    
        optimizer.zero_grad()
        
        output = model(reco)
        
        loss = loss_train(output, images)
        loss.backward()
        optimizer.step()
    
      
        
        if it % 25 == 0:
            if(useTensorboard):
                summaries(train_writer, output, reco, images, loss, it, do_print=False)
            model.eval()
            batch = dataSet.test.next_batch(bSize)
         
         
            test_images = torch.from_numpy(batch[1]).float().to(device)
            # projs = torch.from_numpy(data).float().to(device)
            reco_test = torch.from_numpy(batch[2]).float().to(device)
        
            outputTest = model(reco_test)
            lossTest = loss_test(outputTest, test_images)
            
            if(useTensorboard):
                summaries(test_writer, outputTest, reco_test, test_images, lossTest, it, do_print=True)
    
    
    torch.save(model.state_dict(), filePath + experimentName)
    return




def evaluation(dataSet,A,netType,experimentName,filePath,
             lossFunc = 'l2_loss',
             bSize = 4,
             trainIter = 50001,
             LGSiter = 10,
             useTensorboard = True,
             lValInit=1e-3,
             regParam = 1e-3,
             finalRelu = True):

    device = 'cuda'
    model = UNet(n_in=1, n_out=1).to(device)
   
    model.load_state_dict(torch.load(filePath + experimentName))
    model.eval()

    batch = dataSet.test.next_batch(bSize)
                  
    test_images = torch.from_numpy(batch[1]).float().to(device)
            # projs = torch.from_numpy(data).float().to(device)
    reco_test = torch.from_numpy(batch[2]).float().to(device)
        
    outputTest = model(reco_test)
    reco = outputTest.cpu().detach().numpy()

    return reco
