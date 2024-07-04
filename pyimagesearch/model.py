# import the necessary packages
from pyimagesearch import config
from torch.nn import ConvTranspose2d
from torch.nn import Conv2d
from torch.nn import MaxPool2d
from torch.nn import Module
from torch.nn import ModuleList
from torch.nn import ReLU
from torchvision.transforms import CenterCrop
from torch.nn import functional as F
import torch

class Block(Module):
    def __init__(self, inChannels, outChannels):
        super().__init__()
        # store the convolution and RELU layers
        self.conv1 = Conv2d(inChannels, outChannels, 3)
        self.relu = ReLU()
        self.conv2 = Conv2d(outChannels, outChannels, 3)
    def forward(self, x):
    # apply CONV => RELU => CONV block to the inputs and return it
        return self.conv2(self.relu(self.conv1(x)))
    
class Encoder(Module):
    def __init__(self, channels=(3, 16, 32, 64)):  #3 no. of channels (RGB) #16, 32, 64 -> doubling the channel dimensions
        super().__init__()
        # store the encoder blocks and maxpooling layer
        self.encBlocks = ModuleList(
            [Block(channels[i], channels[i + 1])
                for i in range(len(channels) - 1)])
        self.pool = MaxPool2d(2)
        
    def forward(self, x):
        # initialize an empty list to store the intermediate outputs
        blockOutputs = []
        # loop through the encoder blocks
        for block in self.encBlocks:
            # pass the inputs through the current encoder block, store
            # the outputs, and then apply maxpooling on the output
            x = block(x)
            blockOutputs.append(x)
            x = self.pool(x)
        # return the list containing the intermediate outputs
        return blockOutputs # 
    
class Decoder(Module):
    def __init__(self, channels=(64, 32, 16)):
        super().__init__()
        # initialize the number of channels, upsampler blocks, and
        # decoder blocks
        self.channels = channels
        # use the ConvTranspose2d layer to upsample the spatial dimension (i.e., height and width)
        # of the feature maps by a factor of 2. In addition, the layer also reduces the number of channels by a factor of 2.
        self.upconvs = ModuleList(
            [ConvTranspose2d(channels[i], channels[i + 1], 2, 2) 
                for i in range(len(channels) - 1)])
        self.dec_blocks = ModuleList(
            [Block(channels[i], channels[i + 1])
                for i in range(len(channels) - 1)])
    def forward(self, x, encFeatures): 
        # encFeatures -> list of intermediate outputs from the encoder
        # loop through the number of channels
        for i in range(len(self.channels) - 1):
            # pass the inputs through the upsampler blocks
            x = self.upconvs[i](x) #the i-th intermediate feature map from the encoder with our current output x from the upsampling block
            # crop the current features from the encoder blocks,
            # concatenate them with the current upsampled features, and pass the concatenated output through the current decoder block
            # we need to ensure that the spatial dimensions of encFeatures[i] and x match. To accomplish this, we use crop function
            encFeat = self.crop(encFeatures[i], x)
            x = torch.cat([x, encFeat], dim=1)
            x = self.dec_blocks[i](x)
        # return the final decoder output
        return x
    def crop(self, encFeatures, x):
        # grab the dimensions of the inputs, and crop the encoder
        # features to match the dimensions
        (_, _, H, W) = x.shape
        encFeatures = CenterCrop([H, W])(encFeatures)
        # return the cropped features
        return encFeatures

class UNet(Module):
    def __init__(self, encChannels=(3, 16, 32, 64),  #3 is channel RGB, # double the feature channels
        decChannels=(64, 32, 16),
        #  nbclasses-> the number of channels in our output segmentation map, where we have one channel for each class.
        # Since we are working with two classes (i.e., binary classification),
        #  we keep a single channel and use thresholding for classification
        nbClasses=1, retainDim=True,
        outSize=(config.INPUT_IMAGE_HEIGHT,  config.INPUT_IMAGE_WIDTH)):
        super().__init__()
        # initialize the encoder and decoder
        self.encoder = Encoder(encChannels)
        self.decoder = Decoder(decChannels)
        # initialize the regression head and store the class variables
        self.head = Conv2d(decChannels[-1], nbClasses, 1)
        self.retainDim = retainDim
        self.outSize = outSize
        
    def forward(self, x):
        # grab the features from the encoder
        encFeatures = self.encoder(x)
        # pass the encoder features through decoder making sure that
        # their dimensions are suited for concatenation
        # encFeatures[::-1] list contains the feature map outputs in reverse order (i.e., from the last to the first encoder block).
        # on the decoder side, we will be utilizing the encoder feature maps starting from the last encoder block output to the first.
        decFeatures = self.decoder(encFeatures[::-1][0], 
                      encFeatures[::-1][1:])
        # we pass the output of the final encoder block (i.e., encFeatures[::-1][0]) and the feature map outputs of all intermediate encoder blocks
        # (i.e., encFeatures[::-1][1:]) to the decoder. The output of the decoder is stored as decFeatures
        
        # pass the decoder features through the regression head to obtain the segmentation mask
        map = self.head(decFeatures)
        # check to see if we are retaining the original output
        # dimensions and if so, then resize the output to match them
        if self.retainDim:
            map = F.interpolate(map, self.outSize)
        # return the segmentation map
        return map
        
        
'''
#print("in model.py for u-net")
class Block(Module):
    def __init__(self, inChannels, outChannels):
        super().__init__()
        # store the convolution and RELU layers
        self.conv1 = Conv2d(inChannels, outChannels, 3)
        self.relu = ReLU()
        self.conv2 = Conv2d(outChannels, outChannels, 3)
    def forward(self, x):
    # apply CONV => RELU => CONV block to the inputs and return it
        return self.conv2(self.relu(self.conv1(x)))
    
class Encoder(Module):
    def __init__(self, channels=(3,64,128,256,512,1024)):  #3 no. of channels (RGB) #16, 32, 64 -> doubling the channel dimensions
        super().__init__()
        # store the encoder blocks and maxpooling layer
        self.encBlocks = ModuleList(
            [Block(channels[i], channels[i + 1])
                for i in range(len(channels) - 1)])
        self.pool = MaxPool2d(2)
        
    def forward(self, x):
        # initialize an empty list to store the intermediate outputs
        blockOutputs = []
        # loop through the encoder blocks
        for block in self.encBlocks:
            # pass the inputs through the current encoder block, store
            # the outputs, and then apply maxpooling on the output
            x = block(x)
            blockOutputs.append(x)
            x = self.pool(x)
        # return the list containing the intermediate outputs
        return blockOutputs
    
class Decoder(Module):
    def __init__(self, channels=(1024, 512, 256, 128, 64)):
        super().__init__()
        # initialize the number of channels, upsampler blocks, and
        # decoder blocks
        self.channels = channels
        # use the ConvTranspose2d layer to upsample the spatial dimension (i.e., height and width)
        # of the feature maps by a factor of 2. In addition, the layer also reduces the number of channels by a factor of 2.
        self.upconvs = ModuleList(
            [ConvTranspose2d(channels[i], channels[i + 1], 2, 2) 
                for i in range(len(channels) - 1)])
        self.dec_blocks = ModuleList(
            [Block(channels[i], channels[i + 1])
                for i in range(len(channels) - 1)])
    def forward(self, x, encFeatures): 
        # encFeatures -> list of intermediate outputs from the encoder
        # loop through the number of channels
        for i in range(len(self.channels) - 1):
            # pass the inputs through the upsampler blocks
            x = self.upconvs[i](x) #the i-th intermediate feature map from the encoder with our current output x from the upsampling block
            # crop the current features from the encoder blocks,
            # concatenate them with the current upsampled features, and pass the concatenated output through the current decoder block
            # we need to ensure that the spatial dimensions of encFeatures[i] and x match. To accomplish this, we use crop function
            encFeat = self.crop(encFeatures[i], x)
            x = torch.cat([x, encFeat], dim=1)
            x = self.dec_blocks[i](x)
        # return the final decoder output
        return x
    def crop(self, encFeatures, x):
        # grab the dimensions of the inputs, and crop the encoder
        # features to match the dimensions
        (_, _, H, W) = x.shape
        encFeatures = CenterCrop([H, W])(encFeatures)
        # return the cropped features
        return encFeatures

class UNet(Module):
    def __init__(self, encChannels=(3,64,128,256,512,1024),  #3 is channel RGB, # double the feature channels
        decChannels=(1024, 512, 256, 128, 64),
        #  nbclasses-> the number of channels in our output segmentation map, where we have one channel for each class.
        # Since we are working with two classes (i.e., binary classification),
        #  we keep a single channel and use thresholding for classification
        nbClasses=1, retainDim=True,
        outSize=(config.INPUT_IMAGE_HEIGHT,  config.INPUT_IMAGE_WIDTH)):
        super().__init__()
        # initialize the encoder and decoder
        self.encoder = Encoder(encChannels)
        self.decoder = Decoder(decChannels)
        # initialize the regression head and store the class variables
        self.head = Conv2d(decChannels[-1], nbClasses, 1)
        self.retainDim = retainDim
        self.outSize = outSize
        
    def forward(self, x):
        # grab the features from the encoder
        encFeatures = self.encoder(x)
        # pass the encoder features through decoder making sure that
        # their dimensions are suited for concatenation
        # encFeatures[::-1] list contains the feature map outputs in reverse order (i.e., from the last to the first encoder block).
        # on the decoder side, we will be utilizing the encoder feature maps starting from the last encoder block output to the first.
        decFeatures = self.decoder(encFeatures[::-1][0], 
                      encFeatures[::-1][1:])
        # we pass the output of the final encoder block (i.e., encFeatures[::-1][0]) and the feature map outputs of all intermediate encoder blocks
        # (i.e., encFeatures[::-1][1:]) to the decoder. The output of the decoder is stored as decFeatures
        
        # pass the decoder features through the regression head to obtain the segmentation mask
        map = self.head(decFeatures)
        # check to see if we are retaining the original output
        # dimensions and if so, then resize the output to match them
        if self.retainDim:
            map = F.interpolate(map, self.outSize)
        # return the segmentation map
        return map
   
'''
