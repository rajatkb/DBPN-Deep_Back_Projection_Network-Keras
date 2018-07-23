import cv2
import os

'''

This is the standard DENSE NET with Global and Local Feature fusion + learning and 
upscalling using subpixel convolution. That is just better way of space filling by
learnable parameters in the deconvolution layer which has gradient during
backpropagation instead of the stadard 0 gradient.

Now we add the infamous PERCEPTUAL Loss in a more controlled fashion so that
content is maximum and color is kept but the texture i.e the strokes are transfered

We will try with bothh VGG and INCEPTION as loss factor
This one will rely on stanard VGG

'''


'''
train where the train data is build somewhere and you are pointing to it
python main.py --ep 1 --to 200 --bs 16 --lr 0.0001 --gpu 1 --sample 16384 --data ../Data --test_image 0016.png

testing on the data

 python main.py --test_only True --chk 98 --test_image 1072_927.png

use the above to run the file. this is the orinal configuration as per the paper

'''

import argparse
parser = argparse.ArgumentParser(description='control RDNSR')
parser.add_argument('--to', action="store",dest="tryout", default=200)
parser.add_argument('--ep', action="store",dest="epochs", default=1)
parser.add_argument('--bs', action="store",dest="batch_size", default=20)
parser.add_argument('--lr', action="store",dest="learning_rate", default=0.0001)
parser.add_argument('--gpu', action="store",dest="gpu", default=-1)
parser.add_argument('--chk',action="store",dest="chk",default=-1)
parser.add_argument('--sample',action='store',dest="sample",default=16368)
# parser.add_argument('--test_sample', action="store",dest="test_sample",default=190)
# parser.add_argument('--scale', action='store' , dest = 'scale' , default = 8)
parser.add_argument('--data', action='store' , dest = 'folder' , default = '../Data_8x')
parser.add_argument('--test_image', action = 'store' , dest = 'test_image' , default = 'test.png')
parser.add_argument('--test_only' , action = 'store', dest = 'test_only' , default = False)
parser.add_argument('--zoom' , action = 'store' , dest = 'zoom' , default = False)
# parser.add_argument('--l1_factor' , action = 'store' , dest= 'l1_factor' , default = 1)
# parser.add_argument('--lambda_content', action = 'store' , dest = 'lambda_content' , default = 0.1)
# parser.add_argument('--lambda_style', action = 'store' , dest = 'lambda_style' , default = 0.1)

values = parser.parse_args()
learning_rate = float(values.learning_rate)
batch_size = int(values.batch_size)
epochs = int(values.epochs)
tryout = int(values.tryout)
gpu=int(values.gpu)
sample = int(values.sample)
# test_sample = int(values.test_sample)
# scale = int(values.scale)
scale = 8
folder = values.folder
test_only = values.test_only
zoom = values.zoom
# l1_factor = float(values.l1_factor)
# lambda_content = float(values.lambda_content)
# lambda_style = float(values.lambda_style)

if gpu >= 0:
    os.environ["CUDA_VISIBLE_DEVICES"]=str(gpu)

chk = int(values.chk)

import sys
import numpy as np
import matplotlib.pyplot as plt
from SRIP_DATA_BUILDER import DATA
from keras.models import Model
from keras.layers import ZeroPadding2D , PReLU , Input,MaxPool2D,Conv2DTranspose ,Conv2D , Add, Dense , AveragePooling2D , UpSampling2D , Reshape , Flatten , Subtract , Concatenate , Lambda
from keras.optimizers import Adam 
from keras.callbacks import LearningRateScheduler
from keras import backend as k
from keras.applications import vgg16
from keras.utils import multi_gpu_model
from keras.applications.imagenet_utils import preprocess_input
import tensorflow as tf
from keras.utils import plot_model
from subpixel import Subpixel
from PIL import Image
import cv2


def PSNRLossnp(y_true,y_pred):
		return 10* np.log(255*2 / (np.mean(np.square(y_pred - y_true))))


def SSIMnp(y_true , y_pred):
    u_true = np.mean(y_true)
    u_pred = np.mean(y_pred)
    var_true = np.var(y_true)
    var_pred = np.var(y_pred)
    std_true = np.sqrt(var_true)
    std_pred = np.sqrt(var_pred)
    c1 = np.square(0.01*7)
    c2 = np.square(0.03*7)
    ssim = (2 * u_true * u_pred + c1) * (2 * std_pred * std_true + c2)
    denom = (u_true ** 2 + u_pred ** 2 + c1) * (var_pred + var_true + c2)
    return ssim / denom


def SSIM( y_true,y_pred):
    u_true = k.mean(y_true)
    u_pred = k.mean(y_pred)
    var_true = k.var(y_true)
    var_pred = k.var(y_pred)
    std_true = k.sqrt(var_true)
    std_pred = k.sqrt(var_pred)
    c1 = k.square(0.01*7)
    c2 = k.square(0.03*7)
    ssim = (2 * u_true * u_pred + c1) * (2 * std_pred * std_true + c2)
    denom = (u_true ** 2 + u_pred ** 2 + c1) * (var_pred + var_true + c2)
    return ssim / denom




def PSNRLoss(y_true, y_pred):
        return 10* k.log(255**2 /(k.mean(k.square(y_pred - y_true))))

class DBN:
    def L1_loss(self , y_true , y_pred):
    	   return k.mean(k.abs(y_true - y_pred))
    
    #def L1_plus_PSNR_loss(self,y_true, y_pred):
        #return self.L1_loss(y_true , y_pred) - 0.0001*PSNRLoss(y_true , y_pred)
    
    def UpBlocks(self,L,filters=12 , channel = 64):
        # x = ZeroPadding2D(padding=(2,2))(x)
        L = Conv2D(filters=channel , kernel_size = 1 , strides=1,kernel_initializer='glorot_uniform' ,padding='valid' )(L)
        L = PReLU(alpha_initializer='zero' , shared_axes=[1, 2 ])(L)

        Ht = Conv2DTranspose(filters=channel , kernel_size=filters  , strides=(8,8) ,  kernel_initializer='glorot_uniform' ,padding='same')(L)
        Ht = PReLU(alpha_initializer='zero' , shared_axes=[1, 2 ])(Ht)
        
        Lt = Conv2D(filters= channel , kernel_size=filters, strides=(8 , 8) , kernel_initializer='glorot_uniform' , padding='same')(Ht)
        Lt = PReLU(alpha_initializer='zero' , shared_axes=[1, 2 ])(Lt)
        
        et = Subtract()([Lt ,L])

        H1t = Conv2DTranspose(filters=channel , kernel_size=filters  , strides=(8,8) ,  kernel_initializer='glorot_uniform' ,padding='same')(et)
        H1t = PReLU(alpha_initializer='zero' , shared_axes=[1, 2 ])(H1t)
        om = Add()([Ht,H1t])

        return om

    def DownBlocks(self , H , filters=12 , channel = 64):

    	H = Conv2D(filters=channel , kernel_size = 1 , strides=1,kernel_initializer='glorot_uniform' ,padding='same' )(H)
    	H = PReLU(alpha_initializer='zero' , shared_axes=[1, 2 ])(H)

    	Lt = Conv2D(filters= channel , kernel_size=filters, strides=(8 , 8) , kernel_initializer='glorot_uniform' , padding='same')(H)
    	Lt = PReLU(alpha_initializer='zero' , shared_axes=[1, 2 ])(Lt)
    	
    	Ht = Conv2DTranspose(filters=channel , kernel_size=filters  , strides=(8,8) ,  kernel_initializer='glorot_uniform' ,padding='same')(Lt)
    	Ht = PReLU(alpha_initializer='zero' , shared_axes=[1, 2 ])(Ht)

    	et = Subtract()([Ht,H])

    	L1t = Conv2D(filters= channel , kernel_size=filters , strides=(8 , 8) , kernel_initializer='glorot_uniform' , padding='same')(et)
    	L1t = PReLU(alpha_initializer='zero' , shared_axes=[1, 2 ])(L1t)

    	om = Add()([Lt , L1t])

    	return om
        
    def visualize(self):
            plot_model(self.model, to_file='model.png' , show_shapes = True)
   			 
    def get_model(self):
        	return self.model

    def __init__(self , channel = 3 , lr=0.0001 , patch_size=32 , T_count=7,n0 = 256 , nr=64 ,chk = -1 , scale = 8):
            self.channel_axis = 3
            self.patch_size = patch_size
            self.scale = scale

            inp = Input(shape = (patch_size , patch_size , channel) , name='dbn_input')

            #feature extraction layer
            x = Conv2D(filters=n0, kernel_size=(3,3), strides=(1, 1), padding='same' , kernel_initializer='glorot_uniform' )(inp)
            x = PReLU(alpha_initializer='zero' , shared_axes=[1, 2 ])(x)
            # feature smashing
            x = Conv2D(filters=nr, kernel_size=(1,1), strides=(1, 1), padding='same' , kernel_initializer='glorot_uniform')(x)
            x = PReLU(alpha_initializer='zero' , shared_axes=[1, 2 ])(x)

            up_projection_blocks = []
            down_projection_blocks = []

            x = self.UpBlocks(x , channel = nr)
            up_projection_blocks.append(x)
            x = self.DownBlocks(x , channel = nr)
            down_projection_blocks.append(x)
            

            for i in range(1,T_count-1):
            	x = self.UpBlocks(x , channel = nr)
            	up_projection_blocks.append(x)
            	x = Concatenate()(up_projection_blocks)
            	x =self.DownBlocks(x , channel = 64)
            	down_projection_blocks.append(x)
            	x = Concatenate()(down_projection_blocks)
            
            x = self.UpBlocks(x , channel = nr)
            up_projection_blocks.append(x)
            x = Concatenate()(up_projection_blocks)
            x = Conv2D(filters = 3 , kernel_size = 3 , strides = 1 , padding='same' , kernel_initializer='glorot_uniform')(x)

            self.model = Model(inputs=inp , outputs = x)

            ## multi gpu setting
            
            if gpu < 0:
               self.model = multi_gpu_model(self.model, gpus=3)

            adam = Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=None)
            self.model.compile(loss='mse', optimizer=adam , metrics=[PSNRLoss,SSIM])

            
            if chk >=0 :
                print("loading existing weights !!!")
                self.model.load_weights('model_'+str(scale)+'x_iter'+str(chk)+'.h5')

    def fit(self , X , Y ,batch_size=16 , epoch = 1000 ):
            # with tf.device('/gpu:'+str(gpu)):
            zero = np.zeros(Y.shape[0])    
            hist = self.model.fit(x = X, y = Y , batch_size = batch_size , verbose =1 , nb_epoch=epoch)
            return hist.history
    

if __name__ == '__main__':

    CHANNEL = 3

    DATA = DATA(folder = folder , patch_size = int(scale * 32))

    out_patch_size =  DATA.patch_size 
    inp_patch_size = int(out_patch_size/ scale)
    if not test_only:
        DATA.load_data(folder=folder)
        if scale == 2:
            x = DATA.training_patches_2x
        elif scale == 4:
            x = DATA.training_patches_4x
        elif scale == 8:
            x = DATA.training_patches_8x
        y = DATA.training_patches_Y
    
    net = DBN(lr = learning_rate,scale = scale , chk = chk )
    if not test_only:
        net.visualize()
        net.get_model().summary()

    image_name = values.test_image
    try:
        img = cv2.imread(image_name)
        img = cv2.cvtColor(img , cv2.COLOR_BGR2RGB)
    except cv2.error as e:
        print("Bad image path check the name or path !!")
        exit()

    R = DATA.patch_size - img.shape[0] % DATA.patch_size
    C = DATA.patch_size - img.shape[1] % DATA.patch_size
    img = np.pad(img, [(0,R),(0,C),(0,0)] , 'constant')
    Image.fromarray(img).save("test_image_padded.png")
    lr_img = cv2.resize(img , (int(img.shape[1]/scale),int(img.shape[0]/scale)) ,cv2.INTER_CUBIC)
    Image.fromarray(lr_img).save("test_"+str(scale)+"x_lr_padded.png")
    hr_img_bi = cv2.resize(lr_img ,(int(img.shape[1]),int(img.shape[0])),cv2.INTER_CUBIC)
    Image.fromarray(hr_img_bi).save("test_"+str(scale)+"x_hr_bicubic_padded.png")


    p , r , c = DATA.patchify(lr_img,scale=scale) 

    if not test_only:
        for i in range(chk+1,tryout):
            print("tryout no: ",i)   
            
            samplev = np.random.random_integers(0 , x.shape[0]-1 , sample)
           
            net.fit(x[samplev] , y[samplev] , batch_size , epochs )
            
            net.get_model().save_weights('model_'+str(scale)+'x_iter'+str(i)+'.h5')
            g = net.get_model().predict(np.array(p))
            gen = DATA.reconstruct(g , r , c , scale=1)
            Image.fromarray(gen).save("test_"+str(scale)+"x_gen_"+str(i)+".png")
            print("Reconstruction Gain:", PSNRLossnp(img , gen))
    else:
        
        g = net.get_model().predict(np.array(p))        
        gen = DATA.reconstruct(g , r , c , scale=1)
        
        gen = gen[0:-R , 0:-C , :]
        img = img[0:-R , 0:-C , :]
        
        Image.fromarray(gen).save("test_"+str(scale)+"x_gen_.png")
        
        if zoom:
           z , r , c = DATA.patchify(img , scale = scale)
           gz = net.get_model().predict(np.array(z))
           genz = DATA.reconstruct(gz , r , c , scale=1)
           Image.fromarray(genz).save("zoomed_test_sample.png")
        
        print("Reconstruction Gain:", PSNRLossnp( img, gen))
