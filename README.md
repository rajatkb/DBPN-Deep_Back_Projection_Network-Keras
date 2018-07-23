# DBPN-Keras : Deep Back Projection Network
Implementation of paper Deep Back Projection Network paper  

Paper: https://arxiv.org/abs/1803.02735

<img src="https://camo.githubusercontent.com/4abd3a8873a79014d3d09b5cb1d7cb0c19d75ecd/687474703a2f2f7777772e746f796f74612d74692e61632e6a702f4c61622f44656e7368692f69696d2f6d656d626572732f6d7568616d6d61642e68617269732f70726f6a656374732f4442504e2e706e67">  

## Observation  
So i have tried this model with a 8x factor with 10 DBPN blocks with Dense connection i.e D-DBPN. The results on div2k were fine not really the most exciting. Possibly because of training time and batch size the convergance was skewed.
1. Gives better results than RDN visually by a notch (Personal take)
2. The idea of calculating error from upscalling and downscaliing is was fresh and can be used in other architecture  
3. Have similar structure as dense net by using global feature learning  
4. Suffers from excessive information loss and bad recreation. As observed from other model also if the required amount of information is not available in the downscaled image then 

## Usage  

### Data  

In the script SRIP DATA BUILDER just point the below line to whichever folder you have your training images in 
```
d = DATA(folder='./BSDS200_Padded' , patch_size = int(scale * 32))
```
It will create few npy files which training data. The training_patch_Y.npy contains the original data in patches  
of 64x64 or 128x128 how you specify. In case the image provided for training is not of proper shape, the script will pad them by black borders and convert the image in patches and save them in the npy file. trainig_patches_2x contains the 2x  
downscaled patches and respcitvely 4x and 8x contains the same. The noisy one is in case if you want to have  
them trained on grainy and bad image sample for robust training. The lines for noisy are commented , just uncomment and maybe at them in if else.  

```
p , r, c = DATA.patchify(img  , scale = 1)
```
By default patch size is 64. So  
<b> img </b>: Input image of shape (H,W,C)  
<b> scale </b>: scale determines the size of patch = patch_size / scale  
<b> returns </b>: list of patches , number of rows , number of columns  

### Training  

Just run the main.py , there are a few arguments in below format  

```
(rajat-py3) sanchit@Alpha:~/rajatKb/SRResNet$ python main.py -h
usage: main.py [-h] [--to TRYOUT] [--ep EPOCHS] [--bs BATCH_SIZE]
               [--lr LEARNING_RATE] [--gpu GPU] [--chk CHK] [--sample SAMPLE]
               [--scale SCALE] [--data FOLDER] [--test_image TEST_IMAGE]
               [--test_only TEST_ONLY]

control RDNSR

optional arguments:
  -h, --help            show this help message and exit
  --to TRYOUT
  --ep EPOCHS
  --bs BATCH_SIZE
  --lr LEARNING_RATE
  --gpu GPU
  --chk CHK
  --sample SAMPLE
  --scale SCALE
  --data FOLDER
  --test_image TEST_IMAGE
  --test_only TEST_ONLY
```
The file will be looking for few images to generate test results after every try out run. So you can provide  
your own test image or use one provided.  

test_only activate when you dont need to train , so you just point to model checkpoint get the results.

### Dataset  
used the DIV2K dataset  

## Results  
In the results folder. Otherwise here is one from DIV2k

8x zoomed H.R image 
<img src="https://i.imgur.com/qYsmPiG.jpg">  

Actual H.R imaage  
<img src="https://i.imgur.com/N6UjQNi.jpg">  

There is also a zoomed version where i took the the H.R image and zoomed it with the network. The result is far better than any fixed kernel based approach  but the generated image is a 16K image of 135mb size. So can't upload it. You can test it urself with the --zoom True arguement when running main.py

## Note
The model was trained with multi gpu mode since the DDBPN is big. Like really big when trained on 8x. So i had to train on multi gpu and mistakenly i save the weights in multi gpu mode. I will see if i can conert them to single mode. Just some working out the script should make it possible. So will post weights after a while



