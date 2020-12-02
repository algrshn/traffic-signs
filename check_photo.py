import numpy as np
import tensorflow as tf
import skimage.transform
import skimage.io
import argparse
import sys

import lib


parser = argparse.ArgumentParser()
parser.add_argument('--path_to_trained_model',type=str, help='Example: --path_to_trained_model sample_results/979.')
parser.add_argument('--imgfile',type=str, help='Example: --imgfile val_img/26.jpg.')
args=parser.parse_args()

if(not args.path_to_trained_model):
    print("\n-------\nYou must specify the path to a trained model. The program will now terminate.\n-------\n")
    sys.exit()
    
if(not args.imgfile):
    print("\n-------\nYou must specify an image file to be checked. The program will now terminate.\n-------\n")
    sys.exit()


img=skimage.io.imread(args.imgfile, as_gray=True)
img_resized=skimage.transform.resize(img,[32,32])

Xnp=np.zeros([1,32,32,1])
Xnp[0,:,:,0]=img_resized[:,:]

X=tf.convert_to_tensor(Xnp,dtype=tf.dtypes.float32)

p=lib.predict(X, args.path_to_trained_model)

res=""

if(p[0]==1):
    res="Added Lane"
elif(p[0]==2):
    res="Left Curve"
elif(p[0]==3):
    res="Right Curve"
elif(p[0]==4):
    res="Keep Right"
elif(p[0]==5):
    res="Lane Ends"
elif(p[0]==6):
    res="Merge"
elif(p[0]==7):
    res="Pedestrian Crossing"
elif(p[0]==8):
    res="School"
elif(p[0]==9):
    res="Signal Ahead"
elif(p[0]==10):
    res="Stop"
elif(p[0]==11):
    res="Stop Ahead"
elif(p[0]==12):
    res="Yield"
elif(p[0]==13):
    res="We could not recognize any of 12 US traffic signs we are capable of recognizing"
    
print("\n\n\n")
print(res)
print("\n\n\n")

