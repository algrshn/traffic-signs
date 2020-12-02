import numpy as np
import tensorflow as tf
import argparse
import sys

import lib

parser = argparse.ArgumentParser()
parser.add_argument('--path_to_trained_model',type=str, help='Example: --path_to_trained_model sample_results/979.')
parser.add_argument('--thr',type=float, help='Optional. Prediction threshold. Example: --thr 0.4. If the highest output  probability for an image is higher than this threshold, then the image will be classified. If lower, then the predictor will return "Detection failed". Too low threshold value might result in wrong classifications in cases where higher threshold would force the predictor to acknowledge that it does not know the answer (which is better than giving a wrong answer). Too agressively high threshold would lead to too many "Detection failed" results where the predictor could have provided the right answers. Default value is 0.5.')
args=parser.parse_args()

if(not args.path_to_trained_model):
    print("\n-------\nYou must specify a path to trained model. The program will now terminate.\n-------\n")
    sys.exit()
    
if(not args.thr):
    print("\n-------\nYou have not specified the prediction threshold. Default value 0.5 will be used.\n-------\n")
    thr=0.5
else:
    thr=args.thr


lib.load_dataset()

Xn=np.load("dataset/X.npy")
Yn=np.load("dataset/Y.npy")

N=Xn.shape[0]

X=tf.convert_to_tensor(Xn,dtype=tf.dtypes.float32)

p=lib.predict(X,args.path_to_trained_model,thr)

y=np.zeros([N])
y01=np.sum(Yn,axis=1)
yargmax=np.argmax(Yn,axis=1)+1

for i in range(N):
    if(y01[i]==1):
        y[i]=yargmax[i]
        

sp_corr=0
sp_df=0
sp_wrong=0

snp_df=0
snp_wrong=0


for i in range(N):
    if(y[i]==13):
        if(p[i]==13):
            snp_df=snp_df+1
        else:
            snp_wrong=snp_wrong+1
    else:
        if(p[i]==13):
            sp_df=sp_df+1
        elif(p[i]==y[i]):
            sp_corr=sp_corr+1
        else:
            sp_wrong=sp_wrong+1
            
sp_total=sp_corr+sp_df+sp_wrong
snp_total=snp_df+snp_wrong

print("\n\n\n-------------\n\n\n")
print("Signs present in the dataset:\n\n")
print("Correct: %5i / %5i" %(sp_corr,sp_total))
print("Detection failed: %5i / %5i" %(sp_df,sp_total))
print("Wrong: %5i / %5i" %(sp_wrong,sp_total))
print("\n\n\n-------------\n\n\n")
print("Signs not present in the dataset:\n\n")
print("Detection failed: %5i / %5i" %(snp_df,snp_total))
print("Wrong: %5i / %5i" %(snp_wrong,snp_total))