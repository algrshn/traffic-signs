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

X_irrelevant_np=np.load("val_npy/photos_irrelevant.npy")
X_no_signs_np=np.load("val_npy/photos_no_signs.npy")
X_signs_np=np.load("val_npy/photos_signs.npy")

CA=np.load("val_npy/correct_answers.npy")

X_irrelevant=tf.convert_to_tensor(X_irrelevant_np,dtype=tf.dtypes.float32)
X_no_signs=tf.convert_to_tensor(X_no_signs_np,dtype=tf.dtypes.float32)
X_signs=tf.convert_to_tensor(X_signs_np,dtype=tf.dtypes.float32)

p_irrelevant=lib.predict(X_irrelevant,args.path_to_trained_model, thr=thr)
p_no_signs=lib.predict(X_no_signs,args.path_to_trained_model, thr=thr)

p=lib.predict(X_signs,args.path_to_trained_model, thr=thr)

irr_df=0
irr_wrong=0

for i in range(50):
    if(p_irrelevant[i]==13):
        irr_df=irr_df+1
    else:
        irr_wrong=irr_wrong+1


ns_df=0
ns_wrong=0

for i in range(130):
    if(p_no_signs[i]==13):
        ns_df=ns_df+1
    else:
        ns_wrong=ns_wrong+1
        
s_corr=0        
s_df=0
s_wrong=0

for i in range(175):
    if(p[i]==13):
        s_df=s_df+1
    else:
        if(p[i]==CA[i]):
            s_corr=s_corr+1
        else:
            s_wrong=s_wrong+1

score=irr_wrong+ns_wrong+2*s_wrong+0.5*s_df
                       
print("\n\n\nTotally irrelevant images\n\n\n")
print("Detection failed: %3i" % (irr_df) )
print("Wrong: %3i" % (irr_wrong) ) 
print("-----------------\n\n\nSigns not in the dataset\n\n\n")
print("Detection failed: %3i" % (ns_df) )
print("Wrong: %3i" % (ns_wrong) )
print("-----------------\n\n\nSigns in the dataset\n\n\n")
print("Correct: %3i" % (s_corr) )
print("Detection failed: %3i" % (s_df) )
print("Wrong: %3i" % (s_wrong) )
print("\n\n\n------------------\n\n\n") 
print("SCORE = %2.1f" % (score))
print("\n\n\n\n\n\n")        