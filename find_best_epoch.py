import numpy as np
import tensorflow as tf
import argparse
import sys

import lib

parser = argparse.ArgumentParser()
parser.add_argument('--folder_to_load',type=str, help='Example: --folder_to_load sample_results')
parser.add_argument('--epochs',type=int, help='Total number of epochs in the run you are checking. Example: --epochs 1000.')
args=parser.parse_args()

if(not args.folder_to_load):
    print("\n-------\nYou must specify a folder containing trained models. The program will now terminate.\n-------\n")
    sys.exit()
    
if(not args.epochs):
    print("\n-------\nYou must specify total number of epochs to be checked. The program will now terminate.\n-------\n")
    sys.exit()


def predict(X,thr,foldertoload):

    N=len(X)

    loaded = tf.saved_model.load(foldertoload)

    model=lib.Model()

    model.b1=loaded.b1
    model.w1=loaded.w1
    model.b3=loaded.b3
    model.w3=loaded.w3
    model.b6=loaded.b6
    model.w6=loaded.w6
    model.b7=loaded.b7
    model.w7=loaded.w7

    Z8=model(X)
    A8=tf.sigmoid(Z8)

    A8n=A8.numpy()
    
    a8max=np.amax(A8n,axis=1)
    a8argmax=np.argmax(A8n,axis=1)+1
    
    p=np.zeros([N])
    
    for i in range(N):
        if(a8max[i]>thr):
            p[i]=a8argmax[i]
        else:
            p[i]=13
    return p




X_irrelevant_np=np.load("val_npy/photos_irrelevant.npy")
X_no_signs_np=np.load("val_npy/photos_no_signs.npy")
X_signs_np=np.load("val_npy/photos_signs.npy")

CA=np.load("val_npy/correct_answers.npy")

X_irrelevant=tf.convert_to_tensor(X_irrelevant_np,dtype=tf.dtypes.float32)
X_no_signs=tf.convert_to_tensor(X_no_signs_np,dtype=tf.dtypes.float32)
X_signs=tf.convert_to_tensor(X_signs_np,dtype=tf.dtypes.float32)

best_score=1000
best_epoch=0
for j in range(args.epochs):
    foldertoload='saved_models/' + args.folder_to_load + '/' + str(j) +'/'

    p_irrelevant=predict(X_irrelevant,thr=0.5,foldertoload=foldertoload)
    p_no_signs=predict(X_no_signs,thr=0.5,foldertoload=foldertoload)

    p=predict(X_signs,thr=0.5,foldertoload=foldertoload)

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

    score50=irr_wrong+ns_wrong+2*s_wrong+0.5*s_df
    
    
    p_irrelevant=predict(X_irrelevant,thr=0.75,foldertoload=foldertoload)
    p_no_signs=predict(X_no_signs,thr=0.75,foldertoload=foldertoload)

    p=predict(X_signs,thr=0.75,foldertoload=foldertoload)

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

    score75=irr_wrong+ns_wrong+2*s_wrong+0.5*s_df
    
    score=(score50+score75)/2
    
    print("Epoch# %4i | score = %3.1f" %(j,score))
    
    if(score<best_score):
        best_score=score
        best_epoch=j

print("Best average score: {}".format(best_score))
print("Best epoch: {}".format(best_epoch))                 
 