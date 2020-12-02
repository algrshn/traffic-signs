import tensorflow as tf
import numpy as np
import time
import argparse
import sys


import lib

parser = argparse.ArgumentParser()
parser.add_argument('--save_to_folder',type=str, help='Name of the folder to save model to during the train process. This folder goes into the directory saved_models. On completion of each epoch the script will be creating a separate subfolder for this epoch and will be placing the current state of the model in this subfolder. Thus if you specify the parameter --save_to_folder mytestrun, then the result of training after the 0th epoch will be saved to saved_models/mytestrun/0/, after 50th epoch to saved_models/mytestrun/50/ and so on.')
parser.add_argument('--batch_size',type=int, help='Optional. Example: --batch_size 1024. If this parameter is omitted, then the model will be trained with batch size of 4096.')
parser.add_argument('--epochs',type=int, help='Optional. Total number of epochs to train the model for. Example: --epochs 300. If this parameter is omitted, then the model will be trained for 1000 epochs.')
parser.add_argument('--dropout_rates',type=float,nargs=2, help='Optional. Tuple specifying dropout rates for 6th and 7th layers. Example: --dropout_rates 0.4 0.15. If omitted defaults to (0.5, 0.2).')
args=parser.parse_args()

if(not args.save_to_folder):
    print("\n-------\nYou must specify a folder name your trained model will be saved to. The program will now terminate.\n-------\n")
    sys.exit()
if(not args.batch_size):
    print("\n-------\nYou have not specified the batch size. Default value 4096 will be used.\n-------\n")
    batch_size=4096
else:
    batch_size=args.batch_size

if(not args.epochs):
    print("\n-------\nYou have not specified the number of epochs for training. Default value 1000 will be used.\n-------\n")
    epochs=1000
else:
    epochs=args.epochs

if(not args.dropout_rates):
    print("\n-------\nYou have not specified the dropout rates for 6th and 7th layers. Default values (0.5, 0.2) will be used.\n-------\n")
    dropout_rates=(0.5, 0.2)
else:
    dropout_rates=args.dropout_rates    

lib.load_dataset()

Xn=np.load("dataset/X.npy")
Yn=np.load("dataset/Y.npy")

X=tf.convert_to_tensor(Xn,dtype=tf.dtypes.float32)
Y=tf.convert_to_tensor(Yn,dtype=tf.dtypes.float32)



model=lib.Model()

N=Xn.shape[0]

indices=tf.range(N)

opt = tf.keras.optimizers.Adam()

start=time.time()
for epoch in range(epochs):
    
    shuffled_indices=tf.random.shuffle(indices)
    X_shuffled=tf.gather(X, shuffled_indices)
    Y_shuffled=tf.gather(Y, shuffled_indices)
    
    tot=0
    for iter1 in range(N//batch_size):
        Xb=X_shuffled[iter1*batch_size:(iter1+1)*batch_size,:,:,:]
        Yb=Y_shuffled[iter1*batch_size:(iter1+1)*batch_size,:]
        model.train_step(opt,Xb,Yb,dropout_rates)
        cost_b=model.loss(Xb,Yb,dropout_rates=(0,0))
        tot=tot+cost_b
    cost=tot/(N//batch_size)
    print("Epoch# %2i | cost = %2.6f" % (epoch,cost.numpy()))
          
    foldertosave="saved_models/" + args.save_to_folder + "/" + str(epoch) +"/"
    tf.saved_model.save(model, foldertosave)


end=time.time()
print("Execution time: {0:.0f} sec".format(end-start))


    
    