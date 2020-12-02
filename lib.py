import tensorflow as tf
import numpy as np
import os
import hashlib
import sys
from subprocess import run

class Model(tf.Module):
    
    def __init__(self):
        self.b1=tf.Variable(tf.zeros(shape=[1,1,1,6]))
        self.w1=tf.Variable(tf.random.normal(shape=[5,5,1,6],stddev=0.28,dtype=tf.dtypes.float32))
        self.b3=tf.Variable(tf.zeros(shape=[1,1,1,16]))
        self.w3=tf.Variable(tf.random.normal(shape=[5,5,6,16],stddev=0.12,dtype=tf.dtypes.float32))
        self.b6=tf.Variable(tf.zeros(shape=[65,1]))
        self.w6=tf.Variable(tf.random.normal(shape=[65,1024],stddev=0.044,dtype=tf.dtypes.float32))
        self.b7=tf.Variable(tf.zeros(shape=[13,1]))
        self.w7=tf.Variable(tf.random.uniform(shape=[13,65],minval=-0.28,maxval=0.28,dtype=tf.dtypes.float32))
        
        
    def __call__(self,X,dropout_rates=(0,0)):
               
        dropout_rate6=dropout_rates[0]
        dropout_rate7=dropout_rates[1]
            
        
        N=len(X)
        Z2=self.b1+tf.nn.convolution(input=X,filters=self.w1,padding='SAME')
        A2=tf.nn.elu(Z2)
        A3=tf.nn.pool(A2,window_shape=[2,2],pooling_type='MAX',strides=[2,2],padding="SAME")
        Z4=self.b3+tf.nn.convolution(input=A3,filters=self.w3,padding='SAME')
        A4=tf.nn.elu(Z4)
        A5=tf.nn.pool(A4,window_shape=[2,2],pooling_type='MAX',strides=[2,2],padding="SAME")
        A6=tf.reshape(A5,shape=[N,1024])
        A6_drp=tf.nn.dropout(A6,rate=dropout_rate6)
        Z7=tf.matmul(A6_drp,tf.transpose(self.w6))+tf.transpose(self.b6)
        A7=tf.nn.elu(Z7)
        A7_drp=tf.nn.dropout(A7,rate=dropout_rate7)
        Z8=tf.matmul(A7_drp,tf.transpose(self.w7))+tf.transpose(self.b7)

        
        return Z8
    
    def loss(self,X,Y,dropout_rates):

        N=len(X)
        Z8=self.__call__(X,dropout_rates)
        
        regZ8=tf.nn.relu(Z8)+tf.math.log(1+tf.math.exp(-tf.math.abs(Z8)))
        YTA=tf.matmul(tf.transpose(Y),Z8)-tf.matmul(tf.ones_like(tf.transpose(Y)),regZ8)
        J=(-1/N)*tf.linalg.trace(YTA)
        
        return J
    
    def train_step(self,opt,X,Y,dropout_rates):
        current_loss = lambda : self.loss(X, Y, dropout_rates)
        varslist = [self.b1,self.w1,self.b3,self.w3,self.b6,self.w6,self.b7,self.w7]
        opt.minimize(current_loss, var_list=varslist)
        

def predict(X,path_to_trained_model,thr=0.5):

    N=len(X)

    loaded = tf.saved_model.load('saved_models/' + path_to_trained_model + '/')

    model=Model()

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


def check_if_gdown_installed():
    
    try:
        run(['gdown','-V'])
    except FileNotFoundError:
        print("You will need to install gdown to download the dataset. The program will terminate now.")
        sys.exit()


def load_dataset():
    
    Xnpy_md5='e4f2c2ea03918e7cab515ca01fbddb3c'
    Ynpy_md5='b727356f468137c6f959c96bc3ce7949'
    
    Xnpy_url='https://drive.google.com/uc?id=164UxnaPrtAJ7XNZvFmq-1NoruBWm1v2C'
    Ynpy_url='https://drive.google.com/uc?id=1m0bHHw9RWUaJbuNtTAxL12kP90YkUlsA'
    
    Xnpy_need_to_load='yes'
    Ynpy_need_to_load='yes'
    
    if(os.path.exists('dataset/X.npy')):
        with open("dataset/X.npy", "rb") as f:
            file_hash = hashlib.md5()
            chunk = f.read(8192)
            while chunk:
                file_hash.update(chunk)
                chunk = f.read(8192)
        if(file_hash.hexdigest()==Xnpy_md5):
            Xnpy_need_to_load='no'
            
    if(os.path.exists('dataset/Y.npy')):
        with open("dataset/Y.npy", "rb") as f:
            file_hash = hashlib.md5()
            chunk = f.read(8192)
            while chunk:
                file_hash.update(chunk)
                chunk = f.read(8192)
        if(file_hash.hexdigest()==Ynpy_md5):
            Ynpy_need_to_load='no'
            
    if(Xnpy_need_to_load=='yes'):
        check_if_gdown_installed()
        os.system('gdown ' + Xnpy_url +' --output dataset/')

        
    if(Ynpy_need_to_load=='yes'):
        check_if_gdown_installed()
        os.system('gdown ' + Ynpy_url +' --output dataset/')

    