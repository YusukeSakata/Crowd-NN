
# coding: utf-8

# In[ ]:

#直接データ読み込んでいる、そのうち引数とる形に整形する


# In[1]:

import os
import pandas as pd
from PIL import Image
import numpy as np
import sklearn
from operator import add,mul
import functools
import copy
import subprocess
import MultiGLAD as MGLAD
import dawid_skene as DS
import math
import functools
import matplotlib.pyplot as plt
import random


# In[2]:

#チェイナー関連のimport
from __future__ import print_function

import argparse

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions
from chainer.datasets import tuple_dataset
from chainer import serializers
#hyperropt関連のimport
import hyperopt
from hyperopt import fmin, tpe, hp
from hyperopt.pyll import scope


# In[3]:

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Perceptron
from sklearn.model_selection import StratifiedKFold


# In[4]:

import make_dummy_data

#メール関係
import smtplib
from email.mime.text import MIMEText
from email.utils import formatdate


# In[5]:

def df_init_pics(df_data):
    n_task = len(df_data.object.value_counts())
    n_fit = len(df_data.feature.value_counts())
    n_wif = df_data.feature.value_counts()[0] // n_task
    return n_task * n_fit, n_task,n_wif,n_fit


# In[6]:

def binary_converter (x):
    if x == 1:
        return 1
    else:
        return 0
    
def string_to_label (x, tri_key):
    if x == "Yes":
        return 1
    elif x == "No" and tri_key == 1:
        return -1
    else:
        return 0


# In[7]:

#features,labels,n_feature,n_fit,n_loop,n_worker
def train_test_from_tensor (tensor,labels,train_rate):
    
    #　CNNの入力のshapeは(nsample, channel, filter_size) の3次元、ここの処理は引数生成の方に渡した
    #X_list = (np.float32(np.array_split(tensor, n_sample))).reshape(n_sample, 1, n_feature * n_worker)
    #Y_list = np.int32(list(np.array_split(labels, n_split * 2)))
    
    X_dev, X_val, Y_dev, Y_val = train_test_split(tensor, labels, train_size=train_rate, random_state=0)
    train ,test = tuple_dataset.TupleDataset(X_dev, Y_dev.T), tuple_dataset.TupleDataset(X_val, Y_val.T)
    return train,test


# In[8]:

#train,testに加えて、その配列位置を渡したらその位置で区切ったやつ返してくれる
#features,labels,n_feature,n_fit,n_loop,n_worker
def train_test_from_indices_tensor (X_list,Y_list,train_indices,test_indices):
    
    #　CNNの入力のshapeは(nsample, channel, filter_size) の3次元
    # X_list = (np.float32(np.array_split(tensor, n_sample))).reshape(n_sample, 1, n_feature * n_worker)
    #Y_list = np.int32(list(np.array_split(labels, n_split * 2)))
    X_dev, X_val, Y_dev, Y_val = X_list[train_indices] ,X_list[test_indices], Y_list[train_indices], Y_list[test_indices]
    train ,test = tuple_dataset.TupleDataset(X_dev, Y_dev.T), tuple_dataset.TupleDataset(X_val, Y_val.T)
    return train,test


# In[9]:

def hstack (X_data,n_loop):
    tmp = X_data[1]
    for i in range (1,n_loop):
        tmp = np.hstack((tmp,X_data[i]))
    return tmp


# In[10]:

def lambda_if (x,y):
    if x > y:
        return 0
    else:
        return 1

def mglad_arrange (mglad_tmp_label):
    zero_label = np.array(mglad_tmp_label.label1)
    one_label = np.array(mglad_tmp_label.label2)
    mglad_tmp_label = list(map(lambda_if,zero_label,one_label))
    mglad_label = np.array_split(mglad_tmp_label,4000)
    return mglad_label

def mglad_conti_arrange(mglad_tmp_label):
    mglad_tmp_label = np.array(mglad_tmp_label.label2)
    mglad_label = np.array_split(mglad_tmp_label,4000)
    return mglad_label


# In[11]:

def data_selecter(data_key):#データのマイリスト作る、0:pics01、1:reviews01、2:smiles01、3:articles01、4:pics03
    #5:smiles03、#6:articles03、#7:reviews03
    if data_key == 4:
        df_data =  pd.read_csv('labels_201712.csv').sort_values(by=["object","feature"], ascending=[True,True])
        n_loop = 1
        name = "pics03"
        label_name = "sisley"
    elif data_key == 5:
        df_data =  pd.read_csv('labels_smiles03.csv').sort_values(by=["object","feature"], ascending=[True,True])
        n_loop = 1
        name = "smiles03"
        label_name = "spontaneous"
    elif data_key == 6:
        df_data =  pd.read_csv('labels_articles03.csv').sort_values(by=["object","feature"], ascending=[True,True])
        n_loop = 1
        name = "articles03"
        label_name = "p"
    elif data_key == 7:
        df_data =  pd.read_csv('labels_reviews03.csv').sort_values(by=["object","feature"], ascending=[True,True])
        n_loop = 1
        name = "reviews03"
        label_name = "_t_"
    elif data_key == 8:
        df_data =  pd.read_csv('labels_pics01.csv').sort_values(by=["object","feature"], ascending=[True,True])
        n_loop = 1
        name = "pics01"
        label_name = "sisley"
    elif data_key == 9:
        df_data =  pd.read_csv('labels_smiles01.csv').sort_values(by=["object","feature"], ascending=[True,True])
        n_loop = 1
        name = "smiles01"
        label_name = "spontaneous"
    elif data_key == 10:
        df_data =  pd.read_csv('labels_articles01.csv').sort_values(by=["object","feature"], ascending=[True,True])
        n_loop = 1
        name = "articles01"
        label_name = "p"
    elif data_key == 11:
        df_data =  pd.read_csv('labels_reviews01.csv').sort_values(by=["object","feature"], ascending=[True,True])
        n_loop = 1
        name = "reviews01"
        label_name = "_t_"
    return df_data,n_loop,name,label_name


# In[12]:

def df_to_worker_array (split_df,n_task,n_fit,n_wif):
    worker_array = split_df[0].iloc[0,3]
    for i in range (n_task):
        for j in range (n_fit):
            tf_id = i * n_fit + j
            for k in range (n_wif):
                if split_df[tf_id].iloc[k,3] in worker_array:
                    continue
                else:
                    worker_array = np.append(worker_array,split_df[tf_id].iloc[k,3])
    return worker_array


# In[13]:

def df_to_tensor_tri (split_df, worker_array ,n_task, n_fit, n_wif,tri_key):#タスク毎に整列させた三値のやつを作る
    n_worker,n_feature= len(worker_array),n_task * n_fit
    label_matrix = np.zeros([n_feature, n_worker])
    for i in range (n_task):
        for j in range (n_fit):
            tf_id = i * n_fit + j#全体の何番目かで管理して
            for k in range (n_wif):
                worker_id = np.where(split_df[tf_id].iloc[k,3] == worker_array)#ワーカのid検索してその位置に
                label_matrix[tf_id][worker_id] = string_to_label(split_df[tf_id].iloc[k,4], tri_key)#ラベルを(1,-1)に変換して埋め込む
    return label_matrix


# In[14]:

def df_to_CNN_arg(split_df,n_task,n_fit,n_wif,tri_key):
    n_task_quarter = int(n_task  / 4)
    n_feature_quarter = n_task_quarter * n_fit
    worker_array = df_to_worker_array (split_df,n_task,n_fit,n_wif)#ワーカの配列作って
    n_worker = len(worker_array)#ワーカの数取っといて
    label_matrix = df_to_tensor_tri(split_df, worker_array ,n_task, n_fit, n_wif, tri_key)#テンソル作って
    tmp = copy.copy(label_matrix[n_feature_quarter : n_feature_quarter * 2])#test,trainの順番になる様並び替え
    label_matrix[n_feature_quarter : n_feature_quarter * 2] = copy.copy(label_matrix[n_feature_quarter * 2: n_feature_quarter * 3])
    label_matrix[n_feature_quarter * 2: n_feature_quarter * 3] = copy.copy(tmp)
    Y_labels = np.zeros(n_task)#綺麗に並んでるから01を直で作りに行く,mon_tes,sis_tes,mon_tra,sis_traの順
    Y_labels[n_task_quarter : n_task_quarter * 2] = np.ones(n_task_quarter)
    Y_labels[n_task_quarter * 3 : n_task] = np.ones(n_task_quarter)
    #最後に引数の形に整形する、（サンプルの数、チャンネル数、ワーカ人数×1サンプルの特徴数）
    X_list = (np.float32(np.array_split(label_matrix, n_task))).reshape(n_task, 1, n_fit * n_worker)
    Y_list = np.int32(Y_labels)
    return X_list,Y_list, n_worker


# In[15]:

def map_posi_neg (x):
    if x > 0:
        return 1
    else:
        return -1
    
def per_maj_iter(x):
    if x < 1 and x > -1:
        x = 0
    return x


# In[16]:

def tensor_to_DS (CNN_split_X,n_wif):
    tmp = []
    ini_labels = list(map(lambda x: list(map(lambda y: map_posi_neg (sum(y)),x)),CNN_split_X))
    labels = copy.copy(ini_labels)
    for i in range (50):
        tmp = copy.copy(labels)
        params = copy.copy(labels_to_params(CNN_split_X,labels))
        true_per = labels_to_true_per(labels)
        cont_labels = copy.copy(params_to_labels(CNN_split_X,params,true_per,n_wif))
        labels =  np.array(list(map(lambda x:list(map(prop_to_pos_neg,x)),labels)))
        if not any(sum(np.array(labels) - np.array(tmp))):
            break
    return params


# In[17]:

def labels_to_params(CNN_split_X,labels):
    worker_tf_list = np.array(list(map(lambda x,y: list(map(lambda a:map_tensor_to_labels(a,y),x.T)),CNN_split_X,labels)))
    Sensitivity = map_in(sum, worker_tf_list.T[1]) / map_in(sum, worker_tf_list.T[0])#positiveの正解率
    Specificity = map_in(sum, worker_tf_list.T[3]) / map_in(sum, worker_tf_list.T[2])#negativeの正解率
    return np.array([Sensitivity,Specificity])

def labels_to_true_per(labels):
    true_per = sum(np.array(list(map(lambda x: sum(map(binary_converter,x)),labels)))) / (len(labels) * len(labels[0]))#positiveの割合
    return true_per

def map_tensor_to_labels(CsXiw,liw):#100次元,CNN_split_X_in_worker、labels_in_worker
    true = CsXiw[CsXiw == liw]
    n_pos = sum(CsXiw[CsXiw > 0])
    n_pos_true = sum(true[true > 0])
    n_neg = sum(CsXiw[CsXiw < 0])
    n_neg_true = sum(true[true < 0])
    return n_pos,n_pos_true,n_neg,n_neg_true

def map_in (func,array):#配列の中の配列をmapする処理
    return np.array(list(map(lambda x: func(x),array)))


# In[18]:

def params_to_labels(CNN_split_X,params,true_per,n_wif):#パラメータから"1"になる確率を計算する
    labels = list(map(
        lambda x: list(map(
            lambda a: pow(functools.reduce(mul,map(
                lambda i, j:bernoulli(i,j,true_per),
                a,params.T)),1/sum(abs(a))),
        x)),
        CNN_split_X))
    return np.array(labels)

def bernoulli (CNN_split_X_elem, param_elem, true_per):
    if CNN_split_X_elem == 0:
        return 1
    p,a,b,y = true_per,param_elem[0],param_elem[1],binary_converter(CNN_split_X_elem)
    labels_elem = p * (a ** y) * ((1 - a) ** (1 - y)) / (p * (a ** y) * ((1 - a) ** (1 - y)) +(1 - p) * ((1 - b) ** y) * (b ** (1 - y)))
    return labels_elem

def prop_to_pos_neg (x):
    if x > 0.5:
        return 1
    else:
        return -1


# In[19]:

class CNN(chainer.Chain):

    def __init__(self, n_units, n_out,n_worker):
        super(CNN, self).__init__()
        with self.init_scope():
            dim, c_in, c_out, = 1,1,1
            #引数は、次元、入力チャネル、出力チャネル、フィルタサイズ、ストライド
            self.c1= L.ConvolutionND(dim, c_in, c_out,n_worker,stride = n_worker)
            #正則化してみる
            self.bn1 = L.BatchNormalization(c_out)
            # the size of the inputs to each layer will be inferred
            self.l1 = L.Linear(None, n_units)  # n_in -> n_units
            #self.l2 = L.Linear(None, n_units)  # n_units -> n_units
            #self.l3 = L.Linear(None, n_units)  # n_units -> n_units
            #self.l4 = L.Linear(None, n_units)  # n_units -> n_units
            self.lout = L.Linear(None, n_out)  # n_units -> n_out

    def __call__(self, x):
        c1 =  F.leaky_relu(self.c1(x))
      #  bn1 = self.bn1(c1)
        h1 = F.leaky_relu(self.l1(c1))
        #h2 = F.leaky_relu(self.l2(h1))
        #h3 = F.leaky_relu(self.l3(h2))
        #h4 = F.leaky_relu(self.l4(h3))
        return self.lout(h1)


# In[20]:

class C_CNN(chainer.Chain):#カスタマイズCNN

    def __init__(self, n_out,n_worker, n_units1, n_units2, n_units3, n_units4, n_units5, n_units6,
                 layer_num, c_out,activate, conv_key, dropout_key, n_fit):
        super(C_CNN, self).__init__()
        with self.init_scope():
            dim, c_in = 1,1
            #引数は、次元、入力チャネル、出力チャネル、フィルタサイズ、ストライド
            self.c1= L.ConvolutionND(dim, c_in, c_out,n_worker,stride = n_worker)
            # the size of the inputs to each layer will be inferred
            #特徴毎にワーカの能力が変わるモデル、引数は、入力チャネル、出力チャネル、フィルタサイズ、ストライド
            self.c2= L.Convolution2D(c_in, c_out,(n_fit,n_worker),stride = n_worker)
            # the size of the inputs to each layer will be inferred
            self.l1 = L.Linear(None, n_units1)  # n_in -> n_units
            self.l2 = L.Linear(None, n_units2)  # n_units -> n_units
            self.l3 = L.Linear(None, n_units3)  # n_units -> n_units
            self.l4 = L.Linear(None, n_units4)  # n_units -> n_units
            self.l5 = L.Linear(None, n_units4)  # n_units -> n_units
            self.l6 = L.Linear(None, n_units5)  # n_units -> n_units
            self.lout = L.Linear(None, n_out)  # n_units -> n_out
        self.layer_num = layer_num
        if activate == 'relu':
            self.act = F.relu
        else:
            self.act = F.leaky_relu
            
        if dropout_key == 1:#ここ0にしたらバグる問題解決してないから
            with chainer.using_config('train', True):
                self.dropout = F.dropout
        self.conv_key = conv_key
            
    def __call__(self, x):
        if self.conv_key == 1:
            c1 =  self.dropout(self.act(self.c1(x)))
        elif self.conv_key == 0:
            c1 = x
        else:
            c1 = self.dropout(self.act(self.c2(x)))
        h1 = self.dropout(self.act(self.l1(c1)))
        if self.layer_num == 2:
            return self.lout(h1)
        h2 = self.dropout(self.act(self.l2(h1)))
        if self.layer_num == 3:
            return self.lout(h2)
        h3 = self.dropout(self.act(self.l3(h2)))
        if self.layer_num == 4:
            return self.lout(h3)
        h4 = self.dropout(self.act(self.l4(h3)))
        if self.layer_num == 5:
            return self.lout(h4)
        h5 = self.dropout(self.act(self.l5(h4)))
        if self.layer_num == 6:
            return self.lout(h5)
        h6 = self.dropout(self.act(self.l6(h5)))
        if self.layer_num == 7:
            return self.lout(h6)


# In[21]:

def Tuning(params,features, labels,n_worker,name,train_rate,conv_key,gpu_key, dropout_key, n_fit):
    n_out = 2
    file_name = "C_CNN"

    n_units1 = params['n_units1']
    n_units2 = params['n_units2']
    n_units3 = params['n_units3']
    n_units4 = params['n_units4']
    n_units5 = params['n_units5']
    n_units6 = params['n_units6']
    layer_num = params['layer_num']
    activate = params['activate']
    epoch = params['epoch']
    batch_size = params['batch_size']
    c_out = params['c_out']

    #引数はn_out,n_worker, n_units1, n_units2, n_units3, n_units4, layer_num, c_out,activate
    model = L.Classifier(C_CNN(n_out, n_worker, n_units1, n_units2, n_units3, n_units4,n_units5, n_units6,
                               layer_num, c_out, activate,conv_key, dropout_key, n_fit))
    if gpu_key >= 0:
        # Make a specified GPU current
            chainer.cuda.get_device_from_id(gpu_key).use()
            model.to_gpu()  # Copy the model to the GPU
    
    
     # Setup an optimizer
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    # 自分のデータセット入れてみる
    train, test = train_test_from_tensor(features,labels,train_rate)

    train_iter = chainer.iterators.SerialIterator(train, batch_size)
    test_iter = chainer.iterators.SerialIterator(test, batch_size,
                                                 repeat=False, shuffle=False)

    # Set up a trainer
    updater = training.StandardUpdater(train_iter, optimizer, device = gpu_key)
    trainer = training.Trainer(updater, (epoch, 'epoch'), out='result_' + name)

    # Evaluate the model with the test dataset for each epoch
    trainer.extend(extensions.Evaluator(test_iter, model, device = gpu_key))

    # Dump a computational graph from 'loss' variable at the first iteration
    # The "main" refers to the target link of the "main" optimizer.
#     trainer.extend(extensions.dump_graph('main/loss'))

    # Write a log of evaluation statistics for each epoch
    trainer.extend(extensions.LogReport())

    # Save two plot images to the result dir
    if extensions.PlotReport.available():
        trainer.extend(
            extensions.PlotReport(['main/loss', 'validation/main/loss'],
                                  'epoch', file_name=(file_name +'loss.png')))
#         trainer.extend(
#             extensions.PlotReport(
#                 ['main/accuracy', 'validation/main/accuracy'],
#                 'epoch', file_name=(file_name + 'accuracy.png')))

    # Print selected entries of the log to stdout
    # Here "main" refers to the target link of the "main" optimizer again, and
    # "validation" refers to the default name of the Evaluator extension.
    # Entries other than 'epoch' are reported by the Classifier link, called by
    # either the updater or the evaluator.
#     trainer.extend(extensions.PrintReport(
#         ['epoch', 'main/loss', 'validation/main/loss',
#          'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))

    # Print a progress bar to stdout
   # trainer.extend(extensions.ProgressBar())

#     if ' ':
#         # Resume from a snapshot
#         chainer.serializers.load_npz(' ', trainer)

    # Run the training
    trainer.run()
    
    valid_data = trainer._extensions['PlotReport'].extension._data
    loss_data = [data for i, data in valid_data['validation/main/loss']]
    last_loss = list(reversed(loss_data))[0]
    return last_loss


# In[22]:

def Cross_Tuning(params,features, labels,n_worker,name,n_cv,train_rate,conv_key,gpu_key, dropout_key, n_fit):
    n_out = 2
    file_name = "C_CNN"

    n_units1 = params['n_units1']
    n_units2 = params['n_units2']
    n_units3 = params['n_units3']
    n_units4 = params['n_units4']
    n_units5 = params['n_units5']
    n_units6 = params['n_units6']
    layer_num = params['layer_num']
    activate = params['activate']
    epoch = params['epoch']
    batch_size = params['batch_size']
    c_out = params['c_out']
    
    if n_cv == 0:
        return Tuning(params,features, labels,n_worker,name,train_rate,conv_key,gpu_key, dropout_key, n_fit)
    last_loss = 0
    kf = StratifiedKFold(n_splits=n_cv,shuffle = True)
    for train_indices, test_indeices in kf.split(features,labels):

        #引数はn_out,n_worker, n_units1, n_units2, n_units3, n_units4, layer_num, c_out,activate
        model = L.Classifier(C_CNN(n_out, n_worker, n_units1, n_units2, n_units3, n_units4, 
                                   layer_num, c_out, activate, conv_key ,dropout_key, n_fit))
        
        if gpu_key >= 0:
        # Make a specified GPU current
            chainer.cuda.get_device_from_id(gpu_key).use()
            model.to_gpu()  # Copy the model to the GPU
    
    
         # Setup an optimizer
        optimizer = chainer.optimizers.Adam()
        optimizer.setup(model)

        # 自分のデータセット入れてみる
        train, test = train_test_from_indices_tensor(features,labels,train_indices,test_indeices)

        train_iter = chainer.iterators.SerialIterator(train, batch_size)
        test_iter = chainer.iterators.SerialIterator(test, batch_size,
                                                 repeat=False, shuffle=False)

        # Set up a trainer
        updater = training.StandardUpdater(train_iter, optimizer, device = gpu_key)
        trainer = training.Trainer(updater, (epoch, 'epoch'), out='result_' + name)

        # Evaluate the model with the test dataset for each epoch
        trainer.extend(extensions.Evaluator(test_iter, model, device = gpu_key))

        # Dump a computational graph from 'loss' variable at the first iteration
        # The "main" refers to the target link of the "main" optimizer.
#         trainer.extend(extensions.dump_graph('main/loss'))

        # Write a log of evaluation statistics for each epoch
        trainer.extend(extensions.LogReport())
        
        if extensions.PlotReport.available():
            trainer.extend(
                extensions.PlotReport(['main/loss', 'validation/main/loss'],
                                      'epoch', file_name=(file_name +'loss.png')))
#             trainer.extend(
#                 extensions.PlotReport(
#                     ['main/accuracy', 'validation/main/accuracy'],
#                     'epoch', file_name=(file_name + 'accuracy.png')))


        # Print selected entries of the log to stdout
        # Here "main" refers to the target link of the "main" optimizer again, and
        # "validation" refers to the default name of the Evaluator extension.
        # Entries other than 'epoch' are reported by the Classifier link, called by
        # either the updater or the evaluator.
#         trainer.extend(extensions.PrintReport(
#             ['epoch', 'main/loss', 'validation/main/loss',
#              'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))

        # Print a progress bar to stdout
       # trainer.extend(extensions.ProgressBar())

    #     if ' ':
    #         # Resume from a snapshot
    #         chainer.serializers.load_npz(' ', trainer)

        # Run the training
        trainer.run()
    
        valid_data = trainer._extensions['PlotReport'].extension._data
        loss_data = [data for i, data in valid_data['validation/main/loss']]
        last_loss += list(reversed(loss_data))[0]
    return last_loss


# In[23]:

class Tuning_Object:

    def __init__(self, Tuning_function,features, labels,n_worker,name,n_cv,
                 train_rate,conv_key,gpu_key,dropout_key, n_fit):
        self.Tuning_function = Tuning_function
        self.features = features
        self.labels = labels
        self.n_worker = n_worker
        self.name = name
        self.n_cv = n_cv
        self.train_rate = train_rate
        self.conv_key = conv_key
        self.gpu_key = gpu_key
        self.dropout_key = dropout_key
        self.n_fit = n_fit

    def __call__(self, params):
        return self.Tuning_function (params, self.features, self.labels, self.n_worker, self.name, self.n_cv,
                                     self.train_rate, self.conv_key, self.gpu_key, self.dropout_key, self.n_fit)


# In[24]:

def CNN_Tuning (Tuning_function,features, labels,n_worker,name,n_cv,train_rate,
                max_eval,conv_key,gpu_key,dropout_key, n_fit):
    params = {'n_units1': scope.int(hp.quniform('n_units1', 100, 300, 100)),
             'n_units2': scope.int(hp.quniform('n_units2', 100, 300, 100)),
             'n_units3': scope.int(hp.quniform('n_units3', 100, 300, 100)),
             'n_units4': scope.int(hp.quniform('n_units4', 100, 300, 100)),
             'n_units5': scope.int(hp.quniform('n_units5', 100, 300, 100)),
             'n_units6': scope.int(hp.quniform('n_units6', 100, 300, 100)),
             'layer_num': scope.int(hp.quniform('layer_num', 2, 7, 1)),
             'activate': hp.choice('activate',
                                         ('relu', 'leaky_relu')),
             'epoch': scope.int(hp.quniform('epoch', 50, 110, 10)),
             'batch_size' : scope.int(hp.quniform('batch_size', 40, 200, 40)),
             'c_out' : scope.int (hp.quniform('c_out', 20, 40, 10)),
             }
        
    tuning_object = Tuning_Object(Tuning_function,features, labels,n_worker,
                                  name,n_cv,train_rate,conv_key,gpu_key,dropout_key, n_fit)
    best = fmin(tuning_object, params, algo=tpe.suggest, max_evals = max_eval,
               rstate = np.random.RandomState(0))
    best = hyperopt.space_eval(params, best)
    return best


# In[25]:

def eval_CNN (params,train_features, train_labels,test_features,test_labels,
              n_worker,name,conv_key,gpu_key,dropout_key, n_fit):
    n_out = 2
    file_name = "eval_CNN"

    n_units1 = params['n_units1']
    n_units2 = params['n_units2']
    n_units3 = params['n_units3']
    n_units4 = params['n_units4']
    n_units5 = params['n_units5']
    n_units6 = params['n_units6']
    layer_num = params['layer_num']
    activate = params['activate']
    epoch = params['epoch']
    batch_size = params['batch_size']
    c_out = params['c_out']

    #引数はn_out,n_worker, n_units1, n_units2, n_units3, n_units4, layer_num, c_out,activate
    model = L.Classifier(C_CNN(n_out, n_worker, n_units1, n_units2, n_units3, n_units4, n_units5, n_units6,
                               layer_num, c_out, activate,conv_key, dropout_key, n_fit))
    if gpu_key >= 0:
        # Make a specified GPU current
            chainer.cuda.get_device_from_id(gpu_key).use()
            model.to_gpu()  # Copy the model to the GPU
    
     # Setup an optimizer
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    # 自分のデータセット入れてみる
    train ,test = tuple_dataset.TupleDataset(train_features, train_labels.T), tuple_dataset.TupleDataset(test_features, test_labels.T)

    train_iter = chainer.iterators.SerialIterator(train, batch_size)
    test_iter = chainer.iterators.SerialIterator(test, batch_size,
                                                 repeat=False, shuffle=False)

    # Set up a trainer
    updater = training.StandardUpdater(train_iter, optimizer, device = gpu_key)
    trainer = training.Trainer(updater, (epoch, 'epoch'), out='result_' + name)

    # Evaluate the model with the test dataset for each epoch
    trainer.extend(extensions.Evaluator(test_iter, model, device = gpu_key))

    # Dump a computational graph from 'loss' variable at the first iteration
    # The "main" refers to the target link of the "main" optimizer.
#     trainer.extend(extensions.dump_graph('main/loss'))

    # Write a log of evaluation statistics for each epoch
    trainer.extend(extensions.LogReport())

    # Save two plot images to the result dir
    if extensions.PlotReport.available():
        trainer.extend(
            extensions.PlotReport(
                ['main/accuracy', 'validation/main/accuracy'],
                'epoch', file_name=(file_name + 'accuracy.png')))

    # Print selected entries of the log to stdout
    # Here "main" refers to the target link of the "main" optimizer again, and
    # "validation" refers to the default name of the Evaluator extension.
    # Entries other than 'epoch' are reported by the Classifier link, called by
    # either the updater or the evaluator.
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss',
         'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))

    # Print a progress bar to stdout
   # trainer.extend(extensions.ProgressBar())

#     if ' ':
#         # Resume from a snapshot
#         chainer.serializers.load_npz(' ', trainer)

    # Run the training
    trainer.run()
    
    valid_data = trainer._extensions['PlotReport'].extension._data
    accuracy_data = [data for i, data in valid_data['validation/main/accuracy']]
    last_accuracy = list(reversed(accuracy_data))[0]
    return last_accuracy


# In[26]:

class Eval_Object:
    
    def __init__(self, Eval_function,train_features, train_labels,test_features, test_labels, 
                 n_worker,name,conv_key, gpu_key, dropout_key, n_fit):
        self.Eval_function = Eval_function
        self.train_features = train_features
        self.train_labels = train_labels
        self.test_features = test_features
        self.test_labels = test_labels
        self.n_worker = n_worker
        self.name = name
        self.conv_key =conv_key
        self.gpu_key = gpu_key
        self.dropout_key = dropout_key
        self.n_fit = n_fit

    def __call__(self, params):
        return self.Eval_function (params,self.train_features, self.train_labels,self.test_features, self.test_labels, 
                                   self.n_worker, self.name, self.conv_key, self.gpu_key, self.dropout_key, self.n_fit)


# In[71]:

def cross_valid (Tuning_function,Eval_function,features, labels,
                 n_worker,name,n_in_cv,train_rate,n_out_cv,max_eval,
                 conv_key,gpu_key,n_wif,dropout_key, n_fit):
    kf = StratifiedKFold(n_splits=n_out_cv,shuffle = True)
    accuracy,bests = [],[]
    for train_indices, test_indices in kf.split(features,labels):
        train_features = copy.copy(features[train_indices])
        best = CNN_Tuning (Tuning_function, train_features, labels[train_indices],
                           n_worker,name,n_in_cv,train_rate,max_eval,conv_key,gpu_key,dropout_key, n_fit)
        eval_cnn = Eval_Object(Eval_function, train_features, labels[train_indices],
                               features[test_indices], labels[test_indices],n_worker,name,conv_key,gpu_key,dropout_key, n_fit)
        bests += [best]
        accuracy += [eval_cnn(best)]
    return bests,accuracy


# In[28]:

def ave_dev (CNN_accuracy, n_cv):
    average = sum(CNN_accuracy) / n_cv
    difference = CNN_accuracy - np.array(list([average]) * n_cv)
    deviation = np.sqrt(sum(map(lambda x : x ** 2 , difference)) / n_cv)
    return average,deviation


# In[29]:

def map_CNN_X_elem (n_worker, n_fit, CNN_X_elem_elem, id_feature):#ワーカ*特徴のリストから必要なとこだけ抜き取って埋め
    new_CNN_X_elem = np.float32(np.zeros(n_worker * n_fit))
    new_CNN_X_elem[id_feature * n_worker : (id_feature + 1) * n_worker] = CNN_X_elem_elem
    return new_CNN_X_elem


# In[30]:

def CNN_X_2d_reshaper(CNN_X_elem, n_worker, n_fit):#各データ毎に埋め合わせ
    new_CNN_X= list(map(lambda x,y : map_CNN_X_elem(n_worker, n_fit, x,y), 
                              CNN_X_elem.reshape(n_fit,n_worker),range(n_fit)))
    return new_CNN_X


# In[31]:

def CNN_X_reshaper(CNN_X, n_worker,n_fit):#データ数*1チャネル*特徴数*(ワーカ数*特徴数)のテンソルにする
    new_CNN_X = np.array(list(map(lambda x:CNN_X_2d_reshaper(x, n_worker, n_fit), CNN_X.reshape(-1, n_fit, n_worker))))
    return new_CNN_X.reshape(-1,1,n_fit, n_fit * n_worker)


# In[51]:

#メーる関係
def create_message(from_addr, to_addr, bcc_addrs, subject, body):
    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From'] = from_addr
    msg['To'] = to_addr
    msg['Bcc'] = bcc_addrs
    msg['Date'] = formatdate()
    return msg


def send(from_addr, to_addrs, msg):
    smtpobj = smtplib.SMTP('smtp.gmail.com', 587)
    smtpobj.ehlo()
    smtpobj.starttls()
    smtpobj.ehlo()
    smtpobj.login(FROM_ADDRESS, MY_PASSWORD)
    smtpobj.sendmail(from_addr, to_addrs, msg.as_string())
    smtpobj.close()


# In[52]:

#ダミーのn_wifの数をそろえる
def map_map_check_n(Dummy_X_telem,n_dummy_wif,n_dummy_worker):
    n_now_wif = sum(abs(Dummy_X_telem))
    if n_now_wif > n_dummy_wif:
        for i in range (int(n_now_wif - n_dummy_wif)):
            Dummy_X_telem[np.where(Dummy_X_telem != 0)[0][int(np.random.rand() * (sum(abs(Dummy_X_telem))))]] = 0
    elif n_now_wif < n_dummy_wif:
        for j in range (int(n_dummy_wif - n_now_wif)):#とりあえず代入データはランダムで
            Dummy_X_telem[np.where(Dummy_X_telem == 0)[0]
                          [int(np.random.rand() * (n_dummy_worker - sum(abs(Dummy_X_telem))))]] = map_posi_neg(np.random.rand()*2-1)
    return Dummy_X_telem
    
def map_check_n(Dummy_X_elem,n_dummy_wif,n_dummy_worker):
    return list(map(lambda x:map_map_check_n(x,n_dummy_wif,n_dummy_worker),Dummy_X_elem[0]))

def check_n_worker(Dummy_X,r_work,n_dummy_worker,n_fit):
    n_dummy_wif = int(r_work * n_dummy_worker)
    return np.array(list(map(lambda x:map_check_n(x,n_dummy_wif,n_dummy_worker),Dummy_X.reshape(len(Dummy_X),1,n_fit,-1)))).reshape(len(Dummy_X),1,-1)


# In[53]:

def minus_eraser(x,tri_key):
    if tri_key == 0 and x < 0:
        x = 0
    return np.float32(x)


# In[54]:

def mmmap_tensor(CNN_X_eelem,any_key,func):
    return list(map(lambda x:func(x,any_key),CNN_X_eelem))

def mmap_tensor(CNN_X_elem,any_key,func):
    return list(map(lambda x:mmmap_tensor(x,any_key,func),CNN_X_elem))

def map_tensor (CNN_X,any_key,n_fit,func):#今更やけどテンソル全体にmapかけるための関数作る，funcは(対象、鍵)の組み合わせ
    return np.array(list(map(lambda x:mmap_tensor(x,any_key,func),CNN_X.reshape(len(CNN_X),n_fit,-1)))).reshape(len(CNN_X),1,-1)


# In[55]:

def amb_elim_iiter(CNN_X_eelem):
    if abs(sum(CNN_X_eelem) / sum(abs(CNN_X_eelem))) < 0.5:
        CNN_X_eelem = copy.copy(np.float32(np.zeros(len(CNN_X_eelem))))
    return CNN_X_eelem

def amb_elim_iter(CNN_X_elem):
    return list(map(amb_elim_iiter, (CNN_X_elem)))

def amb_elim(CNN_X,n_fit):#ambigious_eliminater、曖昧なラベルを抹消する
    return np.array(list(map(amb_elim_iter, (CNN_X.reshape(len(CNN_X), n_fit,-1))))).reshape(len(CNN_X),1,-1)


# In[56]:

def for_tensor(CNN_X,n_fit,func):#階層的処理の為の
    return np.array(list(map(func, (CNN_X.reshape(len(CNN_X), n_fit,-1))))).reshape(len(CNN_X),1,-1)

def partially_reverse(CNN_X_elem, rev_index):#指定に合わせてひっくり返す
    for i in rev_index: 
        CNN_X_elem.T[i]  = copy.copy(-1 * CNN_X_elem.T[i])
    return CNN_X_elem

def func_pr(rev_index):#func可する
    return lambda x:(partially_reverse(x, rev_index))


# In[57]:

def feature_decleaser (CNN_X, n_fit, dec_rate):#特徴量を減らしてみる
    n_data = len(CNN_X) 
    return np.split(CNN_X.reshape(n_data,n_fit, -1),int(n_fit * dec_rate), axis = 1)[0].reshape(n_data, 1, -1)


# In[58]:

def worker_sorted(CNN_X,n_fit):#ワーカを仕事が多い順に並べなおす
    length = len(CNN_X)
    sorted_index = np.array(list(map(sum,(map(sum,abs(CNN_X.reshape(length,n_fit,-1).T)))))).argsort()[::-1]
    return CNN_X.reshape(400,100,-1).T[sorted_index].T.reshape(length,1,-1)


# In[59]:

def worker_eraser(CNN_X_elem, n_worker):#ワーカの数を減らす
    return list(map(lambda x:worker_erase(x,n_worker),CNN_X_elem))

def worker_erase(CNN_X_eelem, n_worker):
    counter,task_counter = 0,0
    for i in CNN_X_eelem:
        if task_counter == n_worker:
            break
        if i != 0:
            task_counter += 1
        counter += 1
    return list(map(lambda x,y:work_eraser(x,y,counter),CNN_X_eelem,list(range(len(CNN_X_eelem)))))
    
def work_eraser(x,y,index):
    if y < index:
        return x
    if y >= index:
        return np.float32(0)  
            
def func_we(n_worker):#func可する
    return lambda x:(worker_eraser(x, n_worker))


def main():
    data_key = 4
    #4:pics03,5:smiles03、6:articles03、7:reviews03、8:pics01、9:smiles01、10:articles01、11:reviews01
    
    #初期値ここで設定

    df_data,n_loop,name,label_name = data_selecter(data_key)
    arranged_key = 0
    n_feature,n_task,n_wit,n_fit = df_init_pics(df_data)
    np.random.seed(0)
    if chainer.cuda.available:
        chainer.cuda.cupy.random.seed(0)#乱数固定

    #Crowd NN 関連
    n_in_cv = 0
    n_out_cv = 10
    train_rate = (n_out_cv - 2) / (n_out_cv - 1)
    max_eval = 200
    #0ならgpuを使う、-1なら使わない
    gpu_key = 0
    tri_key = 1
    dropout_key = 1
    conv_key = 1

    #入力反転,全反転、一部反転
    switch = 1
    
    n_wif = n_wit
    CNN_X , CNN_Y ,n_worker= df_to_CNN_arg(np.split(df_data,n_feature),n_task,n_fit,n_wif,tri_key)
    
    CNN_best,CNN_accuracy = cross_valid (Cross_Tuning,eval_CNN,(CNN_X * switch), CNN_Y,n_worker,name,n_in_cv,
                            train_rate,n_out_cv,max_eval,conv_key,gpu_key,n_wif,dropout_key, n_fit)
    CNN_ave, CNN_dev = ave_dev(CNN_accuracy, n_out_cv)
    return CNN_ave, CNN_dev
    

if __name__ == '__main__':
    main()

