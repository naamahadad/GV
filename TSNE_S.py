#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 20:53:51 2017

@author: naamahadad
"""

from sklearn.manifold import TSNE

import numpy as np
import sys
import os
#import KerasModels
#import yaml
import dataFactory_pyprep
#import keras
import DistNet
#from keras.models import load_model

home = os.path.expanduser('~')

params = {}
debug = True

params['config_file'] = sys.argv[1] if len(sys.argv)>1 else 'config_adam_epoch5000.yaml'
#params['data'] = [home +'/FinData/prices_debug.hdf']
params['data'] = '/media/data2/naamahadad/PyData/1976_2015.hdf'

#with open('yamls/' + params['config_file'],'r') as f:
#    params.update(yaml.load(f))

#params['res_path'] = home + '/results_nn/VAE'
params['res_path'] = '/media/data2/naamahadad/results/Debug'
params['years_dict'] = {'train_top' : 2012, # 2009
                      'test_bottom' : 2013, # 2010
                      'test_top' : 2015} # 2012

params['BN'] = True
params['Init'] = 'glorot_normal'
params['Activation'] = 'tanh'

params['recweight'] = 0.5#0.5#135,45
params['swap1weight'] = 1#100#1#280,93
params['swap2weight'] = 0.5#1#0.01#1.2,1.14
params['klweightZ'] = 0#0#0.5#0.25#10#0.1#0.121-->0,0.176
params['prodweight'] = 60

batch_size = 100
nb_epoch = 2000
params['batch_size'] = batch_size
params['nb_epoch'] = nb_epoch

params['original_dim'] = 230
params['intermediate_dim'] = 150
params['s_dim'] = 30
params['l_size'] = 60
params['latent_dim'] = 30#50
params['norm_sim'] = False
#vae = KerasModels.VAE(params,batch_size,original_dim,intermediate_dim,latent_dim)
net = DistNet.DistNet(params,'','',False)

#170117_111424_ clip,no std for z ,250 days
#170117_200219_ 230 days
#300117_211011_ combine s softmax+prod
net.AdvNet.load_weights(params['res_path'] +'/300117_211011_config_adam_epoch5000.yaml_weights.h5_advNet_weights.h5')
net.Adv.load_weights(params['res_path'] +'/300117_211011_config_adam_epoch5000.yaml_weights.h5_adv_weights.h5')
net.DistNet.load_weights(params['res_path'] +'/300117_211011_config_adam_epoch5000.yaml_weights.h5_full_weights.h5')

data_tags = ['day'+str(i) for i in range(0,250)]

print 'loading data...'
valid_years = np.asarray([1985,1990,1995,2000,2005,2010,2015])
datafactory = dataFactory_pyprep.dataFactory_pyprep(data_path=params['data'],years_dict=params['years_dict'],data_tags=data_tags,s_mode=1,load_ready_data=True,valid_years=valid_years)
print 'loading data group...'
x_train,id_train,clss_train,ret_train,beta_tr,x_test,id_test,clss_test,ret_test,beta_te = datafactory.get_samples_per_group(100000)
#x_train,id_train,clss_train,ret_train= datafactory.get_samples_per_group(100000)
#x_test,id_test = datafactory.get_test_data()
#x_test_vals = x_test[data_tags].values

#valinds=np.random.permutation(range(x_train.shape[0]))

#x_test=x_train[valinds[round(valinds.shape[0]*0.8):],:]
#id_test=id_train[valinds[round(valinds.shape[0]*0.8):]]
#clss_test=clss_train[valinds[round(valinds.shape[0]*0.8):]]
#ret_test=ret_train[valinds[round(valinds.shape[0]*0.8):]]
                             
#x_train=x_train[valinds[:round(valinds.shape[0]*0.8)],:]
#id_train=id_train[valinds[:round(valinds.shape[0]*0.8)]]
#clss_train=clss_train[valinds[:round(valinds.shape[0]*0.8)]]
#ret_train=ret_train[valinds[:round(valinds.shape[0]*0.8)]]


print 'train nan',np.sum(np.isnan(x_train))
print 'test nan',np.sum(np.isnan(x_test))
#
x_train[np.isnan(x_train)]=0
x_test[np.isnan(x_test)]=0
#
print 'train shape ',x_train.shape   
print 'clss_shape ' ,clss_train.shape
print 'clss_shape ' ,clss_test.shape
train_std = 0.04
train_avg = 0.0007
x_train = np.clip(x_train,train_avg- 4*train_std,train_avg + 4*train_std)
x_test = np.clip(x_test,train_avg- 4*train_std,train_avg + 4*train_std)
x_train = (x_train-0.0007)/0.04
x_test = (x_test-0.0007)/0.04
#X1 = x_train[:, :x_train.shape[1]/3]
#X1_te = x_test[:, :x_test.shape[1]/3]
X1 = x_train[:, :(x_train.shape[1]/3-20)]
X1_te = x_test[:, :(x_test.shape[1]/3-20)]
z1_m_full, z1_std_full, s1_full = net.predictEnc(X1)
z1_m_te, z1_std_te, s1_te = net.predictEnc(X1_te)


s_full = np.append(s1_full,s1_te,axis=0)
clss_full=np.append(clss_train,clss_test,axis=0)
ret_full=np.append(ret_train,ret_test,axis=0)
id_full=np.append(id_train,id_test,axis=0)

mix_inds = np.random.permutation(range(s_full.shape[0]))
s_full = s_full[mix_inds,:]
clss_full = clss_full[mix_inds]
ret_full = ret_full[mix_inds]
id_full = id_full[mix_inds]

S_full_t = np.zeros((s_full.shape[0],2))
batch_size = 35000
stopat=300000
model = TSNE(n_components=2, random_state=0,verbose=2)
for (first, last) in zip(range(0, min(s_full.shape[0]-batch_size,stopat-batch_size), batch_size),
                         range(batch_size, min(s_full.shape[0],stopat), batch_size)):
    s_cur = s_full[first:last,:]
    S_te = model.fit_transform(s_cur)
    print S_te.shape
    S_full_t[first:last,:] = S_te
    
#te2=s1_te[:5000,:]
#S1_trans_te = model.fit(te2)
#print 'test train: '
#S1_trans_te = model.fit(s1_te)
#S1_trans = model.fit(s1_full) 
#print s1_te.shape
#if s1_te.shape[1]==2:
#    S1_trans_te = s1_te
#    S1_trans = s1_full


dataroot='/media/data2/naamahadad/results/Sclsfrs/'
np.savetxt(dataroot + "/S1_trans.csv", S_full_t,fmt='%10.5f', delimiter=",")
np.savetxt(dataroot + "/tar_val.csv", clss_full,fmt='%d', delimiter=",")
np.savetxt(dataroot + "/tar_ret.csv", ret_full,fmt='%10.5f', delimiter=",")
np.savetxt(dataroot + "/ids.csv", id_full,fmt='%d', delimiter=",")
#np.savetxt(dataroot + "/S1_trans.csv", S1_trans,fmt='%10.5f', delimiter=",")
#np.savetxt(dataroot + "/S1_trans_te.csv", S1_trans_te,fmt='%10.5f', delimiter=",")
#np.savetxt(dataroot + "/tar_val.csv", clss_test,fmt='%d', delimiter=",")
#np.savetxt(dataroot + "/tar_ret.csv", ret_test,fmt='%10.5f', delimiter=",")
#np.savetxt(dataroot + "/id_test.csv", id_test,fmt='%d', delimiter=",")
#np.savetxt(dataroot + "/tar_val_tr.csv", clss_train,fmt='%d', delimiter=",")
#np.savetxt(dataroot + "/tar_ret_tr.csv", ret_train,fmt='%10.5f', delimiter=",")
#np.savetxt(dataroot + "/id_test_tr.csv", id_train,fmt='%d', delimiter=",")