import numpy as np
#import matplotlib.pyplot as plt
#from scipy.stats import norm
#
#from keras.layers import Input, Dense, Lambda
#from keras.models import Model
#from keras import backend as K
#from keras import objectives
#from keras.datasets import mnist
from time import strftime,localtime
import sys
import os
#import KerasModels
#import yaml
import dataFactory_pyprep
#import keras
import DistNet
import Sclsfr
from keras.utils import np_utils
from sklearn import linear_model
from collections import deque
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
params['res_path'] = '/media/data2/naamahadad/results/MainVAE'
params['years_dict'] = {'train_top' : 2012, # 2009
                      'test_bottom' : 2013, # 2010
                      'test_top' : 2015} # 2012

params['BN'] = True
params['Init'] = 'glorot_normal'
params['Activation'] = 'tanh'

params['recweight'] = 50#0.5#135,45
params['swap1weight'] = 100#100#1#280,93
params['swap2weight'] = 0.25#1#0.01#1.2,1.14
params['klweightZ'] = 60#10#0.1#0.121-->0,0.176
params['prodweight'] = 60

batch_size = 100
nb_epoch = 1500
params['batch_size'] = batch_size
params['nb_epoch'] = nb_epoch

params['original_dim'] = 230
params['intermediate_dim'] = 120
params['s_dim'] = 30
params['l_size'] = 60
params['latent_dim'] = 30#50
params['norm_sim'] = False

#vae = KerasModels.VAE(params,batch_size,original_dim,intermediate_dim,latent_dim)
net = DistNet.DistNet(params,'','',False)

num_days_mode = 1

load_clsfr = False
params_clsfr = {}
params_clsfr['reg'] = False
savename = 'histweek'

#170117_111424_ "" +clip+no std for z
#170117_200219_ 230 days
#220117_204007_ z only loss
#240117_215706_ z,s wo prod
#260117_120441_ z only with prod
#270117_231627_ all losses, combine s1,s1'
#280117_222954_ "" softmax
#300117_211011_ softmax + prod x1t loss
data_tags = ['day'+str(i) for i in range(0,250)]

print 'loading data...'
valid_years = np.asarray([1985,1990,1995,2000,2005,2010,2015])#-1
datafactory = dataFactory_pyprep.dataFactory_pyprep(data_path=params['data'],years_dict=params['years_dict'],data_tags=data_tags,s_mode=1,load_ready_data=True,valid_years=valid_years,calc_beta=True)

nets = deque(["010217_130904_", "010217_204651_", "020217_212635_","030217_164919_","060217_070121_","070217_064522_","060217_213414_"])
preds_reg_full = np.asarray([])
clss_test_full = np.asarray([])
ret_test_full = np.asarray([])
id_test_full = np.asarray([])
for maxyear in valid_years:
    net_name = nets.popleft()
    print 'loading net:',net_name
    net.AdvNet.load_weights(params['res_path'] +'/' + net_name + 'config_adam_epoch5000.yaml_weights.h5_advNet_weights.h5')
    net.Adv.load_weights(params['res_path'] +'/' + net_name + 'config_adam_epoch5000.yaml_weights.h5_adv_weights.h5')
    net.DistNet.load_weights(params['res_path'] +'/' + net_name + 'config_adam_epoch5000.yaml_weights.h5_full_weights.h5')
    
    #x_train,id_train,x_test,id_test= datafactory.get_data_by_time(maxyear-10,maxyear)
    x_train,id_train= datafactory.get_data_by_time(maxyear-50,maxyear-1)
    x_test,id_test= datafactory.get_data_by_time(maxyear,maxyear)
    
    x_train[np.isnan(x_train)]=0
    x_test[np.isnan(x_test)]=0

    print 'train max ',max(id_train), ' train min ',min(id_test) 
    print 'train_shape ' ,x_train.shape
    print 'test_shape ' ,x_test.shape
    x_train_orig = x_train
    x_test_orig = x_test
    train_std = 0.04
    train_avg = 0.0007
    x_train = np.clip(x_train,train_avg- 4*train_std,train_avg + 4*train_std)
    x_test = np.clip(x_test,train_avg- 4*train_std,train_avg + 4*train_std)
    x_train = (x_train-0.0007)/0.04
    x_test = (x_test-0.0007)/0.04
    X1 = x_train[:, :(x_train.shape[1]/3-20)]
    X1_te = x_test[:, :(x_test.shape[1]/3-20)]

    if num_days_mode==0:
        ret_train = x_train_orig[:,x_train.shape[1]/3-20]
        ret_test = x_test_orig[:,x_train.shape[1]/3-20]
    elif num_days_mode==1:
        ret_train = np.sum(x_train_orig[:,(x_train.shape[1]/3-20):(x_train.shape[1]/3-15)],axis=1)
        ret_test = np.sum(x_test_orig[:,(x_train.shape[1]/3-20):(x_train.shape[1]/3-15)],axis=1)
    #elif num_days_mode==3:
    #    ret_train = beta_tr
    #    ret_test = beta_te
   
    if num_days_mode<3:
        retavg1 = np.percentile(ret_train,33)
        retavg2 = np.percentile(ret_train,66)
        clss_train = 1*((ret_train - retavg1)>0)+1*((ret_train - retavg2)>0)
        clss_test = 1*((ret_test - retavg1)>0)+1*((ret_test - retavg2)>0)
    else:
        retavg1 = np.percentile(ret_train,25)
        retavg2 = np.percentile(ret_train,50)
        retavg3 = np.percentile(ret_train,75)
        clss_train = 1*((ret_train - retavg1)>0)+1*((ret_train - retavg2)>0)+1*((ret_train - retavg3)>0)
        clss_test = 1*((ret_test - retavg1)>0)+1*((ret_test - retavg2)>0)+1*((ret_test - retavg3)>0)
    print 'day230 avg1: ' ,retavg1,'avg2: ' ,retavg2, ' clss_tr_avg: ' ,np.average(clss_train), ' clss_te_avg: ' ,np.average(clss_test)
    
    z1_m_full, z1_std_full, s1_full = net.predictEnc(X1)
    z1_m_te, z1_std_te, s1_te = net.predictEnc(X1_te)

    print 'z1mue: '+ str(np.average(z1_m_full))+' std: '+ str(np.std(z1_m_full))+' max: '+ str(np.max(np.abs((z1_m_full))))
    print 'z1std: '+ str(np.average(z1_std_full))+' std: '+ str(np.std(z1_std_full))+' max: '+ str(np.max(np.abs((z1_std_full))))
    #print 's1mean: '+ str(np.average(s1_full))+' std: '+ str(np.std(s1_full))+' max: '+ str(np.max(np.abs((s1_full))))

    print 'logistic train...'
    if params_clsfr['reg'] :
        bdt = linear_model.LinearRegression()
        bdt.fit(z1_m_full, ret_train)
        preds_reg = bdt.predict(z1_m_te)#_proba         
    else:
        bdt = linear_model.LogisticRegression()
        bdt.fit(z1_m_full, clss_train)
        preds_reg = bdt.predict_proba(z1_m_te)#
        
    preds_reg_full = preds_reg if preds_reg_full.shape[0]==0 else np.concatenate((preds_reg_full,preds_reg),axis=0)
    clss_test_full = clss_test if clss_test_full.shape[0]==0 else np.concatenate((clss_test_full,clss_test),axis=0)
    ret_test_full = ret_test if ret_test_full.shape[0]==0 else np.concatenate((ret_test_full,ret_test),axis=0)
    id_test_full = id_test if id_test_full.shape[0]==0 else np.concatenate((id_test_full,id_test),axis=0)

dataroot='/media/data2/naamahadad/results/EncRes/'
np.savetxt(dataroot + "/pred_reg"+savename+ ".csv", preds_reg_full,fmt='%10.5f', delimiter=",")
np.savetxt(dataroot + "/tar_val"+savename+ ".csv", clss_test_full,fmt='%d', delimiter=",")
np.savetxt(dataroot + "/tar_ret"+savename+ ".csv", ret_test_full,fmt='%10.5f', delimiter=",")
np.savetxt(dataroot + "/id_test"+savename+ ".csv", id_test_full,fmt='%d', delimiter=",")
