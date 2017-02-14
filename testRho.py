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

params['recweight'] = 50#0.5#135,45
params['swap1weight'] = 100#100#1#280,93
params['swap2weight'] = 0.25#1#0.01#1.2,1.14
params['klweightZ'] = 60#10#0.1#0.121-->0,0.176

batch_size = 100
nb_epoch = 300
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

num_days_mode = 3
num_clss = 3
params_clsfr={}
params_clsfr['BN'] =True
params_clsfr['Init'] = 'glorot_normal'
params_clsfr['Activation'] = 'tanh'
params_clsfr['batch_size'] = batch_size
params_clsfr['nb_epoch'] = nb_epoch

params_clsfr['original_dim'] = 6
params_clsfr['intermediate_dim'] = 6
params_clsfr['num_clss'] = num_clss
params_clsfr['dp'] = 0.5
params_clsfr['reg'] = False
params_clsfr['mainnet'] ='300117_211011'

load_clsfr = False
params_clsfr['res_path'] = '/media/data2/naamahadad/results/EncRes'
if load_clsfr:
    clsfrNet1 = Sclsfr.Sclsfr(params_clsfr,'','',False)
    #200117_082140_ normal net day
    #270117_194145_ z onl+prod
    #280117_164300 mix s
    #290117_184028_ softmax
    clsfrNet1.Net.load_weights(params_clsfr['res_path'] + '/210117_201514_config_adam_epoch5000.yaml_Zclsfr_day_weights.h5')
    
    params_clsfr['original_dim'] = 3
    clsfrNet2 = Sclsfr.Sclsfr(params_clsfr,'','',False)
    clsfrNet2.Net.load_weights(params_clsfr['res_path'] + '/210117_201514_config_adam_epoch5000.yaml_Zclsfr_day_weights.h5')
else:
    curtime = strftime("%d%m%y_%H%M%S", localtime())
    log_filename = params_clsfr['res_path']  +'/'+ curtime + '_' + params['config_file'] + '_keras_Zclsfr'
    weihts_filename = params_clsfr['res_path'] +'/'+ curtime + '_' + params['config_file'] + '_Zclsfr_day_weights'
    outfile=open(log_filename+'.txt','a')
    clsfrNet1 = Sclsfr.Sclsfr(params_clsfr,outfile,weihts_filename+'.h5',True)
    
    params_clsfr['original_dim'] = 5
    outfile2=open(log_filename+'2.txt','a')
    clsfrNet2 = Sclsfr.Sclsfr(params_clsfr,outfile2,weihts_filename+'2.h5',True)

#170117_111424_ "" +clip+no std for z
#170117_200219_ 230 days
#220117_204007_ z only loss
#240117_215706_ z,s wo prod
#260117_120441_ z only with prod
#270117_231627_ all losses, combine s1,s1'
#280117_222954_ "" softmax
#300117_211011_ softmax + prod x1t loss
print 'loading net...'
net.AdvNet.load_weights(params['res_path'] +'/300117_211011_config_adam_epoch5000.yaml_weights.h5_advNet_weights.h5')
net.Adv.load_weights(params['res_path'] +'/300117_211011_config_adam_epoch5000.yaml_weights.h5_adv_weights.h5')
net.DistNet.load_weights(params['res_path'] +'/300117_211011_config_adam_epoch5000.yaml_weights.h5_full_weights.h5')

data_tags = ['day'+str(i) for i in range(0,250)]

print 'loading data...'
valid_years = np.asarray([1985,1990,1995,2000,2005,2010,2015])#-1
datafactory = dataFactory_pyprep.dataFactory_pyprep(data_path=params['data'],years_dict=params['years_dict'],data_tags=data_tags,s_mode=1,load_ready_data=True,valid_years=valid_years,calc_beta=True)
print 'loading data group...'
x_train,id_train,clss_train,ret_train,beta_tr,x_test,id_test,clss_test,ret_test,beta_te = datafactory.get_samples_per_group(100000)
#x_train,id_train,clss_train,ret_train= datafactory.get_samples_per_group(100000)
#x_test,id_test = datafactory.get_test_data()
#x_test_vals = x_test[data_tags].values

#valinds=np.random.permutation(range(x_train.shape[0]))
#
#x_test=x_train[valinds[round(valinds.shape[0]*0.8):],:]
#id_test=id_train[valinds[round(valinds.shape[0]*0.8):]]
#clss_test=clss_train[valinds[round(valinds.shape[0]*0.8):]]
#ret_test=ret_train[valinds[round(valinds.shape[0]*0.8):]]
#                             
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
X1t = x_train[:, (x_train.shape[1]/3):(2*x_train.shape[1]/3-20)]
X1t_te = x_test[:, (x_test.shape[1]/3):(2*x_test.shape[1]/3-20)]
print 'X1 shape ',X1.shape  

ret1 = np.sum(x_train_orig[:,(x_train.shape[1]/3-20):(x_train.shape[1]/3-15)],axis=1)
ret1te = np.sum(x_test_orig[:,(x_train.shape[1]/3-20):(x_train.shape[1]/3-15)],axis=1)
ret2 = np.sum(x_train_orig[:,((2*x_train.shape[1]/3)-20):((2*x_train.shape[1]/3)-15)],axis=1)
ret2te= np.sum(x_test_orig[:,(2*(x_train.shape[1]/3)-20):((2*x_train.shape[1]/3)-15)],axis=1)
ret_train = ret1-ret2
ret_test = ret1te - ret2te

X1o = x_train_orig[:, :(x_train_orig.shape[1]/3-20)]
X1o_te = x_test_orig[:, :(x_test_orig.shape[1]/3-20)]
X1to = x_train_orig[:, (x_train_orig.shape[1]/3):(2*x_train_orig.shape[1]/3-20)]
X1to_te = x_test_orig[:, (x_test_orig.shape[1]/3):(2*x_test_orig.shape[1]/3-20)]
                
days_len = X1o.shape[1]
ret_week1 = X1o[:,(days_len-5):]
ret_week1_te = X1o_te[:,(days_len-5):]
ret_week2 = X1to[:,(days_len-5):]
ret_week2_te= X1to_te[:,(days_len-5):]

ret_mon1 = X1o[:,(days_len-40):]
ret_mon1_te = X1o_te[:,(days_len-40):]
ret_mon2 = X1to[:,(days_len-40):]
ret_mon2_te= X1to_te[:,(days_len-40):]
                         
ret_week1 = ret_week1 - np.repeat(np.expand_dims(np.average(X1o,axis=1),axis=1),ret_week1.shape[1],axis=1)
ret_week2 = ret_week2 - np.repeat(np.expand_dims(np.average(X1to,axis=1),axis=1),ret_week2.shape[1],axis=1)
ret_week1_te = ret_week1_te - np.repeat(np.expand_dims(np.average(X1o_te,axis=1),axis=1),ret_week1.shape[1],axis=1)
ret_week2_te = ret_week2_te - np.repeat(np.expand_dims(np.average(X1to_te,axis=1),axis=1),ret_week2.shape[1],axis=1)
ret_mon1 = ret_mon1 - np.repeat(np.expand_dims(np.average(X1o,axis=1),axis=1),ret_mon1.shape[1],axis=1)
ret_mon2 = ret_mon2 - np.repeat(np.expand_dims(np.average(X1to,axis=1),axis=1),ret_mon2.shape[1],axis=1)
ret_mon1_te = ret_mon1_te - np.repeat(np.expand_dims(np.average(X1o_te,axis=1),axis=1),ret_mon1_te.shape[1],axis=1)
ret_mon2_te = ret_mon2_te - np.repeat(np.expand_dims(np.average(X1to_te,axis=1),axis=1),ret_mon2_te.shape[1],axis=1)
ret_full1 = X1o - np.repeat(np.expand_dims(np.average(X1o,axis=1),axis=1),X1o.shape[1],axis=1)
ret_full2 = X1to - np.repeat(np.expand_dims(np.average(X1to,axis=1),axis=1),X1to.shape[1],axis=1)
ret_full1_te = X1o_te - np.repeat(np.expand_dims(np.average(X1o_te,axis=1),axis=1),X1o_te.shape[1],axis=1)
ret_full2_te = X1to_te - np.repeat(np.expand_dims(np.average(X1to_te,axis=1),axis=1),X1to_te.shape[1],axis=1)

std1 = np.std(X1o,axis=1) + 1e-10
std2 = np.std(X1to,axis=1) + 1e-10
std1_te = np.std(X1o_te,axis=1) + 1e-10
std2_te = np.std(X1to_te,axis=1) + 1e-10
N=ret_week1.shape[1]
rho12_week = np.sum(ret_week1*ret_week2,axis=1)/(N*std1*std2)
rho12_week_te = np.sum(ret_week1_te*ret_week2_te,axis=1)/(N*std1_te*std2_te)
#rho12_week_te = (rho12_week_te-np.average(rho12_week))/np.std(rho12_week)
#rho12_week = (rho12_week-np.average(rho12_week))/np.std(rho12_week)
diffweek = (1/N)*(np.sum(ret_week1,axis=1)/std1 - np.sum(ret_week2,axis=1)/std2)
diffweek_te = (1/N)*(np.sum(ret_week1_te,axis=1)/std1_te - np.sum(ret_week2_te,axis=1)/std2_te)

N=ret_mon1.shape[1]
rho12_mon = np.sum(ret_mon1*ret_mon2,axis=1)/(N*std1*std2)
rho12_mon_te = np.sum(ret_mon1_te*ret_mon2_te,axis=1)/(N*std1_te*std2_te)           
#rho12_mon_te = (rho12_mon_te-np.average(rho12_mon))/np.std(rho12_mon)
#rho12_mon= (rho12_mon-np.average(rho12_mon))/np.std(rho12_mon)
diffmon = (1/N)*(np.sum(ret_mon1,axis=1)/std1 - np.sum(ret_mon2,axis=1)/std2)
diffmon_te = (1/N)*(np.sum(ret_mon1_te,axis=1)/std1_te - np.sum(ret_mon2_te,axis=1)/std2_te)

N=ret_full1.shape[1]
rho12_full = np.sum(ret_full1*ret_full2,axis=1)/(N*std1*std2)
rho12_full_te = np.sum(ret_full1_te*ret_full2_te,axis=1)/(N*std1_te*std2_te)   
#rho12_full_te = (rho12_full_te-np.average(rho12_full))/np.std(rho12_full)
#rho12_full= (rho12_full-np.average(rho12_full))/np.std(rho12_full)
                        
if num_clss==3:
    retavg1 = np.percentile(ret_train,33)
    retavg2 = np.percentile(ret_train,66)
    clss_train = 1*((ret_train - retavg1)>0)+1*((ret_train - retavg2)>0)
    clss_test = 1*((ret_test - retavg1)>0)+1*((ret_test - retavg2)>0)
elif num_clss==4:
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

dataroot='/media/data2/naamahadad/results/EncRes/'
clsfr_mode=0
savename = '_pairtrade_softmax2'

z1t_m_full, z1t_std_full, s1t_full = net.predictEnc(X1t)
z1t_m_te, z1t_std_te, s1t_te = net.predictEnc(X1t_te)

X1_est = net.VAEdecS.predict([z1_m_full,s1_full],batch_size=batch_size)
X1_est_te = net.VAEdecS.predict([z1_m_te,s1_te],batch_size=batch_size)
X1t_est = net.VAEdecS.predict([z1_m_full,s1t_full],batch_size=batch_size)
X1t_est_te = net.VAEdecS.predict([z1_m_te,s1t_te],batch_size=batch_size)

MSEX1X1t = np.sum(np.square(X1_est-X1t_est)/(np.square(X1_est)+1e-10),axis=1)
MSEX1X1t_te = np.sum(np.square(X1_est_te-X1t_est_te)/(np.square(X1_est_te)+1e-10),axis=1)
MSEavg = np.average(MSEX1X1t)
MSEstd = np.std(MSEX1X1t)
print 'train mse: ', MSEavg,MSEstd
print 'test mse: ', np.average(MSEX1X1t_te),np.std(MSEX1X1t_te)
MSEX1X1t = (MSEX1X1t-MSEavg)/MSEstd
MSEX1X1t_te = (MSEX1X1t_te-MSEavg)/MSEstd

data1 = X1_est - np.repeat(np.expand_dims(np.average(X1_est,axis=1),axis=1),X1_est.shape[1],axis=1)
data2 = X1t_est - np.repeat(np.expand_dims(np.average(X1t_est,axis=1),axis=1),X1t_est.shape[1],axis=1)
data1_te = X1_est_te - np.repeat(np.expand_dims(np.average(X1_est_te,axis=1),axis=1),X1_est_te.shape[1],axis=1)
data2_te = X1t_est_te - np.repeat(np.expand_dims(np.average(X1t_est_te,axis=1),axis=1),X1t_est_te.shape[1],axis=1)

std1 = np.std(X1_est,axis=1) + 1e-10
std2 = np.std(X1t_est,axis=1) + 1e-10
std1_te = np.std(X1_est_te,axis=1) + 1e-10
std2_te = np.std(X1t_est_te,axis=1) + 1e-10
N=data1.shape[1]
rho12_est= np.sum(data1*data2,axis=1)/(N*std1*std2)
rho12_est_te = np.sum(data1_te*data2_te,axis=1)/(N*std1_te*std2_te)

print MSEX1X1t.shape,rho12_est.shape,rho12_week.shape,rho12_mon.shape,rho12_full.shape
inputs1 = np.concatenate((np.expand_dims(MSEX1X1t,axis=1),np.expand_dims(rho12_est,axis=1),np.expand_dims(rho12_week,axis=1),np.expand_dims(rho12_mon,axis=1),np.expand_dims(diffweek,axis=1),np.expand_dims(diffmon,axis=1)),axis=1)
inputs1_test = np.concatenate((np.expand_dims(MSEX1X1t_te,axis=1),np.expand_dims(rho12_est_te,axis=1),np.expand_dims(rho12_week_te,axis=1),np.expand_dims(rho12_mon_te,axis=1),np.expand_dims(diffweek_te,axis=1),np.expand_dims(diffmon_te,axis=1)),axis=1)

inputs2 = np.concatenate((np.expand_dims(rho12_full,axis=1),np.expand_dims(rho12_week,axis=1),np.expand_dims(rho12_mon,axis=1),np.expand_dims(diffweek,axis=1),np.expand_dims(diffmon,axis=1)),axis=1)
inputs2_test = np.concatenate((np.expand_dims(rho12_full_te,axis=1),np.expand_dims(rho12_week_te,axis=1),np.expand_dims(rho12_mon_te,axis=1),np.expand_dims(diffweek_te,axis=1),np.expand_dims(diffmon_te,axis=1)),axis=1)
   
print 'logistic train1...'
bdt = linear_model.LogisticRegression()
bdt.fit(inputs1, clss_train)
preds_reg1 = bdt.predict_proba(inputs1_test)#

print 'logistic train2...'
bdt = linear_model.LogisticRegression()
bdt.fit(inputs2, clss_train)
preds_reg2 = bdt.predict_proba(inputs2_test)

np.savetxt(dataroot + "/inputs1_test" +savename+".csv", inputs1_test,fmt='%10.5f', delimiter=",")
np.savetxt(dataroot + "/inputs2_test" +savename+".csv", inputs2_test,fmt='%10.5f', delimiter=",")
np.savetxt(dataroot + "/pred_reg1" +savename+".csv", preds_reg1,fmt='%10.5f', delimiter=",")
np.savetxt(dataroot + "/pred_reg2" +savename+".csv", preds_reg2,fmt='%10.5f', delimiter=",")

np.savetxt(dataroot + "/tar_val"+savename+".csv", clss_test,fmt='%d', delimiter=",")
np.savetxt(dataroot + "/tar_ret"+savename+".csv", ret_test,fmt='%10.5f', delimiter=",")
np.savetxt(dataroot + "/id_test"+savename+".csv", id_test,fmt='%d', delimiter=",")


if load_clsfr:
    pred_ids1 = clsfrNet1.Net.predict(inputs1,batch_size=params['batch_size'])
    pred_ids2 = clsfrNet2.Net.predict(inputs2,batch_size=params['batch_size'])
else:
    if params_clsfr['reg'] :        
        print 'net train1...'
        hist = clsfrNet1.Net.fit(inputs1, ret_train,shuffle=True,nb_epoch=nb_epoch,batch_size=batch_size,
                                verbose=2,validation_data=(inputs1_test, ret_test),callbacks=[clsfrNet1.checkpointer])
        print 'net train2...'
        hist = clsfrNet2.Net.fit(inputs2, ret_train,shuffle=True,nb_epoch=nb_epoch,batch_size=batch_size,
                                verbose=2,validation_data=(inputs2_test, ret_test),callbacks=[clsfrNet2.checkpointer])
  
        #clsfrNet.Net.load_weights(weihts_filename)
        #pred_ids = clsfrNet.Net.predict(z1_m_te,batch_size=params['batch_size'])
        
    else:
        print 'net train1...'
        hist = clsfrNet1.Net.fit(inputs1, np_utils.to_categorical(clss_train,params_clsfr['num_clss']),shuffle=True,nb_epoch=nb_epoch,batch_size=batch_size,
                                verbose=2,validation_data=(inputs1_test, np_utils.to_categorical(clss_test,params_clsfr['num_clss'])),callbacks=[clsfrNet1.checkpointer])
        hist = clsfrNet2.Net.fit(inputs2, np_utils.to_categorical(clss_train,params_clsfr['num_clss']),shuffle=True,nb_epoch=nb_epoch,batch_size=batch_size,
                               verbose=2,validation_data=(inputs2_test, np_utils.to_categorical(clss_test,params_clsfr['num_clss'])),callbacks=[clsfrNet2.checkpointer])

                
    clsfrNet1.Net.load_weights(weihts_filename+'.h5')
    pred_ids1 = clsfrNet1.Net.predict(inputs1_test,batch_size=params['batch_size'])    
 
    clsfrNet2.Net.load_weights(weihts_filename+'2.h5')
    pred_ids2 = clsfrNet2.Net.predict(inputs2_test,batch_size=params['batch_size']) 
    
np.savetxt(dataroot + "/pred_val1"+savename+".csv", pred_ids1,fmt='%10.5f', delimiter=",")
np.savetxt(dataroot + "/pred_val2"+savename+".csv", pred_ids2,fmt='%10.5f', delimiter=",")
    

