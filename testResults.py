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

params_clsfr={}
params_clsfr['BN'] =True
params_clsfr['Init'] = 'glorot_normal'
params_clsfr['Activation'] = 'tanh'
params_clsfr['batch_size'] = batch_size
params_clsfr['nb_epoch'] = nb_epoch

params_clsfr['original_dim'] = params['s_dim']
params_clsfr['intermediate_dim'] = 30
params_clsfr['num_clss'] = 5
params_clsfr['dp'] = 0.5
params_clsfr['reg'] = False
params_clsfr['mainnet'] ='170117_200219'

params_clsfr['res_path'] = '/media/data2/naamahadad/results/EncRes'
load_clsfr = False
if load_clsfr:
    clsfrNet = Sclsfr.Sclsfr(params_clsfr,'','',False)
    clsfrNet.Net.load_weights(params_clsfr['res_path'] + '/180117_203739_config_adam_epoch5000.yaml_weights.h5')
else:
    curtime = strftime("%d%m%y_%H%M%S", localtime())
    log_filename = params_clsfr['res_path']  +'/'+ curtime + '_' + params['config_file'] + '_keras_Sclsfr.txt'
    weihts_filename = params_clsfr['res_path'] +'/'+ curtime + '_' + params['config_file'] + '_weights.h5'
    outfile=open(log_filename,'a')
    
    clsfrNet = Sclsfr.Sclsfr(params_clsfr,outfile,weihts_filename,True)
#060117_090836 - 1 layer
#060117_102959_ bn on for adv
#060117_105714
#060117_121757 all loss
#060117_150233 first loss
#060117_150814 second loss
#060117_151941 loss 3
#060117_152853 loss 4
#250 z,N(10,10^2) 060117_155313 loss1
#070117_132115 loss 1 BN mode2
#070117_140012 loss 1 BNen 
#070117_150943 loss 1 BN disable
#070117_162232 loss1,2 BN enable
#170117_111424_ "" +clip+no std for z --->
#220117_204007_ z only loss
#280117_222954_ softmax
#300117_211011_ softmax + prod x1t loss
net.AdvNet.load_weights(params['res_path'] +'/'+params_clsfr['mainnet']+'_config_adam_epoch5000.yaml_weights.h5_advNet_weights.h5')
net.Adv.load_weights(params['res_path'] +'/'+params_clsfr['mainnet']+'_config_adam_epoch5000.yaml_weights.h5_adv_weights.h5')
net.DistNet.load_weights(params['res_path'] +'/'+params_clsfr['mainnet']+'_config_adam_epoch5000.yaml_weights.h5_full_weights.h5')

data_tags = ['day'+str(i) for i in range(0,250)]

print 'loading data...'
valid_years = np.asarray([1985,1990,1995,2000,2005,2010,2015])
datafactory = dataFactory_pyprep.dataFactory_pyprep(data_path=params['data'],years_dict=params['years_dict'],data_tags=data_tags,s_mode=1,load_ready_data=True,valid_years=valid_years,eleven_month=True)
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
X1 = x_train[:, :x_train.shape[1]/3]
X1_te = x_test[:, :x_test.shape[1]/3]
z1_m_full, z1_std_full, s1_full = net.predictEnc(X1)
z1_m_te, z1_std_te, s1_te = net.predictEnc(X1_te)


print 'z1mue: '+ str(np.average(z1_m_full))+' std: '+ str(np.std(z1_m_full))+' max: '+ str(np.max(np.abs((z1_m_full))))
print 'z1std: '+ str(np.average(z1_std_full))+' std: '+ str(np.std(z1_std_full))+' max: '+ str(np.max(np.abs((z1_std_full))))
print 's1mean: '+ str(np.average(s1_full))+' std: '+ str(np.std(s1_full))+' max: '+ str(np.max(np.abs((s1_full))))

dataroot='/media/data2/naamahadad/results/EncRes/'
np.save(dataroot+'X', X1)
np.save(dataroot+'Xid', id_train)
np.save(dataroot+'clss_train', clss_train)
np.save(dataroot+'ret_train', ret_train)
np.save(dataroot+'Z_m', z1_m_full)
np.save(dataroot+'Z_std', z1_std_full)
np.save(dataroot+'S', s1_full)

np.save(dataroot+'Xte', X1_te)
np.save(dataroot+'Xid_te', id_test)
np.save(dataroot+'clss_te', clss_test)
np.save(dataroot+'ret_te', ret_test)
np.save(dataroot+'Z_m_te', z1_m_te)
np.save(dataroot+'Z_std_te', z1_std_te)
np.save(dataroot+'S_te', s1_te)

np.savetxt(dataroot+ "X_te.csv", X1_te,fmt='%10.5f', delimiter=",")
np.savetxt(dataroot + "X_id_te.csv", id_test,fmt='%d', delimiter=",")
np.savetxt(dataroot + "Z_m_te.csv", z1_m_te,fmt='%10.5f', delimiter=",")
np.savetxt(dataroot + "Z_std_te.csv", z1_std_te,fmt='%10.5f', delimiter=",")
np.savetxt(dataroot + "S_te.csv", s1_te,fmt='%10.5f', delimiter=",")

if load_clsfr:
    pred_ids = clsfrNet.Net.predict(s1_te,batch_size=params['batch_size'])
else:
    print 'logistic train...'
    
    if params_clsfr['reg'] :
        bdt = linear_model.LinearRegression()
        bdt.fit(s1_full, ret_train)
        preds_reg = bdt.predict(s1_te)#_proba
        
        print 'net train...'
        hist = clsfrNet.Net.fit(s1_full, ret_train,shuffle=True,nb_epoch=nb_epoch,batch_size=batch_size,
                                verbose=2,validation_data=(s1_te, ret_test),callbacks=[clsfrNet.checkpointer])
    
        #pred_ids = clsfrNet.Net.predict(s1_te,batch_size=params['batch_size'])
        
    else:
        bdt = linear_model.LogisticRegression()
        bdt.fit(s1_full, clss_train)
        preds_reg = bdt.predict_proba(s1_te)#
        print 'net train...'
        hist = clsfrNet.Net.fit(s1_full, np_utils.to_categorical(clss_train,5),shuffle=True,nb_epoch=nb_epoch,batch_size=batch_size,
                                verbose=2,validation_data=(s1_te, np_utils.to_categorical(clss_test,5)),callbacks=[clsfrNet.checkpointer])
    
    clsfrNet.Net.load_weights(weihts_filename)
    pred_ids = clsfrNet.Net.predict(s1_te,batch_size=params['batch_size'])    
        #pred_ids=0
    np.savetxt(dataroot + "/pred_reg.csv", preds_reg,fmt='%10.5f', delimiter=",")
    
np.savetxt(dataroot + "/pred_val.csv", pred_ids,fmt='%10.5f', delimiter=",")
np.savetxt(dataroot + "/tar_val.csv", clss_test,fmt='%d', delimiter=",")
np.savetxt(dataroot + "/tar_ret.csv", ret_test,fmt='%10.5f', delimiter=",")
np.savetxt(dataroot + "/id_test.csv", id_test,fmt='%d', delimiter=",")
#np.savetxt(params['res_path'] + "/X.csv", X1,fmt='%10.5f', delimiter=",")
#np.savetxt(params['res_path'] + "/X_id.csv", id_train,fmt='%d', delimiter=",")
#np.savetxt(params['res_path'] + "/Z_m.csv", z1_m_full,fmt='%10.5f', delimiter=",")
#np.savetxt(params['res_path'] + "/Z_std.csv", z1_std_full,fmt='%10.5f', delimiter=",")
#np.savetxt(params['res_path'] + "/S.csv", s1_full,fmt='%10.5f', delimiter=",")

#print 'test shape ',x_test_vals.shape         
#
#maxSamples = (np.floor(x_train.shape[0]/batch_size)*batch_size).astype(np.int64)
#x_train = x_train[0:maxSamples,:]
#maxSamples = (np.floor(x_test_vals.shape[0]/batch_size)*batch_size).astype(np.int64)
#x_test_vals = x_test_vals[0:maxSamples,:]
#
#valinds=np.random.permutation(range(x_test_vals.shape[0]))
#valinds=valinds[:params['batch_size']]
#x_train = (x_train-0.0007)/0.04
#X1 = x_train[valinds, :x_train.shape[1]/3]
#X1t = x_train[valinds, (x_train.shape[1]/3):(2*x_train.shape[1]/3)]
#X2 = x_train[valinds, (2*x_train.shape[1]/3):x_train.shape[1]]
#lbls = id_train[valinds]
#
##zn=np.random.randn(100,params['latent_dim'])#
##zn=(0.0001*np.random.randn(100,params['latent_dim']))+10
#zn=(1*np.random.randn(100,params['latent_dim']))+0#10
#
#z1_m,z1_std,s1 = net.VAEencS.predict(X1,batch_size=params['batch_size'])
#X11,X11t,X12,Xp2,adv1,adv2 = net.DistNet.predict([X1,X1t,X2,zn,lbls],batch_size=params['batch_size'])
#print 'z1mue: '+ str(np.average(z1_m))+' z1std: '+ str(np.average(z1_std))
#print 'loss1: '+ str(np.average(np.square(X11-X1)))+' loss2: '+ str(np.average(np.square(X11t-X1)))
##np.average(np.square(X11-X1),axis=0)
#
#mixinds=np.random.permutation(range(X1.shape[0]))
#print 'loss1: '+ str(np.average(np.square(X11-X1[mixinds,:])))+' loss2: '+ str(np.average(np.square(X11t-X1[mixinds,:])))
#print 'adv1: ',np.average(adv1),np.min(adv1),np.max(adv1),np.std(adv1),' adv2: ',np.average(adv2),np.min(adv2),np.max(adv2),np.std(adv2)
