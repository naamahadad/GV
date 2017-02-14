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
from keras.models import load_model

home = os.path.expanduser('~')

params = {}
debug = True

params['config_file'] = sys.argv[1] if len(sys.argv)>1 else 'config_adam_epoch5000.yaml'
#params['data'] = [home +'/FinData/prices_debug.hdf']
params['data'] = '/media/data2/naamahadad/PyData/1996_2015.hdf'

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
params['klweightZ'] = 1#10#0.1#0.121-->0,0.176

batch_size = 100
nb_epoch = 500
params['batch_size'] = batch_size
params['nb_epoch'] = nb_epoch

params['original_dim'] = 250
params['intermediate_dim'] = 150
params['s_dim'] = 30
params['l_size'] = 60
params['latent_dim'] = 30#50

#vae = KerasModels.VAE(params,batch_size,original_dim,intermediate_dim,latent_dim)
net = DistNet.DistNet(params,'','',False)

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
#070117_222627_ full net BN,50,100,5,0.25
#080117_220721 full net prodloss 50,100,0.5,0.5,4000
#110117_211846_ loss 1,2,5
#120117_210751_ loss 1,2,3,5
#160117_204119_ 1,2,3,5 valod years
#170117_111424_ "" +clip+no std for z
#net.Adv.summary()
#net.DistNet.summary()
net.AdvNet.load_weights(params['res_path'] +'/170117_111424_config_adam_epoch5000.yaml_weights.h5_advNet_weights.h5')
net.Adv.load_weights(params['res_path'] +'/170117_111424_config_adam_epoch5000.yaml_weights.h5_adv_weights.h5')
net.DistNet.load_weights(params['res_path'] +'/170117_111424_config_adam_epoch5000.yaml_weights.h5_full_weights.h5')
#net.VAEencS.summary()
#net.DistNet.summary()

data_tags = ['day'+str(i) for i in range(0,250)]

print 'loading data...'
valid_years = [1985,1990,1995,2000,2005,2010,2015]
#valid_years = np.array([])
datafactory = dataFactory_pyprep.dataFactory_pyprep(data_path=params['data'],years_dict=params['years_dict'],data_tags=data_tags,s_mode=1,load_ready_data=True,valid_years=valid_years)
x_train,id_train = datafactory.get_train_data()
x_test,id_test = datafactory.get_test_data()
#x_test_vals = x_test[data_tags].values
x_test_vals = x_test
x_train = x_test_vals
print 'train nan',np.sum(np.isnan(x_train))
print 'test nan',np.sum(np.isnan(x_test_vals))

x_train[np.isnan(x_train)]=0
x_test_vals[np.isnan(x_test_vals)]=0

train_std = 0.04
train_avg = 0.0007
x_train = np.clip(x_train,train_avg- 4*train_std,train_avg + 4*train_std)
x_test_vals = np.clip(x_test_vals,train_avg- 4*train_std,train_avg + 4*train_std)

print 'train shape ',x_train.shape          
print 'test shape ',x_test_vals.shape         

maxSamples = (np.floor(x_train.shape[0]/batch_size)*batch_size).astype(np.int64)
x_train = x_train[0:maxSamples,:]
maxSamples = (np.floor(x_test_vals.shape[0]/batch_size)*batch_size).astype(np.int64)
x_test_vals = x_test_vals[0:maxSamples,:]

valinds=np.random.permutation(range(x_train.shape[0]))
valinds=valinds[:100]#params['batch_size']]
x_train = (x_train-0.0007)/0.04
X1 = x_train[valinds, :x_train.shape[1]/3]
X1t = x_train[valinds, (x_train.shape[1]/3):(2*x_train.shape[1]/3)]
X2 = x_train[valinds, (2*x_train.shape[1]/3):x_train.shape[1]]
lbls = id_train[valinds]
print 'X1 shape ',X1.shape     
#zn=np.random.randn(100,params['latent_dim'])#
#zn=(0.0001*np.random.randn(100,params['latent_dim']))+10
zn=(1*np.random.randn(100,params['latent_dim']))+0#10

z1_m,z1_std,s1 = net.VAEencS.predict(X1,batch_size=params['batch_size'])
X11,X11t,X12,Xp2,adv1,adv2 = net.DistNet.predict([X1,X1t,X2,zn,lbls],batch_size=params['batch_size'])
print 'z1mue: '+ str(np.average(z1_m))+' z1std: '+ str(np.average(z1_std))
print 'loss1: '+ str(np.average(np.square(X11-X1)))+' prod: ' +str(np.average(np.square(np.prod(0.04*X1+1,axis=1)-np.prod(0.04*X11+1,axis=1)))) + ' loss2: '+ str(np.average(np.square(X11t-X1)))+' prod: ' +str(np.average(np.square(np.prod(0.04*X1+1,axis=1)-np.prod(0.04*X11t+1,axis=1)))) 
#np.average(np.square(X11-X1),axis=0)

mixinds=np.random.permutation(range(X1.shape[0]))
print 'loss1: '+ str(np.average(np.square(X11-X1[mixinds,:])))+' prod: ' +str(np.average(np.square(np.prod(0.04*X11+1,axis=1)-np.prod(0.04*X1[mixinds,:]+1,axis=1))))+' loss2: '+ str(np.average(np.square(X11t-X1[mixinds,:])))+' prod: ' +str(np.average(np.square(np.prod(0.04*X1[mixinds,:]+1,axis=1)-np.prod(0.04*X11t+1,axis=1)))) 
print 'adv1: ',np.average(adv1),np.min(adv1),np.max(adv1),np.std(adv1),' adv2: ',np.average(adv2),np.min(adv2),np.max(adv2),np.std(adv2)

X11,X11t,X12,Xp2,adv1,adv2 = net.DistNet.predict([X1,X2,X2,zn,lbls],batch_size=params['batch_size'])
print 'loss1: '+ str(np.average(np.square(X11-X1)))+' prod: ' +str(np.average(np.square(np.prod(0.04*X1+1,axis=1)-np.prod(0.04*X11+1,axis=1)))) + ' loss2: '+ str(np.average(np.square(X11t-X1)))+' prod: ' +str(np.average(np.square(np.prod(0.04*X1+1,axis=1)-np.prod(0.04*X11t+1,axis=1)))) 
