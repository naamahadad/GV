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

home = os.path.expanduser('~')

params = {}
debug = True

params['config_file'] = sys.argv[1] if len(sys.argv)>1 else 'config_adam_epoch5000.yaml'
params['data'] = '/media/data2/naamahadad/PyData/1976_2016.hdf'#[home +'/FinData/prices_debug.hdf']

#with open('yamls/' + params['config_file'],'r') as f:
#    params.update(yaml.load(f))

params['res_path'] = '/media/data2/naamahadad/results/MainVAE'
params['years_dict'] = {'train_top' : 2012, # 2009
                      'test_bottom' : 2013, # 2010
                      'test_top' : 2015} # 2012
valid_years = np.arange(2010,2017,1)#np.arange(1986,2016,1)#xrange(31)+1985#[1985,1990,1995,2000,2005,2010,2015]
params['BN'] = True
params['Init'] = 'glorot_normal'
params['Activation'] = 'tanh'

params['recweight'] = 50#50#0.5#135,45
params['swap1weight'] = 100#100#1#280,93
params['swap2weight'] = 0.5#0.5#5#1#0.01#1.2,1.14
params['klweightZ'] = 0#0#0.5#0.25#10#0.1#0.121-->0,0.176
params['prodweight'] = 60

batch_size = 100
nb_epoch = 5000
eleven_month = True
params['batch_size'] = batch_size
params['nb_epoch'] = nb_epoch

if eleven_month:
    params['original_dim'] = 230
    params['intermediate_dim'] = 120
else:    
    params['original_dim'] = 250
    params['intermediate_dim'] = 150
params['s_dim'] = 30
params['l_size'] = 60
params['latent_dim'] = 30
params['norm_sim'] = False
curtime = strftime("%d%m%y_%H%M%S", localtime())
log_filename = params['res_path'] +'/'+ curtime + '_' + params['config_file'] + '_keras_distnet.txt'
weihts_filename = params['res_path'] +'/'+ curtime + '_' + params['config_file'] + '_weights.h5'
outfile=open(log_filename,'a')

#vae = KerasModels.VAE(params,batch_size,original_dim,intermediate_dim,latent_dim)
net = DistNet.DistNet(params,outfile,weihts_filename,True)

data_tags = ['day'+str(i) for i in range(0,250)]

print 'loading data...'
datafactory = dataFactory_pyprep.dataFactory_pyprep(data_path=params['data'],years_dict=params['years_dict'],data_tags=data_tags,s_mode=1,load_ready_data=False,valid_years=valid_years,eleven_month=eleven_month)
x_train,id_train,rho1_train,rho2_train = datafactory.get_train_data()
x_test,id_test,rho1_test,rho2_test = datafactory.get_test_data()
x_test_vals = x_test#x_test[data_tags].values

print 'train nan',np.sum(np.isnan(x_train))
print 'test nan',np.sum(np.isnan(x_test_vals))

x_train[np.isnan(x_train)]=0
x_test_vals[np.isnan(x_test_vals)]=0
train_std = 0.04
train_avg = 0.0007
x_train = np.clip(x_train,train_avg- 4*train_std,train_avg + 4*train_std)
x_test_vals = np.clip(x_test_vals,train_avg- 4*train_std,train_avg + 4*train_std)

maxSamples = (np.floor(x_train.shape[0]/batch_size)*batch_size).astype(np.int64)
x_train = x_train[0:maxSamples,:]
maxSamples = (np.floor(x_test_vals.shape[0]/batch_size)*batch_size).astype(np.int64)
x_test_vals = x_test_vals[0:maxSamples,:]

loss = net.train(nb_epoch,x_train,x_test_vals,id_train,rho1_train,rho2_train)

outfile.close()  
#x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
#x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

#checkpointer = keras.callbacks.ModelCheckpoint(filepath=home + '/results_nn/VAE/weights/weightsVAE.hdf5', verbose=1, save_best_only=True)

#hist = vae.fit(x_train, x_train,shuffle=True,nb_epoch=nb_epoch,batch_size=batch_size,
#            verbose=0,validation_data=(x_test_vals, x_test_vals),callbacks=[checkpointer])

## build a model to project inputs on the latent space
#encoder = Model(x, z_mean)
#
## display a 2D plot of the digit classes in the latent space
#x_test_encoded = encoder.predict(x_test, batch_size=batch_size)
#plt.figure(figsize=(6, 6))
#plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], c=y_test)
#plt.colorbar()
#plt.show()
#
## build a digit generator that can sample from the learned distribution
#decoder_input = Input(shape=(latent_dim,))
#_h_decoded = decoder_h(decoder_input)
#_x_decoded_mean = decoder_mean(_h_decoded)
#generator = Model(decoder_input, _x_decoded_mean)
#
## display a 2D manifold of the digits
#n = 15  # figure with 15x15 digits
#digit_size = 28
#figure = np.zeros((digit_size * n, digit_size * n))
## linearly spaced coordinates on the unit square were transformed through the inverse CDF (ppf) of the Gaussian
## to produce values of the latent variables z, since the prior of the latent space is Gaussian
#grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
#grid_y = norm.ppf(np.linspace(0.05, 0.95, n))
#
#for i, yi in enumerate(grid_x):
#    for j, xi in enumerate(grid_y):
#        z_sample = np.array([[xi, yi]])
#        x_decoded = generator.predict(z_sample)
#        digit = x_decoded[0].reshape(digit_size, digit_size)
#        figure[i * digit_size: (i + 1) * digit_size,
#               j * digit_size: (j + 1) * digit_size] = digit
#
#plt.figure(figsize=(10, 10))
#plt.imshow(figure, cmap='Greys_r')
#plt.show()
