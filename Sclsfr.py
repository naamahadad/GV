#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 13 14:49:06 2017

@author: naamahadad
"""

from keras.models import Model
from keras import backend as K
from keras import objectives

import numpy as np

from keras.objectives import categorical_crossentropy
from keras.layers import Activation,Input, Dense, Lambda, Dropout, merge
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
class Sclsfr(object):
    
    def log_results(self,log_file,res_dict,debug=False,display_on=False):
        for key,val in res_dict.iteritems():
            #print '%s: %.4f' % (key,val)
            if self.save_files: log_file.write(key + ' ' + str(val) + ' ')
	    if display_on:
                print key + ' ' + str(val) + '\n'
        #print ''
        if self.save_files: log_file.write('\n')
        if self.save_files: log_file.flush()
    def log_weights(self,cur_loss):
        if self.save_files:
            self.min_loss = cur_loss
            self.Net.save(self.weights_file + '_weights.h5')
            print 'saving files'
        
    def __init__(self, params,outfile,weights_file,save_files):
        batch_size=params['batch_size']
        original_dim=params['original_dim']
        intermediate_dim=params['intermediate_dim']
        num_clss=params['num_clss']
        init = params['Init']#eval(params['Init']+'()')
        dp = params['dp']
        if 'LeakyReLU' in params['Activation']:
            act = eval(params['Activation']+'()')
        else:
            act = Activation(params['Activation'])
        use_bn = params['BN']
        is_reg = params['reg']
        
        def add_dense_layer(inp,dim,out_dp=1):
            h = Dense(dim,init=init)(inp)
            if use_bn:
                h = BatchNormalization()(h)
            h = act(h)
            if out_dp==1 and dp>0:
                h = Dropout(dp)(h)
            return h
        
        in_s = Input(batch_shape=(batch_size, original_dim))
        
        #Enc ########################################
        h = add_dense_layer(in_s,intermediate_dim)
        h = add_dense_layer(h,intermediate_dim)
        h = add_dense_layer(h,2*intermediate_dim/3)
        h = add_dense_layer(h,2*intermediate_dim/3)

        if is_reg:
            h = Dense(1,init=init)(h)
            out = Activation('sigmoid')(h)
        else:
            h = add_dense_layer(h,num_clss,out_dp=0)
            h = Dense(num_clss,init=init)(h)
            out = Activation('softmax')(h)
        
        print 'compiling classifier...'
        self.Net = Model(in_s, output=out)
        if is_reg:
            self.Net.compile(optimizer=Adam(),loss='mse')
        else:
            self.Net.compile(optimizer=Adam(),loss='categorical_crossentropy',metrics=['accuracy'])
        
        self.params = params
        self.outfile = outfile
        self.save_files = save_files
        if self.save_files:
            self.log_results(self.outfile,params,debug=False)
            self.weights_file = weights_file
        if is_reg:
            self.checkpointer = ModelCheckpoint(filepath=weights_file, monitor='val_loss', verbose=1, save_best_only=True)
        else:
            self.checkpointer = ModelCheckpoint(filepath=weights_file, monitor='val_acc', verbose=1, save_best_only=True)