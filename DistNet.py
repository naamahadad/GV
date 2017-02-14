# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 13:37:17 2016

@author: algo
"""

from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD, Adam, Adadelta,RMSprop
from keras.layers.advanced_activations import LeakyReLU
from keras.utils import np_utils
from keras.objectives import categorical_crossentropy
from keras.layers import Activation,Input, Dense, Lambda, Dropout, merge
from keras.utils import np_utils, generic_utils
from keras.models import Model
from keras import backend as K
from keras import objectives
from keras.initializations import glorot_uniform

#from keras.utils.layer_utils import print_layer_shapes
import numpy as np
#import tensorflow as tf
import os

class DistNet(object):
    
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
            self.DistNet.save(self.weights_file + '_full_weights.h5')
            self.VAEencS.save(self.weights_file + '_encS_weights.h5')
            self.VAEdecS.save(self.weights_file + '_decS_weights.h5')
            self.Adv.save(self.weights_file + '_adv_weights.h5')
            self.AdvNet.save(self.weights_file + '_advNet_weights.h5')
            print 'saving files'
        
    def __init__(self, params,outfile,weights_file,save_files):
        epsilon_std = 1.0
        batch_size=params['batch_size']
        original_dim=params['original_dim']
        intermediate_dim=params['intermediate_dim']
        latent_dim=params['latent_dim']
        s_dim=params['s_dim']
        l_size=params['l_size']
        init = params['Init']#eval(params['Init']+'()')
        if 'LeakyReLU' in params['Activation']:
            act = eval(params['Activation']+'()')
        else:
            act = Activation(params['Activation'])
        use_bn = params['BN']
        use_bn_orig=use_bn
        
        def add_dense_layer(inp,dim):
            h = Dense(dim,init=init)(inp)
            if use_bn:
                h = BatchNormalization(mode=2)(h)
            h = act(h)
            return h
        
        def clipping(args):
            vals = args
            return K.clip(vals,-30,30)
        
        in_x = Input(batch_shape=(batch_size, original_dim))
        in_lbl = Input(batch_shape = (batch_size,1))
        
        #Enc ########################################
        h = add_dense_layer(in_x,intermediate_dim)
        h = add_dense_layer(h,intermediate_dim)
        h = add_dense_layer(h,2*intermediate_dim/3)
        h = add_dense_layer(h,2*intermediate_dim/3)

        z_mean = Dense(latent_dim,init=init)(h)
        z_log_var = Dense(latent_dim,init=init)(h)
        s = Dense(s_dim,init=init)(h)
        
        self.VAEencS = Model(in_x, output=[z_mean,z_log_var,s])
        
        #Dec ########################################
        in_z = Input(batch_shape=(batch_size, latent_dim))
        in_s = Input(batch_shape=(batch_size, s_dim))
        
        inz_s = merge([in_z, in_s], mode='concat', concat_axis=1)
        decoder_h = add_dense_layer(inz_s,intermediate_dim)
        decoder_h = add_dense_layer(decoder_h,intermediate_dim)
        decoder_h = add_dense_layer(decoder_h,2*intermediate_dim/3)
        decoder_h = add_dense_layer(decoder_h,2*intermediate_dim/3)
        x_decoded_mean = Dense(original_dim,init=init,activation='linear')(decoder_h)#, activation='relu',activation='tanh'
        x_decoded_mean = Lambda(clipping, output_shape=(original_dim,))(x_decoded_mean)
        #x_decoded_log_std = Dense(original_dim, activation='relu',init='glorot_uniform')(decoder_h)
        #logpxz = 0.5* tf.reduce_sum(x_decoded_log_std + tf.square(in_x - x_decoded_mean)/tf.exp(x_decoded_log_std))
        
        self.VAEdecS = Model([in_z,in_s],[x_decoded_mean])#,x_decoded_log_std])#logpxz
        
        #Adv ########################################
        use_bn = True
        inx_lbl = merge([in_x, in_lbl], mode='concat', concat_axis=1)  
        adv_h = add_dense_layer(inx_lbl,l_size)
        adv_h = add_dense_layer(adv_h,l_size)
        adv_h = add_dense_layer(adv_h,l_size)
        use_bn = use_bn_orig
        disc = Dense(1, activation='sigmoid',init=init)(adv_h)
        
        self.Adv = Model([in_x,in_lbl],disc)
        ########################################
        
        
        print 'building encs...'
        x1 = Input(batch_shape=(batch_size, original_dim))
        x1t = Input(batch_shape=(batch_size, original_dim))
        x2 = Input(batch_shape=(batch_size, original_dim))
        zn = Input(batch_shape=(batch_size, latent_dim))
        x12in = Input(batch_shape=(batch_size, original_dim))
        xp2in = Input(batch_shape=(batch_size, original_dim))
        rho1 = Input(batch_shape=(batch_size, s_dim))
        rho2 = Input(batch_shape=(batch_size, s_dim))
            
        z1_mean,z1_log_var,s1 = self.VAEencS(x1)
        z1t_mean,z1t_log_var,s1t = self.VAEencS(x1t)
        z2_mean,z2_log_var,s2 = self.VAEencS(x2)
        
        def weight_tens(args):
            s1,s1t,rho1,rho2 = args
            return rho1*s1 + rho2*s1t
        
        if params['norm_sim']:
            s1t = Lambda(weight_tens, output_shape=(s_dim,))([s1,s1t,rho1,rho2])
            #s1t = rho1*s1 + rho2*s1t
            
        def sampling(args):
            z_mean, z_log_var = args
            epsilon = K.random_normal(shape=(batch_size, latent_dim), mean=0.,
                                      std=epsilon_std)
            return z_mean + K.exp(z_log_var / 2) * epsilon
        
        # note that "output_shape" isn't necessary with the TensorFlow backend
        z1 = Lambda(sampling, output_shape=(latent_dim,))([z1_mean, z1_log_var])
        z1 = z1_mean
        #zn=np.random.randn(batch_size,latent_dim)
        #zn=Lambda(lambda x: x)(K.random_normal(shape=(batch_size, latent_dim), mean=0.,std=1))
           
        print 'building dec...'
        x11 = self.VAEdecS([z1,s1])#,x11_log_std
        x11t = self.VAEdecS([z1,s1t])#,x11t_log_std
        x12 = self.VAEdecS([z1,s2])#,x12_log_std
        xp2 = self.VAEdecS([zn,s2])#,xp2_log_std
        
        print 'building Adv...'
        Adv1 = self.Adv([x12,in_lbl])
        Adv2 = self.Adv([xp2,in_lbl])
        
        Adv1_netAdv = self.Adv([x12in,in_lbl])
        Adv2_netAdv = self.Adv([xp2in,in_lbl])
        
        recweight = params['recweight'] if 'recweight' in params else 0.5
        swap1weight = params['swap1weight'] if 'swap1weight' in params else 1
        swap2weight = params['swap2weight'] if 'swap2weight' in params else 0.01
        klweightZ = params['klweightZ'] if 'klweightZ' in params else 0.1     
        prodweight = params['prodweight'] if 'prodweight' in params else 0.1  
        
        #def loss1(inputs,outputs):
            #return 0.5* tf.reduce_sum(x11_log_std + tf.square(x1 - x11)/tf.exp(x11_log_std))
            #return tf.reduce_sum(tf.square(x1 - x11))
        #def loss2(inputs,outputs):
            #return 0.5* tf.reduce_sum(x11t_log_std + tf.square(x1 - x11t)/tf.exp(x11t_log_std))
            #return tf.reduce_sum(tf.square(x1 - x11t))
        def loss3(inputs,outputs):
            return - 0.5 * K.sum(1 + z1_log_var - K.square(z1_mean) - K.exp(z1_log_var), axis=-1)
            #return - 0.5 * tf.reduce_sum(1 + z1_log_var - tf.square(z1_mean) - tf.exp(z1_log_var))
        def lossProd(inputs,outputs):
            origCum = K.prod(0.04*x1+1,axis=1)
            x11Cum = K.prod(0.04*x11+1,axis=1)
            x11tCum = K.prod(0.04*x11t+1,axis=1)
            #return K.mean(K.square(x11Cum-origCum))
            return 0.5*(K.mean(K.square(x11Cum-origCum))+K.mean(K.square(x11tCum-origCum)))
#        def enc_dec_loss(inputs,outputs):
#            #xent_loss = original_dim * objectives.binary_crossentropy(x, x_decoded_mean)
#            x1_recon = 0.5* tf.reduce_sum(x11_log_std + tf.square(x1 - x11)/tf.exp(x11_log_std))
#            x1t_recon = 0.5* tf.reduce_sum(x11t_log_std + tf.square(x1 - x11t)/tf.exp(x11t_log_std))
#            kl_loss = - 0.5 * K.sum(1 + z1_log_var - K.square(z1_mean) - K.exp(z1_log_var), axis=-1)
#            l1 = -tf.reduce_sum(tf.log(1-Adv1))
#            l2 = -tf.reduce_sum(tf.log(1-Adv2))
#            enc_dec_loss =swap2weight*(l1+l2)#recweight*x1_recon + swap1weight*x1t_recon + klweightZ*kl_loss + swap2weight*(K.log(Adv1)+K.log(Adv2))
#            #kl_loss1 = - 0.5 * K.sum(1 + z1t_log_var - K.square(z1t_mean) - K.exp(z1t_log_var), axis=-1)
#            return enc_dec_loss
        def adv_loss(inputs,outputs):
            #return K.log(1-Adv1_netAdv) + K.log(Adv2_netAdv)
            l1 = K.mean(objectives.binary_crossentropy(Adv1_netAdv,np.ones((batch_size,))))
            l2 = K.mean(objectives.binary_crossentropy(Adv2_netAdv,np.zeros((batch_size,))))
            return l1+l2
        #def loss2(inputs,outputs):
        #    return -K.mean(y_true*K.log(y_pred)+(1-y_true)*K.log(1-y_pred));
            #return K.mean(K.square(y_pred - y_true))#
        print 'compile...'
        if params['norm_sim']:
            self.DistNet = Model([x1,x1t,x2,zn,in_lbl,rho1,rho2], [x11,x11t,x12,xp2,Adv1,Adv2])
        else:
            self.DistNet = Model([x1,x1t,x2,zn,in_lbl], [x11,x11t,x12,xp2,Adv1,Adv2])
        self.freeze_unfreeze_Adv(False)
        self.freeze_unfreeze_EncDec(True)
        opt = Adam(lr=0.001, beta_1=0.9)
        #opt = RMSprop(lr=0.0001)
        self.DistNet.compile(optimizer=opt, loss=['mse','mse',loss3,lossProd,'binary_crossentropy','binary_crossentropy'],
                             loss_weights=[recweight,swap1weight,klweightZ,prodweight,swap2weight,swap2weight])#loss=enc_dec_loss)   
    
        self.AdvNet = Model([x12in,xp2in,in_lbl],[Adv1_netAdv,Adv2_netAdv])
        self.freeze_unfreeze_EncDec(False)
        self.freeze_unfreeze_Adv(True)
        #opt = RMSprop(lr=0.0001)
        opt = SGD(lr=0.0005, momentum=0.9, nesterov=True)
        self.AdvNet.compile(optimizer='sgd', loss=['binary_crossentropy','binary_crossentropy'],loss_weights=[0.5,0.5])#loss#adv_loss)  
        
        self.params = params
        self.outfile = outfile
        self.save_files = save_files
        if self.save_files:
            self.log_results(self.outfile,params,debug=False)
            self.weights_file = weights_file
        
        #self.DistNet.summary()
        #self.AdvNet.summary()
    def predictEnc(self,X1):
        
        batch_size = self.params['batch_size']
        #z1_m_full = np.empty(0)
        #z1_std_full = np.empty(0)
        #s1_full = np.empty(0)
        #for (first, last) in zip(range(0, X1.shape[0]-batch_size, batch_size),
         #   range(batch_size, X1.shape[0], batch_size)):
            
        z1_m,z1_std,s1 = self.VAEencS.predict(X1,batch_size=batch_size)
        print 'z1_m shape ',z1_m.shape
        print 'z1_s shape ',z1_std.shape
        print 's1 shape ',s1.shape

#            z1_m_full = z1_m if first==0 else np.concatenate((z1_m_full,z1_m),axis=0)
#            z1_std_full = z1_std if first==0 else np.concatenate((z1_std_full,z1_std),axis=0)
#            s1_full = s1 if first==0 else np.concatenate((s1_full,s1),axis=0)
#            print 'z1_m shape ',z1_m_full.shape
#            print 'z1_s shape ',z1_std_full.shape
#            print 's1 shape ',s1_full.shape
            
        return z1_m, z1_std, s1#z1_m_full, z1_std_full, s1_full
        
    def train(self,nb_epoch,X_train,X_valid,X_lbls,rho1_train,rho2_train):
        
        train_params={}
        train_params['X_train_len'] = X_train.shape[0]
        train_params['data_cols'] = X_train.shape[1]
        train_params['X_valid_len'] = X_valid.shape[0]
        train_params['X_avg'] = np.average(np.average(X_train,axis=1),axis=0)
        train_params['X_std'] = np.std(X_train)
        
        rho1_train = np.repeat(np.expand_dims(rho1_train,axis=1),self.params['s_dim'],axis=1)
        rho2_train = np.repeat(np.expand_dims(rho2_train,axis=1),self.params['s_dim'],axis=1)
        
        #print 'train nan',np.sum(np.isnan(X_train))
        #print 'train max',np.max(X_train)
        #print 'train min',np.min(X_train)
        #X_train = (X_train - train_params['X_avg'])/train_params['X_std']
        #X_valid = (X_valid - train_params['X_avg'])/train_params['X_std']
        normavg = 0.0007
        normstd = 0.04
        
        X_train = (X_train - normavg)/normstd
        X_valid = (X_valid - normavg)/normstd
        train_params['X_new_avg'] =np.average(np.average(X_train,axis=1),axis=0)
        train_params['X_new_std'] = np.std(X_train)
        
        #print 'train nan',np.sum(np.isnan(X_train))
        print 'train max',np.max(X_train)
        print 'train min',np.min(X_train)
        
        self.log_results(self.outfile,train_params,debug=False,display_on=True)
        #home = os.path.expanduser('~')
        #modelp = home + '/results_nn/weights/'
        #progbar = generic_utils.Progbar(1000)
      
        self.min_loss = 100000000
        batch_size = self.params['batch_size']
        stopat=500000
        print 'l1:', min(X_train.shape[0]-batch_size,stopat-batch_size)
        print 'l2:', min(X_train.shape[0],stopat)
        for e in range(nb_epoch):
            loss0 = 0
            loss1 = 0
            mb=0
            perm_inds = np.random.permutation(range(X_train.shape[0]))
            X_train = X_train[perm_inds,:]
            savebatch=True
            for (first, last) in zip(range(0, min(X_train.shape[0]-batch_size,stopat-batch_size), batch_size),
                                     range(batch_size, min(X_train.shape[0],stopat), batch_size)):
                X1 = X_train[first: last, :X_train.shape[1]/3]
                X1t = X_train[first: last, (X_train.shape[1]/3):(2*X_train.shape[1]/3)]
                X2 = X_train[first: last, (2*X_train.shape[1]/3):X_train.shape[1]]
                lbls = X_lbls[first: last]
                rho1=rho1_train[first: last]
                rho2=rho2_train[first: last]
                
                mue_vec = 6*((np.round(np.random.rand(self.params['batch_size'],self.params['latent_dim'])*10,0)/10) -0.5)
                zn=(1*np.random.randn(self.params['batch_size'],self.params['latent_dim']))+mue_vec
                #uh = 2
                #if mb % uh == 0:
                if self.params['norm_sim']:
                    loss0batch= self.DistNet.train_on_batch([X1,X1t,X2,zn,lbls,rho1,rho2],[X1,X1,X2,zn,np.zeros((self.params['batch_size'],1)),np.zeros((self.params['batch_size'],1))])[0]
                    X11,X11t,X12,Xp2,adv1,adv2 = self.DistNet.predict([X1,X1t,X2,zn,lbls,rho1,rho2],batch_size=self.params['batch_size'])
                else:
                    loss0batch= self.DistNet.train_on_batch([X1,X1t,X2,zn,lbls],[X1,X1,X2,zn,np.zeros((self.params['batch_size'],1)),np.zeros((self.params['batch_size'],1))])[0]
                    X11,X11t,X12,Xp2,adv1,adv2 = self.DistNet.predict([X1,X1t,X2,zn,lbls],batch_size=self.params['batch_size'])
                loss0 +=loss0batch
                
                #if e==0 and first==0:
                #    print_layer_shapes(self.DistNet,input_shapes=[X1.shape,X1t.shape,X2.shape,zn.shape,lbls.shape])
                #    print_layer_shapes(self.AdvNet,input_shapes=[X12.shape,Xp2.shape,lbls.shape])
                
                #else:
                loss1batch= self.AdvNet.train_on_batch([X12,Xp2,lbls],[np.ones((self.params['batch_size'],1)),np.zeros((self.params['batch_size'],1))])[0]
                #adv1_a,adv2_a = self.AdvNet.predict([X12,Xp2,lbls],batch_size=self.params['batch_size'])
                loss1 +=loss1batch
                mb = mb+1
                #if mb==1:
                #    print 'adv1: ',np.average(adv1),np.min(adv1),np.max(adv1),np.std(adv1),' adv2: ',np.average(adv2),np.min(adv2),np.max(adv2),np.std(adv2)
                if e==0 and mb<10:
                    print '[' + str(round(loss0batch,3)) + ' , '+ str(round(loss1batch,3))+ ']'
                if e>3 and savebatch :#(np.isnan(loss0batch) or np.isnan(loss1batch)) :
                    np.save('/media/data2/naamahadad/results/Debug/X1',X1)
                    np.save('/media/data2/naamahadad/results/Debug/X11',X11)
                    np.save('/media/data2/naamahadad/results/Debug/adv1',adv1)
                    np.save('/media/data2/naamahadad/results/Debug/adv2',adv2)
                    savebatch = False

            loss = loss0 + 10*loss1                
            loss0 = loss0/mb
            loss1 = loss1/mb
            loss = loss/mb
            if loss0<self.min_loss and e>8:
                self.log_weights(loss0)
            print 'epoch: ' +str(e)+' train loss: '+str(round(loss,3)) +', G loss: ' + str(round(loss0,3)) + ', D loss: '+ str(round(loss1,3))+ ', min G loss: '+ str(round(self.min_loss,3))
            print 'adv1: ',np.average(adv1),np.min(adv1),np.max(adv1),np.std(adv1),' adv2: ',np.average(adv2),np.min(adv2),np.max(adv2),np.std(adv2)
            #print 'adv1a: ',np.average(adv1_a),np.min(adv1_a),np.max(adv1_a),np.std(adv1_a),' adv2a: ',np.average(adv2_a),np.min(adv2_a),np.max(adv2_a),np.std(adv2_a)

            
            results={}
            results['epoch'] = e
            results['tot_loss'] = round(loss,5)
            results['loss0'] =  round(loss0,5)
            results['loss1'] =  round(loss1,5)
            results['min loss'] =  round(self.min_loss,3)
            results['adv1'] =  round(np.average(adv1),3)
            results['adv2'] =  round(np.average(adv2),3)
            self.log_results(self.outfile,results)

            #progbar.add(1, values=[("train loss", loss),
            #                       ("G loss", loss0),
            #                       ("D loss", loss1)])
                                         
        return loss
    def get_nets(self):
        return self.DistNet,self.AdvNet
    def freeze_unfreeze_Adv(self,trainable = False):
        self.Adv.trainable = trainable
        for l in self.Adv.layers:
            l.trainable = trainable
        
    def freeze_unfreeze_EncDec(self,trainable = False):
        self.VAEencS.trainable = trainable
        self.VAEdecS.trainable = trainable
        for l in self.VAEdecS.layers:
            l.trainable = trainable
        for l in self.VAEencS.layers:
            l.trainable = trainable

