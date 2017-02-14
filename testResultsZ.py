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
from sklearn import linear_model,ensemble,tree
from sklearn.svm import SVC

#from keras.models import load_model

home = os.path.expanduser('~')

params = {}
debug = True

params['config_file'] = sys.argv[1] if len(sys.argv)>1 else 'config_adam_epoch5000.yaml'
#params['data'] = [home +'/FinData/prices_debug.hdf']
params['data'] = '/media/data2/naamahadad/PyData/1976_2016.hdf'

#with open('yamls/' + params['config_file'],'r') as f:
#    params.update(yaml.load(f))

#params['res_path'] = home + '/results_nn/VAE'
params['res_path'] = '/media/data2/naamahadad/results/MainVAE'#Debug'#MainVAE
params['years_dict'] = {'train_top' : 2012, # 2009
                      'test_bottom' : 2013, # 2010
                      'test_top' : 2017} # 2012

params['BN'] = True
params['Init'] = 'glorot_normal'
params['Activation'] = 'tanh'

params['recweight'] = 50#0.5#135,45
params['swap1weight'] = 100#100#1#280,93
params['swap2weight'] = 0.25#1#0.01#1.2,1.14
params['klweightZ'] = 60#10#0.1#0.121-->0,0.176

batch_size = 100
nb_epoch = 1000
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

num_days_mode = 0
params_clsfr={}
params_clsfr['BN'] =True
params_clsfr['Init'] = 'glorot_normal'
params_clsfr['Activation'] = 'tanh'
params_clsfr['batch_size'] = batch_size
params_clsfr['nb_epoch'] = nb_epoch

params_clsfr['original_dim'] = params['latent_dim']
params_clsfr['intermediate_dim'] = 30
params_clsfr['num_clss'] = 3#3+(num_days_mode >2)
params_clsfr['dp'] = 0.5
params_clsfr['reg'] = False
params_clsfr['mainnet'] ='130217_203703'

load_clsfr = False
params_clsfr['res_path'] = '/media/data2/naamahadad/results/EncRes'
if load_clsfr:
    clsfrNet = Sclsfr.Sclsfr(params_clsfr,'','',False)
    #200117_082140_ normal net day
    #270117_194145_ z onl+prod
    #280117_164300 mix s
    #290117_184028_ softmax
    clsfrNet.Net.load_weights(params_clsfr['res_path'] + '/210117_201514_config_adam_epoch5000.yaml_Zclsfr_day_weights.h5')
else:
    curtime = strftime("%d%m%y_%H%M%S", localtime())
    log_filename = params_clsfr['res_path']  +'/'+ curtime + '_' + params['config_file'] + '_keras_Zclsfr.txt'
    weihts_filename = params_clsfr['res_path'] +'/'+ curtime + '_' + params['config_file'] + '_Zclsfr_day_weights.h5'
    outfile=open(log_filename,'a')
    
    clsfrNet = Sclsfr.Sclsfr(params_clsfr,outfile,weihts_filename,True)

#170117_111424_ "" +clip+no std for z
#170117_200219_ 230 days
#220117_204007_ z only loss
#240117_215706_ z,s wo prod
#260117_120441_ z only with prod
#270117_231627_ all losses, combine s1,s1'
#280117_222954_ "" softmax
#300117_211011_ softmax + prod x1t loss
#090217_191209 normal net up to 2009
#130217_203703 normal net up to 2013
print 'loading net...'
net.AdvNet.load_weights(params['res_path'] +'/130217_203703_config_adam_epoch5000.yaml_weights.h5_advNet_weights.h5')
net.Adv.load_weights(params['res_path'] +'/130217_203703_config_adam_epoch5000.yaml_weights.h5_adv_weights.h5')
net.DistNet.load_weights(params['res_path'] +'/130217_203703_config_adam_epoch5000.yaml_weights.h5_full_weights.h5')

data_tags = ['day'+str(i) for i in range(0,250)]

print 'loading data...'
valid_years = np.asarray([])#1985,1990,1995,2000,2005,2010,2015])#-1
datafactory = dataFactory_pyprep.dataFactory_pyprep(data_path=params['data'],years_dict=params['years_dict'],data_tags=data_tags,s_mode=1,load_ready_data=True,valid_years=valid_years,calc_beta=True)
print 'loading data group...'
x_train,id_train,clss_train,ret_train,beta_tr,rho_tr,x_test,id_test,clss_test,ret_test,beta_te,rho_te = datafactory.get_samples_per_group(100000,ret_rho=True)
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

if num_days_mode==0:
    ret_train = x_train_orig[:,x_train.shape[1]/3-20]
    ret_test = x_test_orig[:,x_train.shape[1]/3-20]
elif num_days_mode==1:
    ret_train = np.sum(x_train_orig[:,(x_train.shape[1]/3-20):(x_train.shape[1]/3-15)],axis=1)
    ret_test = np.sum(x_test_orig[:,(x_train.shape[1]/3-20):(x_train.shape[1]/3-15)],axis=1)
elif num_days_mode==3:
    ret_train = beta_tr
    ret_test = beta_te
elif num_days_mode==4:
    ret_train = np.sum(x_train_orig[:,(x_train.shape[1]/3-20):(x_train.shape[1]/3-15)],axis=1)
    ret_test = np.sum(x_test_orig[:,(x_train.shape[1]/3-20):(x_train.shape[1]/3-15)],axis=1)
    ret_train1 = np.sum(x_train_orig[:,((2*x_train.shape[1]/3)-20):((2*x_train.shape[1]/3)-15)],axis=1)
    ret_test1 = np.sum(x_test_orig[:,(2*(x_train.shape[1]/3)-20):((2*x_train.shape[1]/3)-15)],axis=1)
elif num_days_mode==5: #Rho
    ret_train = rho_tr
    ret_test = rho_te
  
retavg1=-10
retavg2=-10
retavg3=-10
if params_clsfr['num_clss']==3:
    retavg1 = np.percentile(ret_train,33)
    retavg2 = np.percentile(ret_train,66)
    clss_train = 1*((ret_train - retavg1)>0)+1*((ret_train - retavg2)>0)
    clss_test = 1*((ret_test - retavg1)>0)+1*((ret_test - retavg2)>0)
elif params_clsfr['num_clss']==2:
    retavg1 = np.percentile(ret_train,50)
    clss_train = 1*((ret_train - retavg1)>0)
    clss_test = 1*((ret_test - retavg1)>0)  
elif params_clsfr['num_clss']==4:
    retavg1 = np.percentile(ret_train,25)
    retavg2 = np.percentile(ret_train,50)
    retavg3 = np.percentile(ret_train,75)
    clss_train = 1*((ret_train - retavg1)>0)+1*((ret_train - retavg2)>0)+1*((ret_train - retavg3)>0)
    clss_test = 1*((ret_test - retavg1)>0)+1*((ret_test - retavg2)>0)+1*((ret_test - retavg3)>0)
print 'day230 avg1: ' ,retavg1,'avg2: ' ,retavg2, ' clss_tr_avg: ' ,np.average(clss_train), ' clss_te_avg: ' ,np.average(clss_test)

z1_m_full, z1_std_full, s1_full = net.predictEnc(X1)
z1_m_te, z1_std_te, s1_te = net.predictEnc(X1_te)
z1t_m_full, z1t_std_full, s1t_full = net.predictEnc(X1t)
z1t_m_te, z1t_std_te, s1t_te = net.predictEnc(X1t_te)

print 'z1mue: '+ str(np.average(z1_m_full))+' std: '+ str(np.std(z1_m_full))+' max: '+ str(np.max(np.abs((z1_m_full))))
print 'z1std: '+ str(np.average(z1_std_full))+' std: '+ str(np.std(z1_std_full))+' max: '+ str(np.max(np.abs((z1_std_full))))
#print 's1mean: '+ str(np.average(s1_full))+' std: '+ str(np.std(s1_full))+' max: '+ str(np.max(np.abs((s1_full))))

dataroot='/media/data2/naamahadad/results/EncRes/'
clsfr_mode=0
savename = '_2013_2016_net_svm'
if clsfr_mode==1: #MSE
    X1_est = net.VAEdecS.predict([z1_m_full,s1_full],batch_size=batch_size)
    X1_est_te = net.VAEdecS.predict([z1_m_te,s1_te],batch_size=batch_size)
    X1t_est = net.VAEdecS.predict([z1_m_full,s1t_full],batch_size=batch_size)
    X1t_est_te = net.VAEdecS.predict([z1_m_te,s1t_te],batch_size=batch_size)
    
    MSEX1 = np.sum(np.square(X1-X1_est),axis=1)
    MSEX1_te = np.sum(np.square(X1_te - X1_est_te),axis=1)
    
    bdt = linear_model.LogisticRegression()
    bdt.fit(MSEX1.reshape(-1, 1), clss_train)
    preds_reg = bdt.predict_proba(MSEX1_te.reshape(-1, 1))#
    
    np.savetxt(dataroot + "/pred_reg" +savename+".csv", preds_reg,fmt='%10.5f', delimiter=",")
    np.savetxt(dataroot + "/mse_vals" +savename+".csv", MSEX1_te,fmt='%10.5f', delimiter=",")
elif clsfr_mode==4: #S
    bdt = linear_model.LogisticRegression()
    bdt.fit(s1_full, clss_train)
    preds_reg = bdt.predict_proba(s1_te)#
    
    np.savetxt(dataroot + "/pred_reg" +savename+".csv", preds_reg,fmt='%10.5f', delimiter=",")
elif clsfr_mode==5: #Z
    bdt = linear_model.LogisticRegression()
    bdt.fit(z1_m_full, clss_train)
    preds_reg = bdt.predict_proba(z1_m_te)#
    
    np.savetxt(dataroot + "/pred_reg" +savename+".csv", preds_reg,fmt='%10.5f', delimiter=",")
elif clsfr_mode==2:
    X1_est = net.VAEdecS.predict([z1_m_full,s1_full],batch_size=batch_size)
    X1_est_te = net.VAEdecS.predict([z1_m_te,s1_te],batch_size=batch_size)
    X1t_est = net.VAEdecS.predict([z1_m_full,s1t_full],batch_size=batch_size)
    X1t_est_te = net.VAEdecS.predict([z1_m_te,s1t_te],batch_size=batch_size)
    
    MSEX1X1t = np.square(X1_est-X1t_est)/np.square(X1_est)
    MSEX1X1t_te = np.square(X1_est_te-X1t_est_te)/X1_est
    
elif clsfr_mode==0:
    np.save(dataroot+'X', X1)
    np.save(dataroot+'Xid', id_train)
    np.save(dataroot+'clss_train', clss_train)
    np.save(dataroot+'ret_train', ret_train)
    np.save(dataroot+'Z_m', z1_m_full)
    np.save(dataroot+'Z_std', z1_std_full)
    #np.save(dataroot+'S', s1_full)
    
    np.save(dataroot+'Xte', X1_te)
    np.save(dataroot+'Xid_te', id_test)
    np.save(dataroot+'clss_te', clss_test)
    np.save(dataroot+'ret_te', ret_test)
    np.save(dataroot+'Z_m_te', z1_m_te)
    np.save(dataroot+'Z_std_te', z1_std_te)
    #np.save(dataroot+'S_te', s1_te)
    
    print 'logistic train...'
    
    if load_clsfr:
        pred_ids = clsfrNet.Net.predict(z1_m_te,batch_size=params['batch_size'])
    else:
        if params_clsfr['reg'] :
            bdt = linear_model.LinearRegression()
            bdt.fit(z1_m_full, ret_train)
            preds_reg = bdt.predict(z1_m_te)#_proba
            
            print 'net train...'
            hist = clsfrNet.Net.fit(z1_m_full, ret_train,shuffle=True,nb_epoch=nb_epoch,batch_size=batch_size,
                                    verbose=2,validation_data=(z1_m_te, ret_test),callbacks=[clsfrNet.checkpointer])
        
            #clsfrNet.Net.load_weights(weihts_filename)
            #pred_ids = clsfrNet.Net.predict(z1_m_te,batch_size=params['batch_size'])
            
        else:
            bdt = linear_model.LogisticRegression()
            bdt.fit(z1_m_full, clss_train)
            preds_reg = bdt.predict_proba(z1_m_te)#
            
            clf = SVC(probability=True)
            clf.fit(z1_m_full, clss_train) 
            preds_svm = clf.predict_proba(z1_m_te)
            np.savetxt(dataroot + "/preds_svm"+savename+".csv", preds_svm,fmt='%10.5f', delimiter=",")
            
#            clf1 = ensemble.AdaBoostClassifier()#tree.DecisionTreeClassifier(max_depth=1),algorithm='SAMME',n_estimators=1000)
#            clf2 = ensemble.AdaBoostClassifier(tree.DecisionTreeClassifier(max_depth=1),algorithm='SAMME',n_estimators=1000)
#            clf3 = tree.DecisionTreeClassifier(max_depth=1)
#            clf1.fit(z1_m_full, clss_train)
#            clf2.fit(z1_m_full, clss_train)
#            clf3.fit(z1_m_full, clss_train)
#            preds_ada = clf2.predict_proba(z1_m_te)
#            preds_adanotree = clf1.predict_proba(z1_m_te)
#            preds_tree = clf3.predict_proba(z1_m_te)
#            np.savetxt(dataroot + "/preds_ada"+savename+".csv", preds_ada,fmt='%10.5f', delimiter=",")
#            np.savetxt(dataroot + "/preds_adanotree"+savename+".csv", preds_adanotree,fmt='%10.5f', delimiter=",")
#            np.savetxt(dataroot + "/pred_tree"+savename+".csv", preds_tree,fmt='%10.5f', delimiter=",")
            print 'net train...'
            hist = clsfrNet.Net.fit(z1_m_full, np_utils.to_categorical(clss_train,params_clsfr['num_clss']),shuffle=True,nb_epoch=nb_epoch,batch_size=batch_size,
                                    verbose=2,validation_data=(z1_m_te, np_utils.to_categorical(clss_test,params_clsfr['num_clss'])),callbacks=[clsfrNet.checkpointer])
            
        ##clsfrNet.Net.load_weights(weihts_filename)
        pred_ids = clsfrNet.Net.predict(z1_m_te,batch_size=params['batch_size'])    
        #    #pred_ids=0
        np.savetxt(dataroot + "/pred_reg"+savename+".csv", preds_reg,fmt='%10.5f', delimiter=",")
        
    #np.savetxt(dataroot + "/pred_val"+savename+".csv", pred_ids,fmt='%10.5f', delimiter=",")
    
np.savetxt(dataroot + "/tar_val"+savename+".csv", clss_test,fmt='%d', delimiter=",")
np.savetxt(dataroot + "/tar_ret"+savename+".csv", ret_test,fmt='%10.5f', delimiter=",")
np.savetxt(dataroot + "/id_test"+savename+".csv", id_test,fmt='%d', delimiter=",")
