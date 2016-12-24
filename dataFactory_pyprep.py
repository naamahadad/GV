
import numpy 
import scipy.io
import h5py
import pandas as pd
import cPickle as pickle
from keras.utils import np_utils
import pdb
from joblib import Parallel, delayed

#from math import floor

class dataFactory_pyprep(object):
    
    def __init__(self, years_dict,data_path,data_tags,exchange_ids=None,s_mode=0):
               
        full_data = pd.concat(pd.read_hdf(cur_path,'table') for cur_path in data_path)
             
        #full_data.set_index('year',drop=True,inplace=True)
        train_data = full_data[full_data.year<=years_dict['train_top']]
        test_data = full_data[full_data.year>=years_dict['test_bottom']]
        test_data = test_data[test_data.year<=years_dict['test_top']]
        
#        if train_data.shape[0] != 0:
#            train_labels = [numpy.where(train_data['class_label'] == 1)[0],
#                            numpy.where(train_data['class_label'] == 2)[0]]          
#        
#            # level datas
#            num_samples = [len(train_labels[0]),len(train_labels[1])]
#            sorted_numsamples = numpy.argsort(num_samples)
#            new_labels = numpy.random.choice(train_labels[sorted_numsamples[1]],size=num_samples[sorted_numsamples[0]],replace=False)            
#            res_train_labels = numpy.concatenate((numpy.asarray(train_labels[sorted_numsamples[0]]),new_labels),axis=0)
#            
#            train_data = train_data.iloc[res_train_labels]
#            train_data.reset_index(inplace=True,drop=False)
#        
#        test_data.reset_index(inplace=True,drop=False)
            
        self.train_data = train_data
        self.test_data = test_data
        self.data_tags = data_tags
        self.s_mode = s_mode
        if s_mode>0:
            self.train_data = self.build_s_data(train_data,s_mode=s_mode)
            self.test_data = self.build_s_data(test_data,s_mode=s_mode)
            aa = [col for col in self.train_data.columns if 'day' in col]
            self.data_tags = aa
            
            
            self.train_lbls = (self.train_data.year_2.values.astype(numpy.float32)-1950)
            self.test_lbls = (self.test_data.year_2.values.astype(numpy.float32)-1950)
            
    def build_s_data(self,data,s_mode=1):
        
        def build_s_year(df):
            couples_perm = numpy.random.permutation(range(numpy.floor(df.shape[0]/2).astype(numpy.int)*2))
            couples_perm = numpy.reshape(couples_perm,(couples_perm.shape[0]/2,2))
            firstDF = df.iloc[couples_perm[:,0],:]
            secondDF = df.iloc[couples_perm[:,1],:]
            secondDF.rename(columns=lambda x: x+'_1', inplace=True)
            newDf = pd.concat((firstDF,secondDF),axis=1)

            firstDF = df.iloc[couples_perm[:,1],:]
            secondDF = df.iloc[couples_perm[:,0],:]
            secondDF.rename(columns=lambda x: x+'_1', inplace=True)
            newDf1 = pd.concat((newDf,pd.concat((firstDF,secondDF),axis=1)),axis=0)
            
            return newDf1
            
        def applyParallel(dfGrouped, func):
            retLst = Parallel(n_jobs=num_workers)(delayed(func)(ind,group) for ind,(name, group) in enumerate(dfGrouped))
            #retLst = func(ind,group) for ind,(name, group) in enumerate(dfGrouped)
            #retLst = extract_data_stock(ind,group) for ind,(name, group) in enumerate(dfGrouped)
            return pd.concat(retLst)
            
        num_workers = 1
        
        count_years = data['year'].value_counts()
        #newTrain = pd.DataFrame(numpy.sum(numpy.floor(count_years.values/2)),self.train_data.shape[1])
        #newTrain = applyParallel(self.train_data.groupby('year'),build_s_year)
        newTrain = data.groupby('year').apply(build_s_year)
        
        years = newTrain['year'].unique()
        #finalTrain = pd.DataFrame()
        for year in years:
            thisYear = newTrain[newTrain.year==year]
            thisYear.reset_index(drop=True,inplace=True)
            diffYears = newTrain[newTrain.year!=year]#self.train_data[self.train_data.year!=year]
            thirdPerm = numpy.random.permutation(range(diffYears.shape[0]))
            thirdPerm = thirdPerm[:thisYear.shape[0]]
            
            secondDF1 = diffYears.iloc[thirdPerm,:diffYears.shape[1]/2]
            secondDF1.rename(columns=lambda x: x+'_2', inplace=True)
            secondDF1.reset_index(drop=True,inplace=True)
            newDf = pd.concat((thisYear,secondDF1),axis=1)
            finalTrain = newDf if year==years[0] else pd.concat((finalTrain,newDf),axis=0)
        
        return finalTrain
    def get_train_data(self):
        #y = np_utils.to_categorical(self.train_data['class_label'].values -1)
        #X = self.train_data[self.data_tags].values.astype(numpy.float32)
    
        return self.train_data[self.data_tags].values.astype(numpy.float32),self.train_lbls

    def get_test_data(self):          
        return self.test_data,self.test_lbls

				
