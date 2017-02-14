
import numpy 
#import scipy.io
#import h5py
import pandas as pd
#import cPickle as pickle
#from keras.utils import np_utils
#import pdb
from joblib import Parallel, delayed
from tempfile import TemporaryFile

#from math import floor

class dataFactory_pyprep(object):
    
    def __init__(self, years_dict,data_path,data_tags,exchange_ids=None,s_mode=0,load_ready_data=False,valid_years=[],eleven_month=False,calc_beta=False):
        
      ##  dataroot = '/media/data2/naamahadad/PyData/smode1_full1984' 4 as test
        #dataroot = '/media/data2/naamahadad/PyData/smode1_full2' #5 as test, s per year
        #dataroot = '/media/data2/naamahadad/PyData/smode1_1985'
        #dataroot = '/media/data2/naamahadad/PyData/smode1_2010a'
        #dataroot = '/media/data2/naamahadad/PyData/smode1_sepyears'
        #dataroot = '/media/data2/naamahadad/PyData/smode1_2009'
        #dataroot = '/media/data2/naamahadad/PyData/smode1_2013'
        dataroot = '/media/data2/naamahadad/PyData/smode1_2012'
        #home = os.path.expanduser('~')
        #dataroot = home + '/FinData/Beast/smode1'

        if load_ready_data:
            self.train_data=pd.read_hdf(dataroot+'_train_data.hdf','table')
            self.test_data=pd.read_hdf(dataroot+'_test_data.hdf','table')
            self.train_lbls = numpy.load(dataroot+'_train_lbls.npy')
            self.test_lbls = numpy.load(dataroot+'_test_lbls.npy')
            self.data_tags = numpy.load(dataroot+'_data_tags.npy')
        else:
            self.data_tags = data_tags
        
        if (not load_ready_data):
            #full_data = pd.concat(pd.read_hdf(cur_path,'table') for cur_path in data_path)
            full_data = pd.read_hdf(data_path,'table')
                 
            #full_data.set_index('year',drop=True,inplace=True)
            full_data.sector = pd.to_numeric(full_data.sector, errors='coerce')
            print 'sector nan',numpy.sum(numpy.isnan(full_data.sector.values))
            full_data.sector.fillna(0, inplace=True)
            full_data.sector = numpy.floor(full_data.sector/100)
            
            if s_mode==1:
                full_data.insert(full_data.shape[1],'groups_val',full_data['year'])
            elif s_mode==2:
                full_data.insert(full_data.shape[1],'groups_val',full_data['year']*100+full_data['sector'])
                
            if len(valid_years)==0:
                train_data = full_data[full_data.year<=years_dict['train_top']]
                test_data = full_data[full_data.year>=years_dict['test_bottom']]
                test_data = test_data[test_data.year<=years_dict['test_top']]
            else:
                train_data = full_data.loc[~full_data['year'].isin(valid_years)]
                test_data = full_data.loc[full_data['year'].isin(valid_years)]
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
            self.s_mode = s_mode
            if s_mode>0:
                self.train_data = self.build_s_data(train_data,s_mode=s_mode)
                self.test_data = self.build_s_data(test_data,s_mode=s_mode)
                aa = [col for col in self.train_data.columns if 'day' in col]
                self.data_tags = aa
                
                
                self.train_lbls = (self.train_data.groups_val_2.values.astype(numpy.float32)-195000)
                self.test_lbls = (self.test_data.groups_val_2.values.astype(numpy.float32)-195000)
            
            print 'done! saving to',dataroot
            self.train_data.to_hdf(dataroot+'_train_data.hdf','table')
            self.test_data.to_hdf(dataroot+'_test_data.hdf','table')
            numpy.save(dataroot+'_train_lbls', self.train_lbls)
            numpy.save(dataroot+'_test_lbls', self.test_lbls)
            numpy.save(dataroot+'_data_tags', self.data_tags)
     
        if eleven_month:
            tmp=[]
            #self.data_tags = [col for col in self.data_tags if (int(col[col.index('y')+1:len(col)])<231)]
            for col in self.data_tags:#ind in range(len(self.data_tags)):
                #col = self.data_tags[ind]
                #if '_' in col:
                #    if (int(col[col.index('y')+1:col.index('_')])>230):
                #        del self.data_tags[ind]
                if '_' in col:
                    if (int(col[col.index('y')+1:col.index('_')])<230):
                        tmp[len(tmp):] = [col]
                else:
                    if (int(col[col.index('y')+1:len(col)])<230):
                        tmp[len(tmp):] = [col]
                        #tmp = tmp.concatenate((tmp,col))
            self.data_tags = tmp
        
        self.add_beta(calc_beta=calc_beta)
    def add_beta(self,calc_beta = False):
        def beta_calc(df):
            data1 =df[self.data_tags].values.astype(numpy.float32)
            data1 = data1[:,:data1.shape[1]/3]
            data1[numpy.isnan(data1)]=0
            data1 = numpy.clip(data1,0.0007- 4*0.04,0.0007 + 4*0.04)
            avg_rets = numpy.expand_dims(numpy.average(data1,axis=0),axis=0)-numpy.average(data1)
            mkt_var = numpy.var(avg_rets)
            N = data1.shape[1]
            data1 = data1 - numpy.repeat(numpy.expand_dims(numpy.average(data1,axis=1),axis=1),data1.shape[1],axis=1)
            cov_s_m = numpy.sum(data1*numpy.repeat(avg_rets,data1.shape[0],axis=0),axis=1)
            beta = cov_s_m/((N-1)*mkt_var)
            #stocks_std = numpy.repeat(numpy.expand_dims(numpy.std(data1,axis=1),axis=1),data1.shape[1],axis=1)
            stocks_std1 = numpy.std(data1,axis=1) + 1e-10
            rho = cov_s_m/(N*numpy.sqrt(mkt_var)*stocks_std1)
            df.insert(df.shape[1],'beta',beta)
            df.insert(df.shape[1],'rho1',rho)
            
            data2 =df[self.data_tags].values.astype(numpy.float32)
            data2 = data2[:,(data2.shape[1]/3):(2*(data2.shape[1]/3))]
            data2[numpy.isnan(data2)]=0
            data2 = numpy.clip(data2,0.0007- 4*0.04,0.0007 + 4*0.04)
            data2 = data2 - numpy.repeat(numpy.expand_dims(numpy.average(data2,axis=1),axis=1),data2.shape[1],axis=1)
            cov_s_m = numpy.sum(data2*numpy.repeat(avg_rets,data2.shape[0],axis=0),axis=1)
            stocks_std2 = numpy.std(data2,axis=1) + 1e-10
            rho2 = cov_s_m/(N*numpy.sqrt(mkt_var)*stocks_std2)
            df.insert(df.shape[1],'rho2',rho2)

            cov1_2 = numpy.sum(data1*data2,axis=1)
            rho1_2 = cov1_2/(N*stocks_std1*stocks_std2)
            df.insert(df.shape[1],'rho1_2',rho1_2)
            
            return df
            
        def applyParallel(dfGrouped, func):
            retLst = Parallel(n_jobs=num_workers)(delayed(func)(ind,group) for ind,(name, group) in enumerate(dfGrouped))
            #retLst = func(ind,group) for ind,(name, group) in enumerate(dfGrouped)
            #retLst = extract_data_stock(ind,group) for ind,(name, group) in enumerate(dfGrouped)
            return pd.concat(retLst)
            
        num_workers = 1
        if calc_beta:
            self.train_data = self.train_data.groupby('groups_val').apply(beta_calc)
            self.test_data = self.test_data.groupby('groups_val').apply(beta_calc)
        else:
            self.train_data.insert(0,'beta',numpy.repeat(0,self.train_data.shape[0]))
            self.test_data.insert(0,'beta',numpy.repeat(0,self.test_data.shape[0]))
            self.train_data.insert(0,'rho1',numpy.repeat(0,self.train_data.shape[0]))
            self.test_data.insert(0,'rho1',numpy.repeat(0,self.test_data.shape[0]))
            self.train_data.insert(0,'rho2',numpy.repeat(0,self.train_data.shape[0]))
            self.test_data.insert(0,'rho2',numpy.repeat(0,self.test_data.shape[0]))
        return
        
    def build_s_data(self,data,s_mode=1):
        global ind
        ind=0
        def build_s_year(df):
		
            global ind
            print ind
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
            ind = ind+1
            return newDf1
            
        def applyParallel(dfGrouped, func):
            retLst = Parallel(n_jobs=num_workers)(delayed(func)(ind,group) for ind,(name, group) in enumerate(dfGrouped))
            #retLst = func(ind,group) for ind,(name, group) in enumerate(dfGrouped)
            #retLst = extract_data_stock(ind,group) for ind,(name, group) in enumerate(dfGrouped)
            return pd.concat(retLst)
            
        num_workers = 1
        
        #count_years = data['year'].value_counts()
        #newTrain = pd.DataFrame(numpy.sum(numpy.floor(count_years.values/2)),self.train_data.shape[1])
        #newTrain = applyParallel(self.train_data.groupby('year'),build_s_year)
        #if s_mode==1:
        #    newTrain = data.groupby('year').apply(build_s_year)
        #    newTrain.groups_val = newTrain['year']
        #elif s_mode==2:
        #    newTrain = data.groupby(['year','sector']).apply(build_s_year)
        #    newTrain.groups_val = newTrain['year']*100+newTrain['sector']
	#newTrain = applyParallel(data.groupby('groups_val'),build_s_year)
        newTrain = data.groupby('groups_val').apply(build_s_year)
            
        #years = newTrain['year'].unique()
        #newTrain.reset_index(drop=True,inplace=True)
        #newTrain = newTrain.groupby('groups_val')
        groups_val = newTrain['groups_val'].unique()
        #finalTrain = pd.DataFrame()
        for group in groups_val:
            thisYear = newTrain[newTrain.groups_val==group]
            thisYear.reset_index(drop=True,inplace=True)
            diffYears = newTrain[newTrain.groups_val!=group]#self.train_data[self.train_data.year!=year]
            thirdPerm = numpy.random.permutation(range(diffYears.shape[0]))
            thirdPerm = thirdPerm[:thisYear.shape[0]]
            
            secondDF1 = diffYears.iloc[thirdPerm,:diffYears.shape[1]/2]
            secondDF1.rename(columns=lambda x: x+'_2', inplace=True)
            secondDF1.reset_index(drop=True,inplace=True)
            newDf = pd.concat((thisYear,secondDF1),axis=1)
            finalTrain = newDf if group==groups_val[0] else pd.concat((finalTrain,newDf),axis=0)
        
        return finalTrain
    def get_train_data(self):
        #y = np_utils.to_categorical(self.train_data['class_label'].values -1)
        #X = self.train_data[self.data_tags].values.astype(numpy.float32)
        rho1_t = self.train_data.rho1.values.astype(numpy.float32)
        rho2_t = self.train_data.rho2.values.astype(numpy.float32)
        #rho1 = rho1_t/(numpy.abs(rho1_t)+numpy.abs(rho2_t))
        #rho2 = rho2_t/(numpy.abs(rho1_t)+numpy.abs(rho2_t))
        rho1 = numpy.exp(rho1_t)/(numpy.exp(rho1_t)+numpy.exp(rho2_t))
        rho2 = numpy.exp(rho2_t)/(numpy.exp(rho1_t)+numpy.exp(rho2_t))
        rho1[numpy.isnan(rho1)]=0
        rho2[numpy.isnan(rho2)]=0
        return self.train_data[self.data_tags].values.astype(numpy.float32),self.train_lbls,rho1,rho2
        
    def get_test_data(self):   
        rho1_t = self.test_data.rho1.values.astype(numpy.float32)
        rho2_t = self.test_data.rho2.values.astype(numpy.float32)
        rho1 = rho1_t/(numpy.abs(rho1_t)+numpy.abs(rho2_t))
        rho2 = rho2_t/(numpy.abs(rho1_t)+numpy.abs(rho2_t))
        rho1[numpy.isnan(rho1)]=0
        rho2[numpy.isnan(rho2)]=0
        return self.test_data[self.data_tags].values.astype(numpy.float32),self.test_lbls,rho1,rho2
    
    def get_ret_per_year(self):
        def calc_ret(df):
            data =df[self.data_tags].values.astype(numpy.float32)
            data = data[:,:data.shape[1]/3]
            data[numpy.isnan(data)]=0
            data = numpy.clip(data,0.0007- 4*0.04,0.0007 + 4*0.04)
            rets1 = numpy.prod(1+data,axis=1)-1
            return numpy.average(rets1)
            
        def applyParallel(dfGrouped, func):
            retLst = Parallel(n_jobs=num_workers)(delayed(func)(ind,group) for ind,(name, group) in enumerate(dfGrouped))
            #retLst = func(ind,group) for ind,(name, group) in enumerate(dfGrouped)
            #retLst = extract_data_stock(ind,group) for ind,(name, group) in enumerate(dfGrouped)
            return pd.concat(retLst)
            
        num_workers = 1
        rets = pd.concat((self.train_data,self.test_data),axis=0).groupby('groups_val').apply(calc_ret)
        return rets
    def get_data_by_time(self,minyear,maxyear):
        data1 = self.train_data.loc[self.train_data['year']>=minyear]
        data1 = data1.loc[data1['year']<=maxyear]
        data2 = self.test_data.loc[self.test_data['year']>=minyear]
        data2 = data2.loc[data2['year']<=maxyear]
        
        ids_tr = data1.groups_val.values.astype(numpy.float32)
        ids_te = data2.groups_val.values.astype(numpy.float32)
        ids = numpy.concatenate((ids_tr,ids_te),axis=0)
        data1 = data1[self.data_tags].values.astype(numpy.float32)
        data2 = data2[self.data_tags].values.astype(numpy.float32)
        data = numpy.concatenate((data1,data2),axis=0)
        
        return data,ids #data1,ids_tr,data2,ids_te#data,ids
    def get_samples_per_group(self,samples_num,ret_rho=False):
        def rand_per_group(df):
            inds = numpy.random.permutation(range(min(samples_num,df.shape[0])))
            newDf = df.iloc[inds,:]
            group_key = newDf.groups_val.values.astype(numpy.float32)
            yearInd = numpy.nonzero(lbls_mapper[:,0]==numpy.asscalar(group_key[0]))
            #print group_key[0],lbls_mapper[yearInd,1]
            newDf.insert(0,'lbl',numpy.repeat(lbls_mapper[yearInd,3],inds.shape))
            newDf.insert(0,'retyear',numpy.repeat(lbls_mapper[yearInd,2],inds.shape))
            return newDf
            
        def applyParallel(dfGrouped, func):
            retLst = Parallel(n_jobs=num_workers)(delayed(func)(ind,group) for ind,(name, group) in enumerate(dfGrouped))
            #retLst = func(ind,group) for ind,(name, group) in enumerate(dfGrouped)
            #retLst = extract_data_stock(ind,group) for ind,(name, group) in enumerate(dfGrouped)
            return pd.concat(retLst)
            
        num_workers = 1
        
        
        lbls_mapper=numpy.genfromtxt('/media/data2/naamahadad/CSVs/YearlyRet.csv', delimiter=',')
        #df_group_tr = pd.concat((self.train_data,self.test_data),axis=0).groupby('groups_val').apply(rand_per_group)
        df_group_tr = self.train_data.groupby('groups_val').apply(rand_per_group)
        #samples_num = 5242
        df_group_te = self.test_data.groupby('groups_val').apply(rand_per_group)
        ids_tr = df_group_tr.groups_val.values.astype(numpy.float32)
        ids_te = df_group_te.groups_val.values.astype(numpy.float32)
        lbls_tr = df_group_tr.lbl.values.astype(numpy.float32)
        lbls_te = df_group_te.lbl.values.astype(numpy.float32)
        ret_tr = df_group_tr.retyear.values.astype(numpy.float32)
        ret_te = df_group_te.retyear.values.astype(numpy.float32)
        beta_tr = df_group_tr.beta.values.astype(numpy.float32)
        beta_te = df_group_te.beta.values.astype(numpy.float32)
        if ret_rho:
            rho_tr = df_group_tr.rho1.values.astype(numpy.float32)
            rho_te = df_group_te.rho1.values.astype(numpy.float32)
            return df_group_tr[self.data_tags].values.astype(numpy.float32),ids_tr,lbls_tr,ret_tr,beta_tr,rho_tr,df_group_te[self.data_tags].values.astype(numpy.float32),ids_te,lbls_te,ret_te,beta_te,rho_te
        else:
            return df_group_tr[self.data_tags].values.astype(numpy.float32),ids_tr,lbls_tr,ret_tr,beta_tr,df_group_te[self.data_tags].values.astype(numpy.float32),ids_te,lbls_te,ret_te,beta_te
        
