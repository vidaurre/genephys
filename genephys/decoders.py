#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Decoding algorithms
@author: Diego Vidaurre 2022
"""

import numpy as np
from numpy import matlib as mb
from scipy import stats as st
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
#from sklearn.linear_model import LogisticRegression 
from sklearn.linear_model import Ridge

class Decoder():
    """
    Class to decode stimulus from data
    """

    def __init__(self,classification=True,get_TGM=True,
            binsize=1,binstep=1,cvscheme=None,ncv=10,alpha=0.01):
        self.classification = classification
        self.get_TGM = get_TGM
        self.binsize = binsize
        self.binstep = binstep
        self.cvscheme = cvscheme
        self.ncv = ncv
        self.alpha = alpha

    def check_y_format(self,y):
        if len(y.shape)==2: 
            y = self.get_Y_from_Stimulus(y)
        y = np.copy(y)
        y = y * 1 # convert to int if it was Boolean 
        if self.classification:
            y = y.astype(int)
            values = np.unique(y)
            ycopy = np.copy(y)
            for k in range(len(values)):
                y[ycopy == values[k]] = k+1            
        else:
            y = y.astype(float)

        return y


    @staticmethod
    def get_Y_from_Stimulus(Stimulus):
        """ Assumes that there is only one ocurrence per trial"""

        T,N = Stimulus.shape
        Y = np.zeros((N,))
        for j in range(N):
            y = np.max(Stimulus[:,j])
            if y>0: Y[j] = y

        return Y


    @staticmethod
    def vcorrcoef(X,y):
        X = X - np.reshape(np.mean(X,axis=0),(1,X.shape[1]))
        y = np.expand_dims(y,axis=1)
        y = y - np.mean(y)
        r_num = np.sum(X*y,axis=0)
        r_den = np.sqrt(np.sum(X**2,axis=0)*np.sum(y**2))
        r = r_num/r_den
        return r


    @staticmethod
    def vaccuracy(X,y):
        acc = np.tile(y,(X.shape[1],1)).T == X
        acc = np.mean(acc,axis=0)
        return acc
    

    def create_folds(self,y):
        """" Creates CV structure given y, which, if this is a classification task, 
        must have elements 1...n_classes
        """

        N = len(y)
        train_inds = np.ones((N,self.ncv),dtype=bool)
        test_inds = np.zeros((N,self.ncv),dtype=bool)
        used = np.zeros((N,),dtype=bool)
        if self.classification:
            M = np.max(y)
            for c in range(M):
                indices = np.where(y==(c+1))[0]
                np.random.shuffle(indices)
                NK = int((1/self.ncv)*len(indices))
                for k in range(self.ncv):                    
                    indices_k = indices[k*NK + np.arange(NK)]
                    test_inds[indices_k,k] = True
                    train_inds[indices_k,k] = False
                    used[indices_k] = True
            # assign the rest if N is not a multiple of ncv
            if any(~used):
                k = 0
                for c in range(M):
                    indices = np.where((y==(c+1)) & ~used)[0]
                    np.random.shuffle(indices)
                    for j in range(len(indices)):
                        test_inds[indices[j],k] = True
                        train_inds[indices[j],k] = False
                        k += 1
                        if k==self.ncv: k = 0
                    used[indices] = True
    
        else:
            n = int(N/self.ncv)
            for k in range(self.ncv):
                indices = np.where(~used)[0]
                np.random.shuffle(indices)
                indices = indices[:n]
                test_inds[indices,k] = True
                train_inds[indices,k] = False
                used[indices] = True   
            unassigned = np.where(~used)[0]
            for k in range(len(unassigned)):
                test_inds[unassigned[k],k] = True
                train_inds[unassigned[k],k] = False
                used[unassigned[k]] = True   

        if any(~used): raise Exception('Something went wrong')
        return train_inds,test_inds

    
    def train_model_within_bin(self, X,y, alpha=0.01):

        s = X.shape
        if len(s) == 3: 
            (T,N,p) = X.shape
        else: 
            (N,p) = X.shape; 
            T = 1

        if T > 1:
            X = np.reshape(X,(T*N,p))
            y = np.tile(y,(1,T))
            y = np.reshape(y,(T*N,))

        if self.classification:
            if len(np.unique(y)) > 2:
                raise Exception('Label has more than two categories')

            #model = LogisticRegression(penalty='l2', C=C) 
            if alpha==0:
                model = LDA(solver = 'lsqr')
            else:
                model = LDA(solver = 'lsqr', shrinkage=alpha)
            model.fit(X,y)
        else:
            model = Ridge(alpha=alpha)
            model.fit(X,y)

        return model



    def TGM_from_model(self,model,X):

        binning = (len(X.shape) == 4)
        if binning: # TGM prediction with bins
            (nbin,binsize,N,p) = X.shape
            yhat = np.reshape(model.predict( np.reshape(X,(nbin*binsize*N,p)) ),[nbin,binsize,N])
            yhat = np.mean(yhat,axis=1)
            # yhat = np.zeros((nbin,N))
            # for j in range(nbin):
            #     yhat_j = np.reshape(model.predict( np.reshape(X[j,:,:,:],(binsize*N,p)) ),[binsize,N])
            #     yhat[j,:] = np.mean(yhat_j,axis=0)
        else: # by time point
            (T,N,p) = X.shape
            yhat = np.reshape(model.predict( np.reshape(X,(T*N,p)) ),[T,N])

        return yhat


    def predict_from_model(self,model,X,t):

        binning = (len(X.shape) == 4)
        if binning: # TGM prediction with bins
            (_,binsize,N,p) = X.shape
            yhat = np.reshape(model.predict( np.reshape(X[t,:,:,:],(binsize*N,p)) ),[binsize,N])
            yhat = np.mean(yhat,axis=0)
        else:  # by time point
            (T,N,p) = X.shape
            yhat = model.predict(X[t,:,:])

        return yhat
        

    def _decode(self,X,y,bins,training,testing):

        (T,_,p) = X.shape

        nbin,binsize = bins.shape
        if self.get_TGM:
            accuracy = np.zeros((nbin,nbin,self.ncv))
        else:
            accuracy = np.zeros((nbin,self.ncv))
        betas = np.zeros((p,nbin,self.ncv))
                    
        for k in range(self.ncv):
            for j in range(nbin):
                if binsize==1:
                    X_train = X[j,training[:,k],:]
                else:
                    X_train = X[bins[j,:],:,:]
                    X_train = X_train[:,training[:,k],:]
                y_train = y[training[:,k]]
                y_test = y[testing[:,k]]
                model = self.train_model_within_bin(X_train,y_train,self.alpha)
                betas[:,j,k] = model.coef_
                
                if binsize==1:
                    X_test = X[:,testing[:,k],:] # T x N x p
                else:
                    Nte = np.sum(testing[:,k])
                    X_test = np.zeros((nbin,binsize,Nte,p)) 
                    for i in range(nbin):
                        X_tmp = X[bins[i,:],:,:]
                        X_test[i,:,:,:] = X_tmp[:,testing[:,k],:]
                
                if self.get_TGM:
                    y_test_hat = self.TGM_from_model(model,X_test)
                    if self.classification:
                        accuracy[j,:,k] = self.vaccuracy(y_test_hat.T,y_test)
                    else:
                        accuracy[j,:,k] = self.vcorrcoef(y_test_hat.T,y_test)
                else:
                    y_test_hat = self.predict_from_model(model,X_test,j)
                    if self.classification:
                        accuracy[j,k] = np.mean(y_test_hat == y_test)
                    else:
                        accuracy[j,k] = np.corrcoef(y_test_hat, y_test)[0,1]

        return accuracy,betas


    def decode(self,X,y):

        (T,N,p) = X.shape
        y = self.check_y_format(y)
        
        #if self.classification and (len(np.unique(y)) != 2):
        #    raise Exception('For classification, y must have two different values only')
        if self.classification: 
            Q = len(np.unique(y))
            nproblems = int(Q * (Q-1) / 2)
        else: 
            nproblems = 1

        bins_start = np.arange(0,T-self.binsize+1,self.binstep)
        nbin = len(bins_start)
        bins = np.zeros((nbin,self.binsize),dtype=int)
        for j in range(self.binsize):
            bins[:,j] = bins_start + j

        r = 0
        if self.classification:
            if self.get_TGM:
                accuracy = np.zeros((nbin,nbin,self.ncv,nproblems))
            else:
                accuracy = np.zeros((nbin,self.ncv,nproblems))
            betas = np.zeros((p,nbin,self.ncv,nproblems))
            for j1 in range(1,Q):
                for j2 in range(j1+1,Q+1):
                    nn = (y==j1) | (y==j2)
                    if self.cvscheme is None: 
                        training,testing = self.create_folds(y[nn])
                    else: 
                        testing = self.cvscheme
                        training = np.logical_not(testing)
                        testing[~nn] = False
                        training[~nn] = False
                        
                    if self.get_TGM:
                        accuracy[:,:,:,r],betas[:,:,:,r] = self._decode(X[:,nn,:],y[nn],bins,training,testing)
                    else:
                        accuracy[:,:,r],betas[:,:,:,r] = self._decode(X[:,nn,:],y[nn],bins,training,testing)
                    r += 1
        else: 
            accuracy,betas = self._decode(self,X,y,bins,training,testing)
            
        if self.get_TGM:
            accuracy = np.mean(accuracy,axis=2)
        else:
            accuracy = np.mean(accuracy,axis=1)
        betas = np.mean(betas,axis=2)

        accuracy = np.squeeze(accuracy)
        betas = np.squeeze(betas)

        return accuracy, betas


    




        
