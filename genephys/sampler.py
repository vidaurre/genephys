#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generates synthetic electrophysiological data, with spontaneously time-varying  
amplitude and frequency, and various options to introduce different types of 
stimulus-evoked effects.

See documentation in readthedocs

@author: Diego Vidaurre 2022
"""

import numpy as np
from numpy import matlib as mb
import math
from scipy import stats as st

class DataSampler():
    """
    Class to sample data
    """

    def __init__(self,T=400,nchan=10,Q=None,spont_options=None,evoked_options=None):

        if spont_options is None: spont_options = {}
        else: spont_options = spont_options.copy()
        if evoked_options is None: evoked_options = {}
        else: evoked_options = evoked_options.copy()        

        if not "FREQ_AR_W" in spont_options:
            spont_options["FREQ_AR_W"] = 0.95   
        elif spont_options["FREQ_AR_W"] > 1.0:
            raise Exception("FREQ_AR_W has to be equal or lower than 1.0") 
        if not "FREQ_RANGE" in spont_options:
            spont_options["FREQ_RANGE"] = (0.01, math.pi/4)
        if not "AMP_AR_W" in spont_options:
            spont_options["AMP_AR_W"] = 0.99
        elif spont_options["AMP_AR_W"] > 1.0:
            raise Exception("AMP_AR_W has to be equal or lower than 1.0") 
        if not "AMP_RANGE" in spont_options:
            spont_options["AMP_RANGE"] = (0.5, 2)
        if not "MEASUREMENT_NOISE" in spont_options:
            spont_options["MEASUREMENT_NOISE"] = 0.5

        if not "phase_reset" in evoked_options:
            evoked_options["phase_reset"] = True        
        if not "amplitude_modulation" in evoked_options:
            evoked_options["amplitude_modulation"] = False
        if not "additive_response" in evoked_options:
            evoked_options["additive_response"] = False
        if not "additive_oscillation" in evoked_options:
            evoked_options["additive_oscillation"] = False        

        if evoked_options["phase_reset"] and evoked_options["additive_oscillation"]:
            raise Exception("Phase resetting and additive oscillations are not compatible")

        task = (Q is not None) and \
            (evoked_options["phase_reset"] or 
            evoked_options["amplitude_modulation"] or  
            evoked_options["additive_response"] or
            evoked_options["additive_oscillation"])

        if not task:
            #print('Purely resting state')
            evoked_options["phase_reset"] = False
            evoked_options["amplitude_modulation"] = False
            evoked_options["additive_response"] = False
            evoked_options["additive_oscillation"] = False
        #else:
        #    print('Stimulation')

        if task:
            if not "CHAN_PROB" in evoked_options:
                evoked_options["CHAN_PROB"] = 0.5 * np.ones((nchan,))
            elif type(evoked_options["CHAN_PROB"])==float:
                evoked_options["CHAN_PROB"] = evoked_options["CHAN_PROB"] * np.ones((nchan,))
            if not "DELAY" in evoked_options:
                evoked_options["DELAY"] = 25 + 5 * np.random.random_sample((Q,nchan))
            elif evoked_options["DELAY"].shape == (Q,):
                evoked_options["DELAY"] = np.transpose(mb.repmat(evoked_options["DELAY"], nchan, 1))
            elif evoked_options["DELAY"].shape != (Q,nchan):
                raise Exception("DELAY has invalid dimensions")
            evoked_options["DELAY"][evoked_options["DELAY"]<0] = 0
            if not "DELAY_JITTER" in evoked_options:
                evoked_options["DELAY_JITTER"] = 2.5
            if type(evoked_options["DELAY_JITTER"]) is not np.ndarray:
                evoked_options["DELAY_JITTER"] = evoked_options["DELAY_JITTER"] * np.ones((Q,nchan))
            elif evoked_options["DELAY_JITTER"].shape == (Q,):
                evoked_options["DELAY_JITTER"] = np.transpose(mb.repmat(evoked_options["DELAY_JITTER"], nchan, 1))
            elif evoked_options["DELAY_JITTER"].shape != (Q,nchan):
                raise Exception("DELAY_JITTER has invalid dimensions")
            if not "DELAY_ABSOLUTE_JITTER" in evoked_options:
                evoked_options["DELAY_ABSOLUTE_JITTER"] = 0.0
            if not "KERNEL_TYPE" in evoked_options:
                evoked_options["KERNEL_TYPE"] = ('Exponential','Log')
            if not "KERNEL_PAR" in evoked_options:
                evoked_options["KERNEL_PAR"] = (round(T*0.2),(10,round(T*0.4),0))  
            if not "PRESTIM_STATE_BIAS" in evoked_options:
                evoked_options["PRESTIM_STATE_BIAS"] = False
            
        if evoked_options["phase_reset"]:
            if not "DIFF_PH" in evoked_options:
                evoked_options["DIFF_PH"] = math.pi
            if evoked_options["DIFF_PH"] > math.pi:
                raise Exception("The maximum value for DIFF_PH is pi")
            if not "STD_PH" in evoked_options:
                evoked_options["STD_PH"] = 0.1         
            if not "PH" in evoked_options:
                evoked_options["PH"] = np.linspace(-evoked_options["DIFF_PH"]/2,+evoked_options["DIFF_PH"]/2,Q) 
            if evoked_options["PH"].shape == (Q,):
                evoked_options["PH"] = np.transpose(mb.repmat(evoked_options["PH"], nchan, 1))
            elif evoked_options["PH"].shape != (Q,nchan):
                raise Exception("PH has invalid dimensions")
            if not "ENTRAINMENT_STRENGTH" in evoked_options:
                evoked_options["ENTRAINMENT_STRENGTH"] = 1
            if (not "F_ENTRAINMENT" in evoked_options) or (evoked_options["F_ENTRAINMENT"] is None):
                # sorting out frequencies
                relevant_channels = evoked_options["CHAN_PROB"] > 0
                n_relevant_channels = np.sum(relevant_channels)
                if not "FREQUENCIES_ENTRAINMENT" in evoked_options:
                    evoked_options["FREQUENCIES_ENTRAINMENT"] = 0
                if isinstance(evoked_options["FREQUENCIES_ENTRAINMENT"], tuple ): # prespecified frequencies
                    freqs = evoked_options["FREQUENCIES_ENTRAINMENT"]
                    r = np.random.randint(0,len(freqs),n_relevant_channels)
                    F_ENTRAINMENT = np.zeros((n_relevant_channels,))
                    for j in range(n_relevant_channels):
                        F_ENTRAINMENT[j] = freqs[r[j]]
                elif evoked_options["FREQUENCIES_ENTRAINMENT"] == 1: # one frequency
                    F_ENTRAINMENT = 0.5 * spont_options["FREQ_RANGE"][1] * np.ones((n_relevant_channels,))
                elif evoked_options["FREQUENCIES_ENTRAINMENT"] == 0: # equidistant array of frequencies, one per channel
                    F_ENTRAINMENT = np.linspace(0.25,0.75,n_relevant_channels) * spont_options["FREQ_RANGE"][1]
                else: # a number higher than 1 
                    nfreqs = int(evoked_options["FREQUENCIES_ENTRAINMENT"])
                    freqs = np.linspace(0.25,0.75,nfreqs) * spont_options["FREQ_RANGE"][1]
                    r = np.random.randint(0,nfreqs,n_relevant_channels)
                    F_ENTRAINMENT = np.zeros((n_relevant_channels,))
                    for j in range(n_relevant_channels):
                        F_ENTRAINMENT[j] = freqs[r[j]]                    
                # sinusoidal or irregular phase progression?
                if (not "ENTRAINMENT_REGIME" in evoked_options) or (evoked_options["ENTRAINMENT_REGIME"] is None): 
                    evoked_options["ENTRAINMENT_REGIME"] = 'linear'
                if evoked_options["ENTRAINMENT_REGIME"] == 'nonlinear':
                    if not "FREQUENCY_NONLINEARITY_RATE" in evoked_options:
                        evoked_options["FREQUENCY_NONLINEARITY_RATE"] = 0.05
                    F_ENTRAINMENT = mb.repmat(F_ENTRAINMENT,T,1)
                    for c in range(n_relevant_channels): # adding variability (-50% to +150%) 
                        for t in range(1,T):
                            F_ENTRAINMENT[t,c] = F_ENTRAINMENT[t-1,c] + \
                                evoked_options["FREQUENCY_NONLINEARITY_RATE"] * \
                                np.random.normal(0,1) * spont_options["FREQ_RANGE"][1]
                elif evoked_options["ENTRAINMENT_REGIME"] != 'linear':
                    raise Exception("ENTRAINMENT_REGIME has an incorrect value")
                # assemble F_ENTRAINMENT
                if evoked_options["ENTRAINMENT_REGIME"] == 'nonlinear':
                    evoked_options["F_ENTRAINMENT"] = np.zeros((T,nchan))
                    evoked_options["F_ENTRAINMENT"][:,relevant_channels] = F_ENTRAINMENT
                else:                    
                    evoked_options["F_ENTRAINMENT"] = np.zeros((nchan,))
                    evoked_options["F_ENTRAINMENT"][relevant_channels] = F_ENTRAINMENT
                    evoked_options["F_ENTRAINMENT"] = mb.repmat(evoked_options["F_ENTRAINMENT"],T,1)
            elif evoked_options["F_ENTRAINMENT"].shape == (nchan,):
                evoked_options["F_ENTRAINMENT"] = mb.repmat(evoked_options["F_ENTRAINMENT"],T,1)
            if evoked_options["F_ENTRAINMENT"].shape != (T,nchan,Q):
                if evoked_options["F_ENTRAINMENT"].shape == (T,nchan):
                    evoked_options["F_ENTRAINMENT"] = np.tile(evoked_options["F_ENTRAINMENT"],(1,Q))
                    evoked_options["F_ENTRAINMENT"] = np.reshape(evoked_options["F_ENTRAINMENT"],(T,nchan,Q),order='F')
                else:
                    raise Exception("F_ENTRAINMENT has invalid dimensions")
            evoked_options["OWN_KERNEL_PH"] =  \
                ("KERNEL_TYPE_PH" in evoked_options) or ("KERNEL_PAR_PH" in evoked_options)
            if not "KERNEL_TYPE_PH" in evoked_options:
                evoked_options["KERNEL_TYPE_PH"] = evoked_options["KERNEL_TYPE"]
            if not "KERNEL_PAR_PH" in evoked_options:
                evoked_options["KERNEL_PAR_PH"] = evoked_options["KERNEL_PAR"]
                

        if evoked_options["amplitude_modulation"]:
            #if not "DIFF_AMP" in evoked_options:
            #    evoked_options["DIFF_AMP"] = 2.0
            #if not "STD_AMP" in evoked_options:
            #    evoked_options["STD_AMP"] = 0.1         
            if not "AMP" in evoked_options:
                evoked_options["AMP"] = 2 * np.ones((Q,nchan))
                #evoked_options["AMP"] = np.linspace(-evoked_options["DIFF_AMP"]/2,+evoked_options["DIFF_AMP"]/2,Q) 
            elif isinstance(evoked_options["AMP"],float) or isinstance(evoked_options["AMP"],int):
                evoked_options["AMP"] = float(evoked_options["AMP"]) * np.ones((Q,nchan))
            elif evoked_options["AMP"].shape == (Q,):
                evoked_options["AMP"] = np.transpose(mb.repmat(evoked_options["AMP"], nchan, 1))
            elif evoked_options["AMP"].shape != (Q,nchan):
                raise Exception("AMP has invalid dimensions")
            evoked_options["OWN_KERNEL_AMP"] = \
                ("KERNEL_TYPE_AMP" in evoked_options) or ("KERNEL_PAR_AMP" in evoked_options)
            if not "KERNEL_TYPE_AMP" in evoked_options:
                evoked_options["KERNEL_TYPE_AMP"] = evoked_options["KERNEL_TYPE"]
            if not "KERNEL_PAR_AMP" in evoked_options:
                evoked_options["KERNEL_PAR_AMP"] = evoked_options["KERNEL_PAR"]
            
        if evoked_options["additive_response"]:
            if not "DIFF_ADDR" in evoked_options:
                evoked_options["DIFF_ADDR"] = 0.5
            if not "STD_ADDR" in evoked_options:
                evoked_options["STD_ADDR"] = 0.5         
            if not "ADDR" in evoked_options:
                evoked_options["ADDR"] = \
                    np.linspace(-evoked_options["DIFF_ADDR"]/2,+evoked_options["DIFF_ADDR"]/2,Q) 
            if evoked_options["ADDR"].shape == (Q,):
                evoked_options["ADDR"] = np.transpose(mb.repmat(evoked_options["ADDR"], nchan, 1))
            if evoked_options["ADDR"].shape == (Q,nchan):
                evoked_options["ADDR"] = np.expand_dims(evoked_options["ADDR"],axis=2)
            # elif evoked_options["ADDR"].shape != (Q,nchan): 
            #   raise Exception("ADDR has invalid dimensions")
            n_additive_responses = evoked_options["ADDR"].shape[2]
            if n_additive_responses == 1:
                evoked_options["OWN_KERNEL_ADDR_0"] = \
                    ("KERNEL_TYPE_ADDR" in evoked_options) or ("KERNEL_PAR_ADDR" in evoked_options) or \
                    ("KERNEL_TYPE_ADDR_0" in evoked_options) or ("KERNEL_PAR_ADDR_0" in evoked_options)
                if "KERNEL_TYPE_ADDR" in evoked_options:
                    evoked_options["KERNEL_TYPE_ADDR_0"] = evoked_options["KERNEL_TYPE_ADDR"]
                elif not ("KERNEL_TYPE_ADDR_0") in evoked_options:
                    evoked_options["KERNEL_TYPE_ADDR_0"] = evoked_options["KERNEL_TYPE"]
                if "KERNEL_PAR_ADDR" in evoked_options:
                    evoked_options["KERNEL_PAR_ADDR_0"] = evoked_options["KERNEL_PAR_ADDR"]
                elif not ("KERNEL_PAR_ADDR_0") in evoked_options:
                    evoked_options["KERNEL_PAR_ADDR_0"] = evoked_options["KERNEL_PAR"]   
            else:             
                for j in range(n_additive_responses):
                    evoked_options["OWN_KERNEL_ADDR_" + str(j)] = \
                        (("KERNEL_TYPE_ADDR_" + str(j)) in evoked_options) or \
                        (("KERNEL_PAR_ADDR_" + str(j)) in evoked_options)
                    if not ("KERNEL_TYPE_ADDR_" + str(j)) in evoked_options:
                        evoked_options["KERNEL_TYPE_ADDR_" + str(j)] = evoked_options["KERNEL_TYPE"]
                    if not ("KERNEL_PAR_ADDR_" + str(j)) in evoked_options:
                        evoked_options["KERNEL_PAR_ADDR_" + str(j)] = evoked_options["KERNEL_PAR"]

        if evoked_options["additive_oscillation"]:
            if not "STD_ADDOF" in evoked_options:
                evoked_options["STD_ADDOF"] = 0.01
            if not "ADDOF" in evoked_options:
                evoked_options["ADDOF"] = \
                    np.linspace(math.pi/10,math.pi/8,Q)
            if evoked_options["ADDOF"].shape == (Q,):
                evoked_options["ADDOF"] = np.transpose(mb.repmat(evoked_options["ADDOF"], nchan, 1))  
            if not "STD_ADDOP" in evoked_options:
                evoked_options["STD_ADDOP"] = 0.01
            if not "ADDOP" in evoked_options:
                evoked_options["ADDOP"] = np.linspace(-10,10,Q)            
            if evoked_options["ADDOP"].shape == (Q,):
                evoked_options["ADDOP"] = np.transpose(mb.repmat(evoked_options["ADDOP"], nchan, 1))  
            if not "STD_ADDOA" in evoked_options:
                evoked_options["STD_ADDOA"] = 0.01            
            if not "ADDOA" in evoked_options:
                evoked_options["ADDOA"] = 2 * np.ones((Q,nchan)) 
            if isinstance(evoked_options["ADDOA"],float) or isinstance(evoked_options["ADDOA"],int):
                evoked_options["ADDOA"] = evoked_options["ADDOA"] * np.ones((Q,nchan)) 
            if evoked_options["ADDOA"].shape == (Q,):
                evoked_options["ADDOA"] = np.transpose(mb.repmat(evoked_options["ADDOA"], nchan, 1))     
            evoked_options["OWN_KERNEL_ADDO"] = \
                ("KERNEL_TYPE_ADDO" in evoked_options) or ("KERNEL_PAR_ADDO" in evoked_options)     
            if not "KERNEL_TYPE_ADDO" in evoked_options:
                evoked_options["KERNEL_TYPE_ADDO"] = evoked_options["KERNEL_TYPE"]
            if not "KERNEL_PAR_ADDO" in evoked_options:
                evoked_options["KERNEL_PAR_ADDO"] = evoked_options["KERNEL_PAR"]

        self.T = T
        self.nchan = nchan
        self.task = task
        self.spont_options = spont_options
        self.evoked_options = evoked_options
        self.Q = Q


    @staticmethod
    def __find_phase_shift(cstim,wave):
        j = np.argmax(cstim)
        return math.pi/2 - wave[j]


    @staticmethod
    def polar_gradient(rho,rho_target):
        """" Find the angle difference using the shortest path """

        d = rho_target - rho
        if abs(d) <= math.pi: 
            delta = d 
        elif rho < 0: 
            rho_target = rho_target - 2 * math.pi
            delta = rho_target - rho
        else:
            rho_target = rho_target + 2 * math.pi
            delta = rho_target - rho
            
        return delta

    @staticmethod
    def convolve_stimulus(stim,kernel,kernel_par,delay=0,jitter=0,d=None):
        """" Convolves the stimulus spikes with a continuous Exponential or Log activation function """

        if kernel[0] == 'Exponential':
            n1 = st.norm.ppf(0.0001)
            x1 = np.linspace(-n1,0,kernel_par[0])
            y1 = st.norm.pdf(x1)
            y1 = y1 / np.max(y1)
        elif kernel[0] == 'Log':
            L = max(round(kernel_par[0][1] +  kernel_par[0][2] * np.random.normal()) , 10 )
            x1 = np.linspace(0,1,L)
            y1 = -np.log(np.power(np.abs(x1),kernel_par[0][0])+1)
            y1 = y1 - np.min(y1)
            y1 = y1 / np.max(y1)
            y1 = np.flip(y1)
        else:
            raise Exception("Invalid kernel")

        if kernel[1] == 'Exponential':
            n2 = st.norm.ppf(0.0001)
            x2 = np.linspace(-n2,0,kernel_par[1])
            y2 = st.norm.pdf(x2)
            y2 = y2 / np.max(y2)
            y2 = np.flip(y2)
        elif kernel[1] == 'Log':
            L = max(round(kernel_par[1][1] +  kernel_par[1][2] * np.random.normal()) , 10 )
            x2 = np.linspace(0,1,L)
            y2 = -np.log(np.power(np.abs(x2),kernel_par[1][0])+1)
            y2 = y2 - np.min(y2)
            y2 = y2 / np.max(y2)
        else:
            raise Exception("Invalid kernel")

        T = len(stim)
        y = np.concatenate((y1,y2))
        L = len(y)
        cstim = np.zeros(T)
        tstim = np.where(stim>0)[0]
        N = len(tstim) # N here is not trials but occurrences of the stim within a trial

        if d is None:
            d = delay + np.abs(jitter * np.random.normal(0,1,N))
            #d = delay + jitter * np.random.normal(0,1,N)
            if len(kernel_par)==3: d += kernel_par[2]
            d = np.round(d)
            #d[d<0] = 0
        
        for j in range(N): 
            t = tstim[j]
            yd = np.concatenate((np.zeros((np.int64(d[j]),)), y ))
            if (t+L+d[j]) > T:
                tmax = T - t
                cstim[t:] = yd[:tmax] + cstim[t:]
            else:
                tend = (t+L+d[j]).astype(int)
                cstim[t:tend] = yd + cstim[t:tend]
        
        cstim[cstim>1] = 1

        return cstim, d


    @staticmethod
    def sample_stimulus(N=200,Q=2,T=100,t=None):
        """ sample stimulus, one single presentation per trial
            N is the number of trials
            Q is the number of stimuli 
            T is the number of time points
         """

        if t is None: t = round(T*0.1)
        Y = np.random.multinomial(1, [1/Q]*Q, size=(N,))
        stim = np.zeros((T,N))
        for k in range(Q):
            stim[t,Y[:,k]==1] = k+1

        return stim


    def sample_freq_amplitude(self,N=200,initial_conditions=None):
        """ Sample spontaneous dynamics for N trials"""

        nchan,T = (self.nchan,self.T)

        if (initial_conditions is None) or (not "Freq" in initial_conditions):
            Freq = np.zeros((T,N,nchan))
            for j in range(N):
                f = np.random.normal(0, 1, size=(T+100,nchan))  
                for c in range(nchan):
                    for t in range(1,T+100):
                        f[t,c] = f[t-1,c] * self.spont_options["FREQ_AR_W"] + f[t,c]
                for c in range(nchan):
                    f[:,c] = f[:,c] - np.min(f[:,c])
                    f[:,c] = f[:,c] / np.max(f[:,c]) * \
                        (self.spont_options["FREQ_RANGE"][1] - self.spont_options["FREQ_RANGE"][0]) + \
                        self.spont_options["FREQ_RANGE"][0]
                Freq[:,j,:] = f[100:]
        else:
            Freq = np.copy(initial_conditions["Freq"])

        if (initial_conditions is None) or (not "Amplitude" in initial_conditions):
            Amplitude = np.zeros((T,N,nchan))
            for j in range(N):
                p = np.random.normal(0, 1, size=(T+100,nchan))  
                for c in range(nchan):
                    for t in range(1,T+100):
                        p[t,c] = p[t-1,c] * self.spont_options["AMP_AR_W"] + p[t,c]
                for c in range(nchan):
                    p[:,c] = p[:,c] - np.min(p[:,c])
                    p[:,c] = p[:,c] / np.max(p[:,c]) * \
                        (self.spont_options["AMP_RANGE"][1] - self.spont_options["AMP_RANGE"][0]) + \
                        self.spont_options["AMP_RANGE"][0]
                Amplitude[:,j,:] = p[100:]
        else:
            Amplitude = np.copy(initial_conditions["Amplitude"])

        return Freq, Amplitude


    def sample_active_channels(self,N=200,initial_conditions=None):
        """ Sample active channels"""

        nchan = self.nchan
        if (initial_conditions is None) or (not "active_channels" in initial_conditions):
            Active_Channels = np.random.random_sample((N,nchan)) <=  \
                np.expand_dims(self.evoked_options["CHAN_PROB"], axis=0)
        else:
            Active_Channels = np.copy(initial_conditions["Active_Channels"])

        return Active_Channels


    def sample_trial(self,f,a,active_channels=None,stim=None):
        """ Sample one trial worth of data"""

        nchan,T,Q,spont_options,evoked_options = \
            (self.nchan,self.T,self.Q,self.spont_options,self.evoked_options)

        x = np.zeros((T,nchan))
        ph = np.zeros((T,nchan)) # phase
        ar = np.zeros((T,nchan)) # additive response 
        ao = np.zeros((T,nchan)) # additive oscillation 

        phase_reset = evoked_options["phase_reset"]
        amplitude_modulation = evoked_options["amplitude_modulation"]
        additive_response = evoked_options["additive_response"]
        additive_oscillation = evoked_options["additive_oscillation"]
        if additive_response: n_additive_responses = evoked_options["ADDR"].shape[2]
        else: n_additive_responses = 1

        x[0,:] = 2 * math.pi * np.random.random_sample((1,nchan)) - math.pi
        if not self.task:
            for t in range(1,T):
                x[t,:] = x[t-1,:] + f[t,:]
                c = (x[t,:] >= math.pi)
                x[t,c] = x[t,c] - 2*math.pi
                c = (x[t,:] < -math.pi)
                x[t,c] = x[t,c] + 2*math.pi
            ph = np.copy(x)
            for c in range(nchan):
                x[:,c] = np.sin(ph[:,c]) * a[:,c] + \
                    self.spont_options["MEASUREMENT_NOISE"] * np.random.normal(0, 1, size=(T,))
            return x,ph,ar,ao,f,a   

        # Generating task evoked effects
        cstim = np.zeros((T,Q,nchan))
        cstim_ph = np.zeros((T,Q,nchan))
        cstim_pow = np.zeros((T,Q,nchan))  
        cstim_addr = np.zeros((T,Q,nchan,n_additive_responses)) 
        cstim_addo = np.zeros((T,Q,nchan))  
        
        speak = np.zeros((Q,nchan), dtype=int)
        d_abs = evoked_options["DELAY_ABSOLUTE_JITTER"] * np.random.normal(0, 1)
        for c in range(nchan):
            if not active_channels[c]:
                continue
            for k in range(Q):
                if not any(stim == k+1): continue
                cstim[:,k,c],d = self.convolve_stimulus(stim == k+1, \
                    self.evoked_options["KERNEL_TYPE"], \
                    self.evoked_options["KERNEL_PAR"], \
                    d_abs + self.evoked_options["DELAY"][k,c], \
                    self.evoked_options["DELAY_JITTER"][k,c])
                if phase_reset:
                    if self.evoked_options["OWN_KERNEL_PH"]:
                        cstim_ph[:,k,c],d = self.convolve_stimulus(stim == k+1, \
                            self.evoked_options["KERNEL_TYPE_PH"], \
                            self.evoked_options["KERNEL_PAR_PH"], \
                            d_abs + self.evoked_options["DELAY"][k,c], \
                            self.evoked_options["DELAY_JITTER"][k,c])
                    else:
                        cstim_ph[:,k,c] = cstim[:,k,c]
                    speak[k,c] = np.argmax(cstim_ph[:,k,c])
                    if cstim_ph[speak[k,c],k,c] == 0: speak[k,c] = 0
                if amplitude_modulation:
                    if self.evoked_options["OWN_KERNEL_AMP"]:
                        cstim_pow[:,k,c],d = self.convolve_stimulus(stim == k+1, \
                            self.evoked_options["KERNEL_TYPE_AMP"], \
                            self.evoked_options["KERNEL_PAR_AMP"], \
                            d_abs + self.evoked_options["DELAY"][k,c], \
                            self.evoked_options["DELAY_JITTER"][k,c])   
                    else:
                        cstim_pow[:,k,c] = cstim[:,k,c]     
                if additive_response:
                    for j in range(n_additive_responses):
                        if self.evoked_options["OWN_KERNEL_ADDR_" + str(j)]:
                            cstim_addr[:,k,c,j],d = self.convolve_stimulus(stim == k+1, \
                                self.evoked_options["KERNEL_TYPE_ADDR_" + str(j)], \
                                self.evoked_options["KERNEL_PAR_ADDR_" + str(j)], \
                                d_abs + self.evoked_options["DELAY"][k,c], \
                                self.evoked_options["DELAY_JITTER"][k,c])  
                        else:
                            cstim_addr[:,k,c,j] = cstim[:,k,c]
                    m = np.max(cstim_addr[:,k,c,:],axis=1)
                    for j in range(n_additive_responses):
                        cstim_addr[cstim_addr[:,k,c,j]<m,k,c,j] = 0
                if additive_oscillation:
                    if self.evoked_options["OWN_KERNEL_ADDO"]:
                        cstim_addo[:,k,c],d = self.convolve_stimulus(stim == k+1, \
                            self.evoked_options["KERNEL_TYPE_ADDO"], \
                            self.evoked_options["KERNEL_PAR_ADDO"], \
                            d_abs + self.evoked_options["DELAY"][k,c], \
                            self.evoked_options["DELAY_JITTER"][k,c])                              
                    else:
                        cstim_addo[:,k,c] = cstim[:,k,c]   

        # apply a multiplier on the response fuctions depending on the initial phase
        if evoked_options["PRESTIM_STATE_BIAS"]:
            state_mod = (x[0,:] + math.pi) / (2*math.pi) # between 0 and 1, highest at +pi, lowest at -pi
            for c in range(nchan): 
                cstim_ph[:,:,c] *= state_mod[c]
                cstim_pow[:,:,c] *= state_mod[c]
                cstim_addr[:,:,c] *= state_mod[c]
                cstim_addo[:,:,c] *= state_mod[c]

        # Phase effect: phase reset until peak time, then entraining 
        if phase_reset:
            PH = evoked_options["PH"]
            STD_PH = evoked_options["STD_PH"]
            F_ENTRAINMENT = evoked_options["F_ENTRAINMENT"]
            target_phase = np.zeros((nchan,Q))
            for c in range(nchan): 
                for k in range(Q):
                    target_phase[c,k] = np.random.vonmises(PH[k,c],1/(STD_PH**2))
            for t in range(1,T):
                x[t,:] = x[t-1,:]
                for c in range(nchan):
                    if active_channels[c]:
                        alpha_comp = max(1-np.sum(cstim_ph[t,:,c]),0) # if <0, saturation
                        x[t,c] += alpha_comp * f[t,c]
                        for k in range(Q):
                            # weight for stimulus response
                            alpha = evoked_options["ENTRAINMENT_STRENGTH"] * cstim_ph[t,k,c] 
                            if alpha == 0: continue
                            if t <= speak[k,c]: # reset to phase
                                delta = self.polar_gradient(x[t,c],target_phase[c,k]) 
                                x[t,c] += alpha * delta
                            else: # entrain
                                x[t,c] += alpha * F_ENTRAINMENT[t-speak[k,c],c,k]
                    else:
                        x[t,c] += f[t,c]
                    # restart cycle
                    if x[t,c] > math.pi: x[t,c] -= 2*math.pi
                    if x[t,c] < -math.pi: x[t,c] += 2*math.pi
        else:
            for t in range(1,T):
                x[t,:] = x[t-1,:] + f[t,:]
                c = (x[t,:] >= math.pi)
                x[t,c] -= 2*math.pi
                c = (x[t,:] < -math.pi)
                x[t,c] += 2*math.pi

        # transform phase into signal value
        ph = np.copy(x)
        x = np.sin(x)

        # Amplitude effect:
        if amplitude_modulation:
            AMP = evoked_options["AMP"]
            for t in range(T):
                for c in range(nchan):
                    if not active_channels[c]: continue
                    for k in range(Q):
                        alpha = cstim_pow[t,k,c] # weight for stimulus response
                        if not alpha: continue
                        rho = 1 + alpha * (AMP[k,c] - 1) # >1 multiplier
                        a[t,c] += rho * a[t,c]
        
        # scale signal by amplitude
        x *= a

        # additive response
        if additive_response:
            n_additive_responses = evoked_options["ADDR"].shape[2]
            STD_ADDR = evoked_options["STD_ADDR"]
            for j in range(n_additive_responses):
                ADDR = evoked_options["ADDR"][:,:,j] + STD_ADDR * np.random.normal(0,1,size=(Q,nchan))
                for t in range(T):
                    for c in range(nchan):
                        if not active_channels[c]: continue
                        for k in range(Q):
                            alpha = cstim_addr[t,k,c,j] # weight for stimulus response
                            if not alpha: continue
                            ar[t,c] += alpha * ADDR[k,c] #(alpha**0.25)
            x += ar

        # additive oscillation
        if additive_oscillation:
            STD_ADDOF = evoked_options["STD_ADDOF"]
            ADDOF = evoked_options["ADDOF"] + STD_ADDOF * np.random.normal(0,1,size=(Q,nchan))
            STD_ADDOP = evoked_options["STD_ADDOP"]
            ADDOP = evoked_options["ADDOP"] + STD_ADDOP * np.random.normal(0,1,size=(Q,nchan)) 
            STD_ADDOA = evoked_options["STD_ADDOA"]
            ADDOA = evoked_options["ADDOA"] + STD_ADDOA * np.random.normal(0,1,size=(Q,nchan)) 
            for c in range(nchan):
                if not active_channels[c]: continue
                for k in range(Q):
                    ph_kc = np.cumsum(ADDOF[k,c] * np.ones(T))
                    shift0 = self.__find_phase_shift(cstim_addo[:,k,c],ph_kc)
                    oscillation = np.sin(ADDOP[k,c] + shift0 + ph_kc)
                    #oscillation = np.sin(ADDOP[k,c] + ph_kc)
                    ao[:,c] += ADDOA[k,c] * oscillation * cstim_addo[:,k,c]
            x += ao

        # re-adjust frequency time series
        if phase_reset:
            for c in range(nchan):
                f[1:,c] = np.abs(ph[:-1,c] - ph[1:,c])
                tt = f[:,c] > math.pi
                f[tt,c] = 2 * math.pi - f[tt,c]

        # additive gaussian noise
        x = x + spont_options["MEASUREMENT_NOISE"] * np.random.normal(0, 1, x.shape)  

        # Normalise trial
        # x = st.zscore(x)

        return x,ph,ar,ao,f,a


    def sample(self,N=200,Stimulus=None,initial_conditions=None):
        """ Main sampling function """

        nchan,T,Q = (self.nchan,self.T,self.Q)

        T = self.T
        nchan = self.nchan
        Q = self.Q  
        task = self.task    

        Freq, Amplitude = self.sample_freq_amplitude(N,initial_conditions)
        if task: 
            Active_Channels = self.sample_active_channels(N,initial_conditions)
        if task and (Stimulus is None): 
            Stimulus = self.sample_stimulus(N,Q,T)

        Phase = np.zeros((T,N,nchan))
        X = np.zeros((T,N,nchan))
        Additive_response = np.zeros((T,N,nchan))
        Additive_oscillation = np.zeros((T,N,nchan))
        #if self.evoked_options["additive_response"]: Z = np.zeros((T,N,nchan))
            
        for j in range(N):
            if task:
                X[:,j,:],Phase[:,j,:],Additive_response[:,j,:],Additive_oscillation[:,j,:],Freq[:,j,:],Amplitude[:,j,:] = \
                    self.sample_trial(Freq[:,j,:],Amplitude[:,j,:],Active_Channels[j,:],Stimulus[:,j])
            else:
                X[:,j,:],Phase[:,j,:],Additive_response[:,j,:],Additive_oscillation[:,j,:],Freq[:,j,:],Amplitude[:,j,:] = \
                    self.sample_trial(Freq[:,j,:],Amplitude[:,j,:])

        return X,Phase,Freq,Amplitude,Additive_response,Additive_oscillation,Stimulus


    @staticmethod
    def project(data,head_model):
        """ Project to sensor space with a head model given by 
        a (original no. channels by number of sensors)"""

        T,N,nchan = data.shape
        nsensor = head_model.shape[1]
        data = np.reshape(data,(T*N,nchan))
        projected_data = np.reshape(data @ head_model,(T,N,nsensor))
        return projected_data

