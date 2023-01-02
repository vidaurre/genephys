import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as st
import sampler, decoders

@staticmethod
def plot_accuracy(accuracy,colorbar_accuracy_lim=(0.25,0.75)):
    """ Plots the TGM and the diagonal, 
        which represents time point by time point accuracy
    """

    T = accuracy.shape[0]
    if len(accuracy.shape)>2:
        raise Exception("accuracy has more than 2 dimensions, \
        probably because there were more than 2 conditions; \
        pass accuracy[:,:,j] instead")

    if len(accuracy.shape)>1:
        fig = plt.figure()
        gs = fig.add_gridspec(1,2)
        ax = fig.add_subplot(gs[0, 0])
        im = ax.imshow(accuracy[np.arange(T-1,-1,-1),:])
        plt.colorbar(im, ax=ax)
        im.set_clim(colorbar_accuracy_lim)
        ax.set_xlabel('Training time')
        ax.set_ylabel('Testing time')
        ax = fig.add_subplot(gs[0, 1])
        ax.plot(np.linspace(0,T,T),np.diag(accuracy),color='b',linewidth=2)
        ax.set_xlabel('Time')
        ax.set_ylabel('Accuracy')
        fig.tight_layout()

    else: 
        fig,ax = plt.subplots()
        plt.plot(np.linspace(0,T,T),accuracy,color='b',linewidth=2)
        ax.set_xlabel('Time')
        ax.set_ylabel('Accuracy')

    # if title is not None:
    #     ax.set_title(title)


@staticmethod
def plot_betas(betas):
    """ Plots betas
    """
    fig,ax = plt.subplots()
    p,T = betas.shape
    labels_x = np.linspace(0,1,5)
    pos_x = np.linspace(0,T,5)
    if p > 10: labels_y = np.linspace(0,p-1,5)
    else: labels_y = np.arange(p)
    im = plt.imshow(betas,aspect='auto')
    #plt.colorbar(im, ax=ax)
    #plt.sca(ax)
    plt.xticks(pos_x, labels_x)
    if p > 10: plt.yticks(labels_y, labels_y)
    else: plt.yticks(labels_y, labels_y)
    ax.set_xlabel('Time')
    ax.set_ylabel('Channels')

    fig.tight_layout()


@staticmethod
def plot_signal(X,Phase=None,Freq=None,Amplitude=None,Additive_response=None,Stimulus=None,n=0,j=0):
    """ Plots the signal, its phase, frequency, amplitude, 
        the additive response, and/or the Stimulus, 
        for trial n and channel j """

    n_plots = 1 + (Phase is not None) + (Freq is not None) \
        + (Amplitude is not None) + (Additive_response is not None) + (Stimulus is not None)
    T = X.shape[0]

    if n_plots == 1:
        fig,ax = plt.subplots()
        plt.plot(np.linspace(0,T,T),X[:,n,j],color='b',linewidth=2)
        ax.set_xlabel('Time')
        ax.set_ylabel('x')
        return

    fig = plt.figure()
    gs = fig.add_gridspec(n_plots, hspace=0)
    ax = gs.subplots(sharex=True)

    i = 0
    ax[i].plot(np.linspace(0,T,T),X[:,n,j],color='b',linewidth=2)
    ax[i].set_ylabel('x')
    if i < n_plots-1: ax[i].tick_params(labelbottom = False, bottom = False)
    i += 1

    if Phase is not None:
        ax[i].plot(np.linspace(0,T,T),Phase[:,n,j],color='r',linewidth=2)
        ax[i].set_ylabel('phase')
        if i < n_plots-1: ax[i].tick_params(labelbottom = False, bottom = False)
        i += 1            
    
    if Freq is not None:
        ax[i].plot(np.linspace(0,T,T),Freq[:,n,j],color='r',linewidth=2)
        ax[i].set_ylabel('frequency')
        if i < n_plots-1: ax[i].tick_params(labelbottom = False, bottom = False)
        i += 1  

    if Amplitude is not None:
        ax[i].plot(np.linspace(0,T,T),Amplitude[:,n,j],color='r',linewidth=2)
        ax[i].set_ylabel('amplitude')
        if i < n_plots-1: ax[i].tick_params(labelbottom = False, bottom = False)
        i += 1  

    if Additive_response is not None:
        ax[i].plot(np.linspace(0,T,T),Additive_response[:,n,j],color='r',linewidth=2)
        ax[i].set_ylabel('additive response')
        if i < n_plots-1: ax[i].tick_params(labelbottom = False, bottom = False)
        i += 1          
        
    if Stimulus is not None:
        ax[i].plot(np.linspace(0,T,T),Stimulus[:,n],color='k',linewidth=2)
        ax[i].set_ylabel('stimulus')
        i += 1       

    ax[i-1].set_xlabel('Time')   
    for axj in ax: axj.label_outer()
    #fig.tight_layout()


@staticmethod
def plot_erp(Stimulus,X,Phase=None,Freq=None,Amplitude=None,Additive_response=None,j=0):
    """ Plots the signal, its phase, frequency, amplitude, 
        and/or additive response, and/or the Stimulus, 
        for channel j """

    n_plots = 1 + (Phase is not None) + (Freq is not None) \
        + (Amplitude is not None) + (Additive_response is not None) 
    [T,N,_] = X.shape
    if Stimulus is not None:
        Y = np.zeros((N,),dtype=int)
        for i in range(N):
            y = np.max(Stimulus[:,i])
            if round(y) != y:
                raise Exception("Stimulus has to be categorical")
            if y>0: Y[i] = int(y)
        Q = np.max(Y)
    blue_gradient = np.zeros((Q,3))
    blue_gradient[:,2] = np.linspace(0.5,1,Q)
    red_gradient = np.zeros((Q,3))
    red_gradient[:,0] = np.linspace(0.5,1,Q)

    if n_plots == 1:
        fig,ax = plt.subplots()
        for k in range(1,Q+1):
            plt.plot(np.linspace(0,T,T),np.mean(X[:,Y==k,j],axis=1),\
                color=blue_gradient[k-1,:],linewidth=2)
        ax.set_xlabel('Time')
        ax.set_ylabel('x')
        return
    
    fig = plt.figure()
    gs = fig.add_gridspec(n_plots, hspace=0)
    ax = gs.subplots(sharex=True)

    i = 0
    for k in range(1,Q+1):
        ax[i].plot(np.linspace(0,T,T),np.mean(X[:,Y==k,j],axis=1),\
            color=blue_gradient[k-1,:],linewidth=2)
    ax[i].set_ylabel('x')
    if i < n_plots-1: ax[i].tick_params(labelbottom = False, bottom = False)
    i += 1

    if Phase is not None:
        for k in range(1,Q+1):
            ax[i].plot(np.linspace(0,T,T),np.mean(Phase[:,Y==k,j],axis=1),\
                color=red_gradient[k-1,:],linewidth=2)
        ax[i].set_ylabel('phase')
        if i < n_plots-1: ax[i].tick_params(labelbottom = False, bottom = False)
        i += 1            
    
    if Freq is not None:
        for k in range(1,Q+1):
            ax[i].plot(np.linspace(0,T,T),np.mean(Freq[:,Y==k,j],axis=1),\
                color=red_gradient[k-1,:],linewidth=2)
        ax[i].set_ylabel('frequency')
        if i < n_plots-1: ax[i].tick_params(labelbottom = False, bottom = False)
        i += 1  

    if Amplitude is not None:
        for k in range(1,Q+1):
            ax[i].plot(np.linspace(0,T,T),np.mean(Amplitude[:,Y==k,j],axis=1),\
                color=red_gradient[k-1,:],linewidth=2)
        ax[i].set_ylabel('amplitude')
        if i < n_plots-1: ax[i].tick_params(labelbottom = False, bottom = False)
        i += 1  

    if Additive_response is not None:
        for k in range(1,Q+1):
            ax[i].plot(np.linspace(0,T,T),np.mean(Additive_response[:,Y==k,j],axis=1),\
                color=red_gradient[k-1,:],linewidth=2)
        ax[i].set_ylabel('additive response')
        if i < n_plots-1: ax[i].tick_params(labelbottom = False, bottom = False)
        i += 1          
        
    ax[i-1].set_xlabel('Time')   
    for axj in ax: axj.label_outer()
    #fig.tight_layout()


@staticmethod
def plot_activation_function(kernel_type=('Exponential','Log'),kernel_par=(25,(10,150,50)),
        T=400,t=100,delay=0,jitter=0):
    """ Plot the activation function, according to the kernel function.
        T is the length of the trial, and tstim indicates when the stimulus occurs
    """

    stim = np.zeros(T)
    stim[t] = 1
    ds = sample_data.DataSampler(T)
    cstim,_ = ds.convolve_stimulus(stim,kernel_type,kernel_par,delay,jitter)
    fig,ax = plt.subplots()
    plt.plot(np.linspace(0,T,T),cstim,color='b',linewidth=2)
    ax.set_xlabel('Time')
    ax.set_ylabel('Activation')
    



    
