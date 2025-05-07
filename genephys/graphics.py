import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as st
import seaborn as sb

from . import sampler

def plot_accuracy(accuracy,colorbar_accuracy_lim=(0.25,0.75),filename=None):
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
        g = sb.heatmap(ax=ax,data=accuracy[np.arange(T-1,-1,-1),:],\
            vmin=colorbar_accuracy_lim[0],vmax=colorbar_accuracy_lim[1],\
            cmap='bwr',xticklabels=False, yticklabels=False,square=True,cbar=True)
        ax.axhline(y=0, color='k',linewidth=4)
        ax.axhline(y=accuracy.shape[1], color='k',linewidth=4)
        ax.axvline(x=0, color='k',linewidth=4)
        ax.axvline(x=accuracy.shape[0], color='k',linewidth=4)

        x0, x1 = (0,accuracy.shape[1])
        y0, y1 = (0,accuracy.shape[1])
        #y0, y1 = g.ax_joint.get_ylim()
        # lims = [max(x0, y0), min(x1, y1)]
        g.plot([x0, x1],[y1, y0], '-k')

        # im = ax.imshow(accuracy[np.arange(T-1,-1,-1),:])
        #plt.colorbar(im, ax=ax)
        #im.set_clim(colorbar_accuracy_lim)
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

    if not filename is None:
        plt.savefig(filename)


def plot_betas(betas,filename=None):
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
    plt.yticks(labels_y, labels_y)
    ax.set_xlabel('Time')
    ax.set_ylabel('Channels')
    fig.tight_layout()

    if not filename is None:
        plt.savefig(filename)


def plot_corrcoefs(X,y,filename=None):

    def get_Y_from_Stimulus(Stimulus):
        """ Assumes that there is only one ocurrence per trial"""

        T,N = Stimulus.shape
        Y = np.zeros((N,))
        for j in range(N):
            y = np.max(Stimulus[:,j])
            if y>0: Y[j] = y
        return Y

    def check_y_format(y):
        if len(y.shape)==2: 
            y = get_Y_from_Stimulus(y)
        y = np.copy(y)
        y = y * 1 # convert to int if it was Boolean 
        y = y.astype(float)
        return y

    (T,N,p) = X.shape
    y = check_y_format(y)

    C = np.zeros((p,T))
    for t in range(T):
        for j in range(p):
            C[j,t] = np.corrcoef(X[t,:,j],y)[0,1]

    plot_betas(C,filename)
    

def plot_signal(X,Phase=None,Freq=None,Amplitude=None,Additive_responses=None,\
        Stimulus=None,n=0,j=0,filename=None):
    """ Plots the signal, its phase, frequency, amplitude, 
        the additive response, and/or the Stimulus, 
        for trial n and channel j """

    n_plots = 1 + (Phase is not None) + (Freq is not None) \
        + (Amplitude is not None) + (Stimulus is not None)
    if Additive_responses is not None:
        if type(Additive_responses) is tuple: n_plots += len(Additive_responses)
        else: n_plots += 1

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

    if Additive_responses is not None:
        if type(Additive_responses) is tuple:
            for ij in range(len(Additive_responses)):
                ax[i].plot(np.linspace(0,T,T),Additive_responses[ij][:,n,j],color='r',linewidth=2)
                ax[i].set_ylabel('additive response ' + str(ij))
                if i < n_plots-1: ax[i].tick_params(labelbottom = False, bottom = False)
                i += 1
        else:
            ax[i].plot(np.linspace(0,T,T),Additive_responses[:,n,j],color='r',linewidth=2)
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

    if not filename is None:
        plt.savefig(filename)


def plot_erp(Stimulus,X,Phase=None,Freq=None,Amplitude=None,Additive_responses=None,\
        j=0,filename=None):
    """ Plots the signal, its phase, frequency, amplitude, 
        and/or additive response, and/or the Stimulus, 
        for channel j """

    n_plots = 1 + (Phase is not None) + (Freq is not None) \
        + (Amplitude is not None)  
    if Additive_responses is not None:
        if type(Additive_responses) is tuple: n_plots += len(Additive_responses)
        else: n_plots += 1

    [T,N,_] = X.shape
    if Stimulus is not None:
        if (len(Stimulus.shape)==2) and (Stimulus.shape[1]==N):
            Y = np.zeros((N,),dtype=int)
            for i in range(N):
                y = np.max(Stimulus[:,i])
                if round(y) != y:
                    raise Exception("Stimulus has to be categorical")
                if y>0: Y[i] = int(y)
        else: 
            Y = Stimulus
        Q = np.max(Y)
        if round(Q) != Q: raise Exception("Stimulus has to be categorical")
        Q = int(Q)

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

    if Additive_responses is not None:
        if type(Additive_responses) is tuple:
            for ij in range(len(Additive_responses)):
                for k in range(1,Q+1):
                    ax[i].plot(np.linspace(0,T,T),np.mean(Additive_responses[ij][:,Y==k,j],axis=1),\
                        color=red_gradient[k-1,:],linewidth=2)
                ax[i].set_ylabel('additive response ' + str(ij))
                if i < n_plots-1: ax[i].tick_params(labelbottom = False, bottom = False)
                i += 1 
        else:
            for k in range(1,Q+1):
                ax[i].plot(np.linspace(0,T,T),np.mean(Additive_responses[:,Y==k,j],axis=1),\
                    color=red_gradient[k-1,:],linewidth=2)
            ax[i].set_ylabel('additive response')
            if i < n_plots-1: ax[i].tick_params(labelbottom = False, bottom = False)
            i += 1          
        
    ax[i-1].set_xlabel('Time')   
    for axj in ax: axj.label_outer()
    #fig.tight_layout()

    if not filename is None:
        plt.savefig(filename)


def plot_activation_function(kernel_type=('Exponential','Log'),kernel_par=(25,(10,150,50)),
        T=400,t=100,delay=0,jitter=0,filename=None):
    """ Plot the activation function, according to the kernel function.
        T is the length of the trial, and tstim indicates when the stimulus occurs
    """

    stim = np.zeros(T)
    stim[t] = 1
    ds = sampler.DataSampler(T)
    cstim,_ = ds.convolve_stimulus(stim,kernel_type,kernel_par,delay,jitter)
    fig,ax = plt.subplots()
    plt.plot(np.linspace(0,T,T),cstim,color='b',linewidth=2)
    ax.set_xlabel('Time')
    ax.set_ylabel('Activation')

    if not filename is None:
        plt.savefig(filename)


def plot_activation_functions(functions,T=400,t=100,delay=0,\
        jitter=0,overlap=True,filename=None):
    """ Plot the activation function, according to the kernel function.
        T is the length of the trial, and tstim indicates when the stimulus occurs
    """

    stim = np.zeros(T)
    stim[t] = 1
    ds = sampler.DataSampler(T)
    if overlap:
        cstim = np.zeros((T,len(functions)))
        for j in range(len(functions)):
                cstim[:,j],_ = ds.convolve_stimulus(stim,functions[j]["kernel_type"],functions[j]["kernel_par"], \
                    delay,jitter)
    else:
        cstim = np.zeros(T)
        for j in range(len(functions)):
            cs,_ = ds.convolve_stimulus(stim,functions[j]["kernel_type"],functions[j]["kernel_par"], \
                delay,jitter)
            cstim = np.maximum(cstim,cs)
    fig,ax = plt.subplots()
    plt.plot(np.linspace(0,T,T),cstim,color='b',linewidth=2)
    ax.set_xlabel('Time')
    ax.set_ylabel('Activation')

    if not filename is None:
        plt.savefig(filename)

