import math
import numpy as np
from genephys.sampler import DataSampler
from genephys.decoders import Decoder

def test_genephys(self):
    N,Q,T,nchan = 250,2,400,32
    F_ENTRAINMENT = np.linspace(0.15,0.15*math.pi,nchan)

    spont_options = {"FREQ_RANGE": [0.01, math.pi/4], "POW_RANGE": [0.8, 1.2]}

    evoked_options = {"phase_reset": True, "power_modulation": False,
        "additive_response": True, "additive_oscillation": False,
        "CHAN_PROB": 1/2, 
        "DIFF_PH": math.pi/2,
        "STD_PH": math.pi/4,
        "DIFF_ADDR": 0.1,
        "STD_ADDR": 0.5,
        "F_ENTRAINMENT": F_ENTRAINMENT,
        "KERNEL_PAR": (25,(2,150,0)),
        "KERNEL_PAR_ADDR_0": (75,(2,250,0))}   

    ds = DataSampler(T,nchan,Q,spont_options,evoked_options)
    (X,Phase,Freq,Amplitude,Additive_response,Stim) = ds.sample(N)

    decoder = Decoder(classification=True,get_TGM=True)
    accuracy2, betas = decoder.decode(X,Stim)  

