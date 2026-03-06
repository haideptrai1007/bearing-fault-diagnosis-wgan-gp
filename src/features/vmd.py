import numpy as np
import matplotlib.pyplot as plt
from vmdpy import VMD
from scipy.stats import kurtosis

# # VMD: The Conaductor's Settings
# K = 3          # Three instruments in our ensemble
# alpha = 1000   # Low bandwidth constraint for better separation
# tau = 0        # Crystal clear, no noise tolerance
# DC = 0         # No constant drone notes
# init = 2       # Fair start for all instruments
# tol = 1e-6     # Studio-grade precision

def vmd(signal, alpha=2000, tau=0, K=6, DC=0, init=1, tol=1e-6):
    u, u_hat, omega = VMD(signal, alpha=alpha, tau=tau, K=K, DC=DC, init=init, tol=tol)
    kurt = [kurtosis(u[k], fisher=False) for k in range(K)]
    idx = np.argsort(kurt)[-3:]
    signal = u[idx].sum(axis=0)
    return signal