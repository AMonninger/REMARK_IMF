# from ConsIndShockModel_HANK_NEW import IndShockConsumerType 
from ConsIndShockModel_HANK import IndShockConsumerType 
import sequence_jacobian as sj  # SSJ will allow us to define blocks, models, compute IRFs, etc
from sequence_jacobian.classes import JacobianDict, SteadyStateDict
from sequence_jacobian import het, simple, create_model              # functions
import matplotlib.pyplot as plt
import numpy as np
import time

from scipy import optimize

## psize is the size of the formal sector.
psize = 1
# pformal_share = psize/(1-psize)
wage_share = 1.0

UnempPrb_f = 0.05
IncUnemp_f = 0.7

# Defining steady state values of the economy
def fiscal_ss(B, r, G): 
    T = (1 + r) * B + G - B + (UnempPrb_f * IncUnemp_f) *psize    
    return T

r_ss = 1.03 - 1
G_ss = .2
B_ss = 0.25 # this is lower than the tutorial by Straub et al. because need Higher MPC
Y_ss = 1.0

T_ss = fiscal_ss(B_ss,r_ss,G_ss)
print('T_ss: ' +str(T_ss))

# Z_ss = Y_ss - T_ss 
# Zf_ss = (Y_ss * psize * wage_share) - (T_ss * psize)
# Zi_ss = (Y_ss*(1-psize) /wage_share) - (T_ss * (1-psize)) # Both sectors can be taxed
Zf_ss = (Y_ss * wage_share) - T_ss 
# Zi_ss = (Y_ss/(pformal_share * wage_share)) - (T_ss / pformal_share) # Both sectors can be taxed


C_ss = Y_ss - G_ss

print('Zf_ss: ' +str(Zf_ss))
# print('Zi_ss: ' +str(Zi_ss))
print('C_ss: ' +str(Y_ss - G_ss))

T = 300 # Dimention of TxT Jacobian matrix

HANK_Dict_Formal = {
    # Parameters shared with the perfect foresight model
    "Rfree": 1.0 + r_ss,                    # Interest factor on assets
    "LivPrb" : [.99375],                   # Survival probability

    # Parameters that specify the income distribution over the lifecycle
    "PermShkStd" : [.06],                  # Standard deviation of log permanent shocks to income
    "PermShkCount" : 5,                    # Number of points in discrete approximation to permanent income shocks
    "TranShkStd" : [0.2],                  # Standard deviation of log transitory shocks to income
    "TranShkCount" : 5,    
    "PermGroFac": [1.0],                   # Permanent income growth factor
    
    # HANK params
    "taxrate" : [0.0], # set to 0.0 because we are going to assume that labor here is actually after tax income
    "labor": [Zf_ss],
    "wage": [1.0],    
    
    # Number of points in discrete approximation to transitory income shocks
    "UnempPrb" : UnempPrb_f,                      # Probability of unemployment while working
    "IncUnemp" :  IncUnemp_f,                     # Unemployment benefits replacement rate
  
    # Parameters for constructing the "assets above minimum" grid
    "aXtraMax" : 500,                      # Maximum end-of-period "assets above minimum" value
    "aXtraCount" : 100,                     # Number of points in the base grid of "assets above minimum"
    "BoroCnstArt": 0.0,
    
    # Transition Matrix simulation parameters
    "mCount": 200,
    "mMax": 500,
    "mMin": 1e-5,
    "mFac": 3,

    # Hank model bool
    "HANK":True,     
}

# to add taxes, labor, and wage. This ensures that we can shock each variable.
def function(taxrate, labor, wage):
    
    z = (1- taxrate)*labor*wage
    return z

HANK_Dict_Formal['TranShkMean_Func'] = [function]

### Target Steady State Asset
def ss_func(beta):
    HANK_Dict_Formal['DiscFac'] = beta
    AgentFormal_func = IndShockConsumerType(**HANK_Dict_Formal, verbose = False)
    A_ss = AgentFormal_func.compute_steady_state()[0]
    return A_ss
    

def ss_dif(beta):    
    return ss_func(beta) - Asset_target 

Asset_target = B_ss
DiscFac = optimize.brentq(ss_dif,.5,.99)

# Create a new agent
HANK_Dict_Formal['DiscFac'] = DiscFac
AgentFormal_GE = IndShockConsumerType(**HANK_Dict_Formal, verbose = False)

A_ss, C_ss = AgentFormal_GE.compute_steady_state()
print(A_ss, C_ss)

start = time.time()

CJACR, AJACR = AgentFormal_GE.calc_jacobian('Rfree',T)
CJACZ, AJACZ = AgentFormal_GE.calc_jacobian('labor', T)

print('Seconds to calculate Jacobian', time.time() - start)