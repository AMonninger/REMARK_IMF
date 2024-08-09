from ConsIndShockModel_HANK import IndShockConsumerType 
from TwoSectorModel_HANK import TwoSectorMarkovConsumerType
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy


### Define shared parameters
Rfree_f = 0.03
Rfree_i = 0.03
PermShkStd_f = 0.06
PermShkStd_i = 0.06
TranShkStd_f = 0.2
TranShkStd_i = 0.2
PermGroFac_f = 1.0
PermGroFac_i = 1.0
UnempPrb_f = 0.05
UnempPrb_i = 0.05
IncUnemp_f = 0.0
IncUnemp_i = 0.0
taxrate_f = 0.0
taxrate_i = 0.0
labor_f = 1.0 #0.8
labor_i = 1.0 #0.8 #0.6
wage_f = 1.0
wage_i = 1.0
BoroCnstArt_f = 0.0
BoroCnstArt_i = 0.0
LivPrb_f = .99375
LivPrb_i = .99375

cycles = 0
T_cycle = 1

# Define the Markov transition matrix for sector f(ormal) to i(nformal)
p_f_to_i = 0.2
p_i_to_f = 0.2
p_f_to_f = 1 - p_f_to_i
p_i_to_i = 1 - p_i_to_f

MrkvArray = np.array(
    [
        [
            p_f_to_f
        ,
            p_f_to_i
        ],
        [
           p_i_to_f
        ,
            p_i_to_i
        ]
    ]
)

NSectors = 2

Formal_Size = 0.5

### Dictionary to be passed to the consumer type
HANK_Dict_TwoSector = {
    "cycles": cycles,
    "T_cycle": T_cycle,
    # Parameters shared with the perfect foresight model
    "Rfree": [np.array([1.0 + Rfree_f, 1.0 + Rfree_i])],                    # Interest factor on assets
    "LivPrb" : [np.array([LivPrb_f, LivPrb_i])],                   # Survival probability

    # Parameters that specify the income distribution over the lifecycle
    "PermShkStd" : [np.array([UnempPrb_f, UnempPrb_i])],                  # Standard deviation of log permanent shocks to income
    "PermShkCount" : 5,                    # Number of points in discrete approximation to permanent income shocks
    "TranShkStd" : [np.array([TranShkStd_f, TranShkStd_i])],                  # Standard deviation of log transitory shocks to income
    "TranShkCount" : 5,    
    "PermGroFac": [np.array([PermGroFac_f, PermGroFac_i])],                  # Permanent income growth factor

    # HANK params
    "taxrate" : [np.array([taxrate_f, taxrate_i])], # set to 0.0 because we are going to assume that labor here is actually after tax income
    "labor": [np.array([labor_f, labor_i])],
    "wage": [np.array([wage_f, wage_i])],    
    
    # Number of points in discrete approximation to transitory income shocks
    "UnempPrb" : [np.array([UnempPrb_f, UnempPrb_i])],                      # Probability of unemployment while working
    "IncUnemp" :  [np.array([IncUnemp_f, IncUnemp_i])],                     # Unemployment benefits replacement rate
  
    # Parameters for constructing the "assets above minimum" grid
    "aXtraMax" : 500,                      # Maximum end-of-period "assets above minimum" value
    "aXtraCount" : 100,                     # Number of points in the base grid of "assets above minimum"
    "BoroCnstArt": [BoroCnstArt_f, BoroCnstArt_i],
    
    # Transition Matrix simulation parameters
    "mCount": 200,
    "mMax": 500,
    "mMin": 1e-5,
    "mFac": 3,

    # Hank model bool
    "HANK":True,     

    ### Markov Parameters
    "MrkvArray": [MrkvArray],  # Transition Matrix for Markov Process
    "global_markov": False,  # If True, then the Markov Process is the same for all agents
    "MrkvPrbsInit": [Formal_Size, 1 - Formal_Size],
}

# to add taxes, labor, and wage. This ensures that we can shock each variable.
def function(taxrate, labor, wage):
    
    z = (1- taxrate)*labor*wage
    return z

HANK_Dict_TwoSector['TranShkMean_Func'] = [function]

T = 300

Agent_TwoSector_Baseline = TwoSectorMarkovConsumerType(**HANK_Dict_TwoSector)
Agent_TwoSector_Baseline.solve()
A_SS_TwoSector_Baseline, C_SS_TwoSector_Baseline, A_SS_Mrkv_TwoSector_Baseline, C_SS_Mrkv_TwoSector_Baseline = Agent_TwoSector_Baseline.compute_steady_state()
MPC_TwoSector_Baseline = Agent_TwoSector_Baseline.calc_jacobian('labor',0,T)[0][0][0]






stop
####################################################################################################


# from ConsIndShockModel_HANK_NEW import IndShockConsumerType 
from TwoSectorModel_HANK import TwoSectorMarkovConsumerType
import sequence_jacobian as sj  # SSJ will allow us to define blocks, models, compute IRFs, etc
from sequence_jacobian.classes import JacobianDict, SteadyStateDict
from sequence_jacobian import het, simple, create_model              # functions
import matplotlib.pyplot as plt
import numpy as np
import time
from copy import copy, deepcopy

from scipy import optimize

# Define the Markov transition matrix for sector f(ormal) to i(nformal)
p_f_to_i = 0.0
p_i_to_f = 0.0
p_f_to_f = 1 - p_f_to_i
p_i_to_i = 1 - p_i_to_f

MrkvArray = np.array(
    [
        [
            p_f_to_f
        ,
            p_f_to_i
        ],
        [
           p_i_to_f
        ,
            p_i_to_i
        ]
    ]
)

NSectors = 2

## psize is the size of the formal sector.
psize = 0.5
pformal_share = psize/(1-psize)
wage_share = 1.0

UnempPrb_f = 0.05
IncUnemp_f = 0.7

# Defining steady state values of the economy
def fiscal_ss(B, r, G): 
    T = (1 + r) * B + G - B #+ (UnempPrb_f * IncUnemp_f) *psize    
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
Zi_ss = (Y_ss/(pformal_share * wage_share)) - T_ss # Both sectors can be taxed


C_ss = Y_ss - G_ss

print('Zf_ss: ' +str(Zf_ss))
# print('Zi_ss: ' +str(Zi_ss))
print('C_ss: ' +str(Y_ss - G_ss))

T = 300 # Dimention of TxT Jacobian matrix

T_cycles = 500
cycles = 1
LivPrb = .99375

PermShkStd_f = 0.06
PermShkStd_i = 0.06
TranShkStd_f = 0.2
TranShkStd_i = 0.2
PermGroFac_f = 1.0
PermGroFac_i = 1.0
UnempPrb_i = 0.05
IncUnemp_i = 0.7
taxrate_f = 0.0
taxrate_i = 0.0
# labor_f = 0.8
# labor_i = 0.8 #0.6
wage_f = 1.0
wage_i = 1.0
BoroCnstArt_f = 0.0
BoroCnstArt_i = 0.0
LivPrb_f = .99375
LivPrb_i = .99375
PermGroFac_f = 1.0
PermGroFac_i = 1.0

init_twosector_life = {
    "cycles" : cycles,
    "T_cycle" : T_cycles,
    # Parameters shared with the perfect foresight model
    "DiscFac": 0.9455718034241029,
    "Rfree": [np.array(2 * [1.03])] * T_cycles, #np.array(2 * [1.0 + r_ss]),                   # Interest factor on assets

    ### Two Sector Model Parameters
    "PermShkStd": [np.array([PermShkStd_f, PermShkStd_i])] * T_cycles,  # Standard deviation of log permanent shocks to income for each sector
    "TranShkStd": [np.array([TranShkStd_f, TranShkStd_i])] * T_cycles,  # Standard deviation of log transitory shocks to income for each sector
    "UnempPrb": [np.array([UnempPrb_f, UnempPrb_i])] * T_cycles,  # Probability of unemployment while working for each sector
    "IncUnemp": [np.array([IncUnemp_f, IncUnemp_i])] * T_cycles,  # Unemployment benefits replacement rate for each sector
    "taxrate": [np.array([taxrate_f, taxrate_i])] * T_cycles,  # Tax Rate for each sector
    "labor": [np.array([Zf_ss, Zi_ss])] * T_cycles,  # Labor for each sector
    "wage": [np.array([wage_f, wage_i])] * T_cycles,  # Wage for each sector
    "BoroCnstArt": [BoroCnstArt_f, BoroCnstArt_i],  # Borrowing constraint for the minimum allowable assets to end the period with  
    "LivPrb": [np.array([LivPrb_f, LivPrb_i])] * T_cycles,
    "PermGroFac": [np.array([PermGroFac_f, PermGroFac_i])] * T_cycles,
    # Parameters that specify the income distribution over the lifecycle
    "PermShkCount" : 5,                    # Number of points in discrete approximation to permanent income shocks
    "TranShkCount" : 5,    
    
    # Parameters for constructing the "assets above minimum" grid
    "aXtraMax" : 500,                      # Maximum end-of-period "assets above minimum" value
    "aXtraCount" : 100,                     # Number of points in the base grid of "assets above minimum"
    
    # Transition Matrix simulation parameters
    "mCount": 200,
    "mMax": 500,
    "mMin": 1e-5,
    "mFac": 3,

    # Hank model bool
    "HANK":True,     
    
     ### Markov Parameters
    "MrkvArray": [MrkvArray] * T_cycles,  # Transition Matrix for Markov Process
    "global_markov": False,  # If True, then the Markov Process is the same for all agents
    "MrkvPrbsInit": [psize, 1 - psize],

}


# to add taxes, labor, and wage. This ensures that we can shock each variable.
def function(taxrate, labor, wage):
    z = (1- taxrate)*labor*wage
    
    return z

init_twosector_life['TranShkMean_Func'] = [function]

cycles = 0
T_cycles = 1
init_twosector = {
    "cycles" : cycles,
    "T_cycle" : T_cycles,
    # Parameters shared with the perfect foresight model
    "DiscFac": 0.9455718034241029,
    "Rfree": [np.array(2 * [1.03])] * T_cycles, #np.array(2 * [1.0 + r_ss]),                   # Interest factor on assets

    ### Two Sector Model Parameters
    "PermShkStd": [np.array([PermShkStd_f, PermShkStd_i])] * T_cycles,  # Standard deviation of log permanent shocks to income for each sector
    "TranShkStd": [np.array([TranShkStd_f, TranShkStd_i])] * T_cycles,  # Standard deviation of log transitory shocks to income for each sector
    "UnempPrb": [np.array([UnempPrb_f, UnempPrb_i])] * T_cycles,  # Probability of unemployment while working for each sector
    "IncUnemp": [np.array([IncUnemp_f, IncUnemp_i])] * T_cycles,  # Unemployment benefits replacement rate for each sector
    "taxrate": [np.array([taxrate_f, taxrate_i])] * T_cycles,  # Tax Rate for each sector
    "labor": [np.array([Zf_ss, Zi_ss])] * T_cycles,  # Labor for each sector
    "wage": [np.array([wage_f, wage_i])] * T_cycles,  # Wage for each sector
    "BoroCnstArt": [BoroCnstArt_f, BoroCnstArt_i],  # Borrowing constraint for the minimum allowable assets to end the period with  
    "LivPrb": [np.array([LivPrb_f, LivPrb_i])] * T_cycles,
    "PermGroFac": [np.array([PermGroFac_f, PermGroFac_i])] * T_cycles,
    # Parameters that specify the income distribution over the lifecycle
    "PermShkCount" : 5,                    # Number of points in discrete approximation to permanent income shocks
    "TranShkCount" : 5,    
    
    # Parameters for constructing the "assets above minimum" grid
    "aXtraMax" : 500,                      # Maximum end-of-period "assets above minimum" value
    "aXtraCount" : 100,                     # Number of points in the base grid of "assets above minimum"
    
    # Transition Matrix simulation parameters
    "mCount": 200,
    "mMax": 500,
    "mMin": 1e-5,
    "mFac": 3,

    # Hank model bool
    "HANK":True,     
    
     ### Markov Parameters
    "MrkvArray": [MrkvArray] * T_cycles,  # Transition Matrix for Markov Process
    "global_markov": False,  # If True, then the Markov Process is the same for all agents
    "MrkvPrbsInit": [psize, 1 - psize],

}


# to add taxes, labor, and wage. This ensures that we can shock each variable.
def function(taxrate, labor, wage):
    z = (1- taxrate)*labor*wage
    
    return z

init_twosector['TranShkMean_Func'] = [function]

### Target Steady State Asset
def ss_func(beta):
    init_twosector_life['DiscFac'] = beta
    init_twosector['DiscFac'] = beta
    Agent_func_life = TwoSectorMarkovConsumerType(**init_twosector_life, verbose = False)
    Agent_func_life.solve()
    Agent_func = TwoSectorMarkovConsumerType(**init_twosector, verbose = False)
    Agent_func.solution_terminal = deepcopy(Agent_func_life.solution[0])
    A_ss = Agent_func.compute_steady_state()[0]
    return A_ss
    

def ss_dif(beta):  
    difference =   ss_func(beta) - Asset_target 
    return difference

Asset_target = B_ss
DiscFac = optimize.brentq(ss_dif,.75,.95)

init_twosector_life['DiscFac'] = DiscFac
init_twosector['DiscFac'] = DiscFac
Agent_GE_life = TwoSectorMarkovConsumerType(**init_twosector_life)
Agent_GE_life.solve()
Agent_GE = TwoSectorMarkovConsumerType(**init_twosector)
Agent_GE.solution_terminal = deepcopy(Agent_GE_life.solution[0])
Agent_GE.solve()

A_ss, C_ss, A_ss_Markv, C_ss_Markv = Agent_GE.compute_steady_state()
print(A_ss, C_ss)

start = time.time()

CJACR, AJACR = Agent_GE.calc_jacobian('Rfree',0,T)
CJACZ, AJACZ = Agent_GE.calc_jacobian('labor',0, T)

print('Seconds to calculate Jacobian', time.time() - start)

