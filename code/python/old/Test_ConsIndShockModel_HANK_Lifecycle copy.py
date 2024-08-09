from HARK.ConsumptionSaving.ConsMarkovModel import MarkovConsumerType
from HARK.ConsumptionSaving.ConsIndShockModel import init_idiosyncratic_shocks
from HARK.distribution import DiscreteDistributionLabeled
import matplotlib.pyplot as plt
from ConsIndShockModel_HANK import HANKIncShkDstn

from TwoSectorModel_HANK import TwoSectorMarkovConsumerType
from ConsIndShockModel_HANK import IndShockConsumerType
import numpy as np
from copy import copy, deepcopy
### TESTING LIFECYCLE MODEL

# # Define the Markov transition matrix for sector f(ormal) to i(nformal)
# p_f_to_i = 0.0 #0.2
# p_i_to_f = 0.0 #0.2
# p_f_to_f = 1 - p_f_to_i
# p_i_to_i = 1 - p_i_to_f

# MrkvArray = np.array(
#     [
#         [
#             p_f_to_f
#         ,
#             p_f_to_i
#         ],
#         [
#            p_i_to_f
#         ,
#             p_i_to_i
#         ]
#     ]
# )

# NSectors = 2

r_ss = 1.03 - 1
G_ss = .2
B_ss = 0.25 # this is lower than the tutorial by Straub et al. because need Higher MPC
Y_ss = 1.0

T_cycles = 1
cycles = 0
init_twosector = {
    "cycles" : cycles,
    "T_cycle" : T_cycles,
    # Parameters shared with the perfect foresight model
    "DiscFac": 0.97,
    "Rfree": 1.0 + r_ss,                    # Interest factor on assets
    "LivPrb" : [.99375] * T_cycles,                   # Survival probability

    # Parameters that specify the income distribution over the lifecycle
    "PermShkStd" : [.06] * T_cycles,                  # Standard deviation of log permanent shocks to income
    "PermShkCount" : 5,                    # Number of points in discrete approximation to permanent income shocks
    "TranShkStd" : [0.2] * T_cycles,                  # Standard deviation of log transitory shocks to income
    "TranShkCount" : 5,    
    
    # HANK params
    "taxrate" : [0.0] * T_cycles, # set to 0.0 because we are going to assume that labor here is actually after tax income
    "labor": [1.0] * T_cycles,
    "wage": [1.0] * T_cycles,    
    
    # Number of points in discrete approximation to transitory income shocks
    "UnempPrb" : 0.0,                      # Probability of unemployment while working
    "IncUnemp" :  0.0,                     # Unemployment benefits replacement rate
  
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
    
    #  ### Markov Parameters
    # "MrkvArray": [[0.9, 0.1], [0.1, 0.9]],  # Transition Matrix for Markov Process
    # "global_markov": False,  # If True, then the Markov Process is the same for all agents
    # "MrkvPrbsInit": [0.8, 0.2],

    
}

# to add taxes, labor, and wage. This ensures that we can shock each variable.
def function(taxrate, labor, wage):
    
    z = (1- taxrate)*labor*wage
    return z

init_twosector['TranShkMean_Func'] = [function]

# Make a consumer with serially correlated unemployment, subject to boom and bust cycles
# init_twosector = copy(init_idiosyncratic_shocks)
# init_twosector["MrkvArray"] = [MrkvArray]
# init_twosector["UnempPrb"] = 0.0  # to make income distribution when employed
# init_twosector["global_markov"] = False

### Use the HARK one and add a different wage rate and tax process
def function(taxrate, labor, wage):
    
    z = (1- taxrate)*labor*wage
    return z

# sigma_Perm = 0.6
# sigma_Tran = 0.2
# n_approx_Perm = 7
# n_approx_Tran = 7
# IncUnemp = 0.0
# TranShkMean_Func = function
# labor = 1.0

# # Sector specific ones
# wage_f = 1.0
# taxrate_f = 0.0 #0.1
# UnempPrb_f = 0.05

# formal_income_dist_HANK = HANKIncShkDstn(sigma_Perm, sigma_Tran, n_approx_Perm, n_approx_Tran, UnempPrb_f, IncUnemp, taxrate_f, TranShkMean_Func, labor, wage_f)

# wage_i = 1.0 #0.5
# taxrate_i = 0.0
# UnempPrb_i = 0.05 #0.1
# informal_income_dist_HANK = HANKIncShkDstn(sigma_Perm, sigma_Tran, n_approx_Perm, n_approx_Tran, UnempPrb_i, IncUnemp, taxrate_i, TranShkMean_Func, labor, wage_i)


### Try new Type
TwoSectorExampleHANK = IndShockConsumerType(**init_twosector)
TwoSectorExampleHANK.compute_steady_state()
TwoSectorExampleHANK.calc_jacobian("Rfree", 3)
# TwoSectorExampleHANK.update_income_process()
TwoSectorExampleHANK.solve()


stop 

TwoSectorExampleHANK.assign_parameters(
    Rfree=np.array(NSectors * [TwoSectorExampleHANK.Rfree])
)
TwoSectorExampleHANK.PermGroFac = [
    np.array(NSectors * TwoSectorExampleHANK.PermGroFac)
]

# TwoSectorExampleHANK = MarkovConsumerType(**init_twosector)
# TwoSectorExampleHANK.IncShkDstn = [
#     [
#         formal_income_dist_HANK,
#         informal_income_dist_HANK,
#     ]
# ]


TwoSectorExampleHANK.LivPrb = [TwoSectorExampleHANK.LivPrb * np.ones(NSectors)]
TwoSectorExampleHANK.BoroCnstArt = np.array([0.0, 0.0])

TwoSectorExampleHANK.solve()
# TwoSectorExampleHANK.compute_steady_state()
# TwoSectorExampleHANK.calc_jacobian("Rfree", 3)
# 285.30983078304945
# 10.120636925963431
stop


# TwoSectorExampleHANK = MarkovConsumerType(**init_twosector)
TwoSectorExampleHANK.IncShkDstn = [
    [
        formal_income_dist_HANK,
        informal_income_dist_HANK,
    ]
]

# Time varying shape
TwoSectorExampleHANK.assign_parameters(
    Rfree=np.array(NSectors * [TwoSectorExampleHANK.Rfree])
)

TwoSectorExampleHANK.PermGroFac = [
    np.array(NSectors * TwoSectorExampleHANK.PermGroFac)
]
TwoSectorExampleHANK.LivPrb = [TwoSectorExampleHANK.LivPrb * np.ones(NSectors)]

TwoSectorExampleHANK.BoroCnstArt = np.array([0.0, 0.0])

TwoSectorExampleHANK.cycles = 0
TwoSectorExampleHANK.vFuncBool = False 

TwoSectorExampleHANK.MrkvPrbsInit = [0.5, 0.5]

TwoSectorExampleHANK.neutral_measure = True

TwoSectorExampleHANK.compute_steady_state()