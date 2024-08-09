"""
Checking the size of Steady State distributions for the Two Sector Model.

"""


# from HARK.ConsumptionSaving.ConsMarkovModel import MarkovConsumerType
# from HARK.ConsumptionSaving.ConsIndShockModel import init_idiosyncratic_shocks
# from HARK.distribution import DiscreteDistributionLabeled
import matplotlib.pyplot as plt
from ConsIndShockModel_HANK import HANKIncShkDstn

from TwoSectorModel_HANK import TwoSectorMarkovConsumerType
import numpy as np
from copy import copy, deepcopy

# Define the Markov transition matrix for sector f(ormal) to i(nformal)
p_f_to_i = 0.2
p_i_to_f = 0.4
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
psize = 0.5


Zf_ss = 0.8
Zi_ss = 0.8

T_cycles = 1
cycles = 0
LivPrb = .99375

PermShkStd_f = 0.06
PermShkStd_i = 0.06
TranShkStd_f = 0.2
TranShkStd_i = 0.2
PermGroFac_f = 1.0
PermGroFac_i = 1.0
UnempPrb_f = 0.05
IncUnemp_f = 0.0
UnempPrb_i = 0.05
IncUnemp_i = 0.0
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

# Create a TwoSectorMarkovConsumerType
TwoSectorExample = TwoSectorMarkovConsumerType(**init_twosector_life)
TwoSectorExample.solve()
TwoSectorExample.completed_cycles

# Calculate Steady State
TwoSectorExample.compute_steady_state()