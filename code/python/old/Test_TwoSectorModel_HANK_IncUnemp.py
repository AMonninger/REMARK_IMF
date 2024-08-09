"""
Testing TwoSectorModel_HANK with Income at Unemployment
"""

import numpy as np
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
PermGroFac_f = 1.01
PermGroFac_i = 1.01
UnempPrb_f = 0.1
UnempPrb_i = 0.1
IncUnemp_f = 0.2
IncUnemp_i = 0.2
taxrate_f = 0.0
taxrate_i = 0.0
labor_f = 0.8
labor_i = 0.8 #0.6
wage_f = 1.0
wage_i = 1.0
BoroCnstArt_f = 0.0
BoroCnstArt_i = 0.0
LivPrb_f = .99375
LivPrb_i = .99375
PermGroFac_f = 1.01
PermGroFac_i = 1.01

cycles = 0
T_cycle = 1

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

Formal_Size = 0.5

### Dictionary to be passed to the consumer type
HANK_Dict_TwoSector = {
    "cycles": cycles,
    "T_cycle": T_cycle,
    # Parameters shared with the perfect foresight model
    "Rfree": [np.array([1.0 + Rfree_f, 1.0 + Rfree_i])] * T_cycle,                    # Interest factor on assets
    "LivPrb" : [np.array([LivPrb_f, LivPrb_i])] * T_cycle,                   # Survival probability

    # Parameters that specify the income distribution over the lifecycle
    "PermShkStd" : [np.array([UnempPrb_f, UnempPrb_i])] * T_cycle,                  # Standard deviation of log permanent shocks to income
    "PermShkCount" : 7,                    # Number of points in discrete approximation to permanent income shocks
    "TranShkStd" : [np.array([TranShkStd_f, TranShkStd_i])] * T_cycle,                  # Standard deviation of log transitory shocks to income
    "TranShkCount" : 7,    
    "PermGroFac": [np.array([PermGroFac_f, PermGroFac_i])] * T_cycle,                  # Permanent income growth factor

    # HANK params
    "taxrate" : [np.array([taxrate_f, taxrate_i])] * T_cycle, # set to 0.0 because we are going to assume that labor here is actually after tax income
    "labor": [np.array([labor_f, labor_i])] * T_cycle,
    "wage": [np.array([wage_f, wage_i])] * T_cycle,    
    
    # Number of points in discrete approximation to transitory income shocks
    "UnempPrb" : [np.array([UnempPrb_f, UnempPrb_i])] * T_cycle,                      # Probability of unemployment while working
    "IncUnemp" :  [np.array([IncUnemp_f, IncUnemp_i])] * T_cycle,                     # Unemployment benefits replacement rate
  
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
    "MrkvArray": [MrkvArray]* T_cycle,  # Transition Matrix for Markov Process
    "global_markov": False,  # If True, then the Markov Process is the same for all agents
    "MrkvPrbsInit": [Formal_Size, 1 - Formal_Size],
}

# to add taxes, labor, and wage. This ensures that we can shock each variable.
def function(taxrate, labor, wage):
    
    z = (1- taxrate)*labor*wage
    return z

HANK_Dict_TwoSector['TranShkMean_Func'] = [function] * T_cycle


Agent_TwoSector_Baseline = TwoSectorMarkovConsumerType(**HANK_Dict_TwoSector)

# Agent_TwoSector_Baseline.update()
Agent_TwoSector_Baseline.solve()

Agent_TwoSector_Baseline.compute_steady_state()
Agent_TwoSector_Baseline.calc_jacobian("Rfree", 0, 3)

mNrmGrid = np.linspace(Agent_TwoSector_Baseline.mMin, 10, 1000)

vPfunc_0 = Agent_TwoSector_Baseline.solution[0].vPfunc[0](mNrmGrid) 
vPfunc_1 = Agent_TwoSector_Baseline.solution[1].vPfunc[0](mNrmGrid)

vPfunc_0 - vPfunc_1
print("done")
# A_SS_TwoSector_Baseline, C_SS_TwoSector_Baseline = Agent_TwoSector_Baseline.compute_steady_state()