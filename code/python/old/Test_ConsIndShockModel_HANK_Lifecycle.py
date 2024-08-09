from HARK.ConsumptionSaving.ConsMarkovModel import MarkovConsumerType
from HARK.ConsumptionSaving.ConsIndShockModel import init_idiosyncratic_shocks
from HARK.distribution import DiscreteDistributionLabeled
import matplotlib.pyplot as plt
from ConsIndShockModel_HANK import HANKIncShkDstn

from TwoSectorModel_HANK import TwoSectorMarkovConsumerType
import numpy as np
from copy import copy, deepcopy
### TESTING LIFECYCLE MODEL

# Define the Markov transition matrix for sector f(ormal) to i(nformal)
p_f_to_i = 0.0 #0.2
p_i_to_f = 0.0 #0.2
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

r_ss = 1.03 - 1
G_ss = .2
B_ss = 0.25 # this is lower than the tutorial by Straub et al. because need Higher MPC
Y_ss = 1.0

T_cycles = 1
cycles = 0
LivPrb = 0.99375
init_twosector = {
    "cycles" : cycles,
    "T_cycle" : T_cycles,
    # Parameters shared with the perfect foresight model
    "DiscFac": 0.97,
    "Rfree": np.array(2 * [1.0 + r_ss]), #[np.array(2 * [1.03])] * T_cycles                  # Interest factor on assets
    # "LivPrb" : [np.array([0.06, 0.06])] * T_cycles,                   # Survival probability

    ### Two Sector Model Parameters
    "PermShkStd": [np.array([0.06, 0.06])] * T_cycles,  # Standard deviation of log permanent shocks to income for each sector
    "TranShkStd": [np.array([0.2, 0.2])] * T_cycles,  # Standard deviation of log transitory shocks to income for each sector
    "UnempPrb": [np.array([0.0, 0.0])] * T_cycles,  # Probability of unemployment while working for each sector
    "IncUnemp": [np.array([0.0, 0.0])] * T_cycles,  # Unemployment benefits replacement rate for each sector
    "taxrate": [np.array([0.0, 0.0])] * T_cycles,  # Tax Rate for each sector
    "labor": [np.array([1.0, 1.0])] * T_cycles,  # Labor for each sector
    "wage": [np.array([1.0, 1.0])] * T_cycles,  # Wage for each sector
    "BoroCnstArt": [0.0, 0.0],  # Borrowing constraint for the minimum allowable assets to end the period with  
    "LivPrb": [LivPrb * np.ones(2)] * T_cycles,
    "PermGroFac": [1.0 * np.ones(2)] * T_cycles,
    # Parameters that specify the income distribution over the lifecycle
    "PermShkCount" : 5,                    # Number of points in discrete approximation to permanent income shocks
    "TranShkCount" : 5,    
    
  
    # Parameters for constructing the "assets above minimum" grid
    "aXtraMax" : 500,                      # Maximum end-of-period "assets above minimum" value
    "aXtraCount" : 100,                     # Number of points in the base grid of "assets above minimum"
    # "BoroCnstArt": 0.0,
    
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
    "MrkvPrbsInit": [0.8, 0.2],

    
}

# to add taxes, labor, and wage. This ensures that we can shock each variable.
def function(taxrate, labor, wage):
    
    z = (1- taxrate)*labor*wage
    return z

init_twosector['TranShkMean_Func'] = [function]

# Make a consumer with serially correlated unemployment, subject to boom and bust cycles
init_twosector["global_markov"] = False

### Use the HARK one and add a different wage rate and tax process
def function(taxrate, labor, wage):
    
    z = (1- taxrate)*labor*wage
    return z


### Try new Type
TwoSectorExampleHANK = TwoSectorMarkovConsumerType(**init_twosector)



### RFREE WITH TIME VARYING
TwoSectorExampleHANK.assign_parameters(
    Rfree=[np.array(2 * [1.03])] * T_cycles
)

TwoSectorExampleHANK.solve()
print('here')

print(TwoSectorExampleHANK.completed_cycles)
TwoSectorExampleHANK.compute_steady_state()
CJACR, AJACR = TwoSectorExampleHANK.calc_jacobian("Rfree", 300)
plt.plot(CJACR[0])
plt.plot(CJACR[10])
plt.plot(CJACR[-1])
plt.show()

plt.plot(AJACR[0])
plt.plot(AJACR[10])
plt.plot(AJACR[-1])

plt.show()

# 285.30983078304945
# 10.120636925963431

# TwoSectorExampleHANK.solution[0].cFunc[0]
# TwoSectorExampleHANK.neutral_measure = True
# TwoSectorExampleHANK.define_distribution_grid()
# TwoSectorExampleHANK.calc_transition_matrix_Markov()
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