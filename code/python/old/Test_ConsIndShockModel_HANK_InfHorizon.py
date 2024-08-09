from HARK.ConsumptionSaving.ConsMarkovModel import MarkovConsumerType
from HARK.ConsumptionSaving.ConsIndShockModel import init_idiosyncratic_shocks
from HARK.distribution import DiscreteDistributionLabeled
import matplotlib.pyplot as plt
from ConsIndShockModel_HANK import HANKIncShkDstn

from TwoSectorModel_HANK import TwoSectorMarkovConsumerType
import numpy as np
from copy import copy, deepcopy
### TESTING LIFECYCLE MODEL

### Define shared parameters
Rfree_f = 0.03
Rfree_i = 0.03
PermShkStd_f = 0.06
PermShkStd_i = 0.06
TranShkStd_f = 0.2
TranShkStd_i = 0.2
PermGroFac_f = 1.01
PermGroFac_i = 1.01
UnempPrb_f = 0.05
UnempPrb_i = 0.05
IncUnemp_f = 0.0
IncUnemp_i = 0.0
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

Formal_Size = 0.8

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

UnempPrb_f = 0.05
UnempPrb_i = 0.1
IncUnemp_f = 0.7
IncUnemp_i = 0.0

HANK_Dict_TwoSector_URisk = deepcopy(HANK_Dict_TwoSector)
HANK_Dict_TwoSector_URisk['UnempPrb'] = [np.array([UnempPrb_f, UnempPrb_i])]
HANK_Dict_TwoSector_URisk['IncUnemp'] = [np.array([IncUnemp_f, IncUnemp_i])]


Agent_TwoSector_URisk = TwoSectorMarkovConsumerType(**HANK_Dict_TwoSector_URisk)
Agent_TwoSector_URisk.solve()
A_SS_TwoSector_URisk, C_SS_TwoSector_URisk, A_SS_Mrkv_TwoSector_URisk, C_SS_Mrkv_TwoSector_URisk = Agent_TwoSector_URisk.compute_steady_state()
MPC_TwoSector_URisk = Agent_TwoSector_URisk.calc_jacobian('labor',T)[0][0][0]

STOP




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
    "Rfree": [np.array(2 * [1.0 + r_ss])]* T_cycles,                   # Interest factor on assets
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
# TwoSectorExampleHANK.assign_parameters(
#     Rfree=[np.array(2 * [1.03])] * T_cycles
# )


TwoSectorExampleHANK.solve()
TwoSectorExampleHANK.compute_steady_state()
TwoSectorExampleHANK.calc_jacobian('Rfree', 300)
print(TwoSectorExampleHANK.completed_cycles)

# TwoSectorExampleHANK.compute_steady_state()
# CJACR, AJACR = TwoSectorExampleHANK.calc_jacobian("Rfree", 300)
# plt.plot(CJACR[0])
# plt.plot(CJACR[10])
# plt.plot(CJACR[-1])
# plt.show()

# plt.plot(AJACR[0])
# plt.plot(AJACR[10])
# plt.plot(AJACR[-1])

# plt.show()

### TESTIN LIFECYCLE MODEL WITH LAST PERIOD INFINITE HORIZON
T = 3
shk_param = "Rfree"

# Set up finite Horizon dictionary
params = deepcopy(TwoSectorExampleHANK.__dict__["parameters"])
params["T_cycle"] = T  # Dimension of Jacobian Matrix
params["cycles"] = 1  # required

# Specify a dictionary of lists because problem we are solving is technically finite horizon so variables can be time varying (see section on fake news algorithm in https://onlinelibrary.wiley.com/doi/abs/10.3982/ECTA17434 )
params["LivPrb"] = params["T_cycle"] * [TwoSectorExampleHANK.LivPrb[0]]
params["PermGroFac"] = params["T_cycle"] * [TwoSectorExampleHANK.PermGroFac[0]]
params["PermShkStd"] = params["T_cycle"] * [TwoSectorExampleHANK.PermShkStd[0]]
params["TranShkStd"] = params["T_cycle"] * [TwoSectorExampleHANK.TranShkStd[0]]
params["Rfree"] = params["T_cycle"] * [TwoSectorExampleHANK.Rfree[0]]
params["UnempPrb"] = params["T_cycle"] * [TwoSectorExampleHANK.UnempPrb[0]]
params["IncUnemp"] = params["T_cycle"] * [TwoSectorExampleHANK.IncUnemp[0]]

params['wage'] = params['T_cycle']*[TwoSectorExampleHANK.wage[0]]
params['taxrate'] = params['T_cycle']*[TwoSectorExampleHANK.taxrate[0]]
params['labor'] = params['T_cycle']*[TwoSectorExampleHANK.labor[0]]
params['TranShkMean_Func'] = params['T_cycle']*[TwoSectorExampleHANK.TranShkMean_Func[0]]

params["MrkvArray"] = params["T_cycle"] * [TwoSectorExampleHANK.MrkvArray[0]]

# Create instance of a finite horizon agent
FinHorizonAgent = TwoSectorMarkovConsumerType(**params)
# FinHorizonAgent.cycles = 1  # required done above

# delete Rfree from time invariant list since it varies overtime
FinHorizonAgent.del_from_time_inv("Rfree")
# Add Rfree to time varying list to be able to introduce time varying interest rates
FinHorizonAgent.add_to_time_vary("Rfree")

# Set Terminal Solution as Steady State Consumption Function
# FinHorizonAgent.cFunc_terminal_ = deepcopy(TwoSectorExampleHANK.solution[0].cFunc)

FinHorizonAgent.solution_terminal = deepcopy(TwoSectorExampleHANK.solution[0])

FinHorizonAgent.solve()

### Plot policy functions
mGrid = np.linspace(0, 100, 1000)
plt.plot(mGrid, TwoSectorExampleHANK.solution[0].cFunc[0](mGrid), label = "Steady State Sector 1")
plt.plot(mGrid, TwoSectorExampleHANK.solution[0].cFunc[1](mGrid), label = "Steady State Sector 2")
plt.plot(mGrid, FinHorizonAgent.cFunc_terminal_[0](mGrid))
plt.plot(mGrid, FinHorizonAgent.cFunc_terminal_[1](mGrid))
# plt.plot(mGrid, FinHorizonAgent.solution[299].cFunc[0](mGrid))
# plt.plot(mGrid, FinHorizonAgent.solution[299].cFunc[1](mGrid))
plt.plot(mGrid, FinHorizonAgent.solution[T].cFunc[0](mGrid))
plt.plot(mGrid, FinHorizonAgent.solution[T].cFunc[1](mGrid))
plt.legend()
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