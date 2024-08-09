"""
This file creates a ConsumerType which can work in two sectors (Formal and Informal) with a fixed Transition Matrix 
between them.
Differences in sectors are: 
- permanent shock, 
- transitory shock, 
- wagerate, 
- taxrate, 
- unemployment probability.
- Income Unemployment

Additionally, being in the informal sector requires an artificial borrowing constraint.

Author:
Adrian Monninger
amonnin1@jhu.edu

For the IMF.
"""

from HARK.ConsumptionSaving.ConsMarkovModel import ConsMarkovSolver, MarkovConsumerType
from ConsIndShockModel_HANK import HANKIncShkDstn
from HARK.ConsumptionSaving.ConsIndShockModel import init_idiosyncratic_shocks, IndShockConsumerType
import numpy as np
from utilities_TwoSectorModel import gen_tran_matrix_1D_Markov #, gen_tran_matrix_1D_Markov_weighted
from scipy import sparse as sp
from copy import deepcopy

from HARK.distribution import (
    DiscreteDistribution,
    DiscreteDistributionLabeled,
    IndexDistribution,
    Lognormal,
    MeanOneLogNormal,
    Uniform,
    add_discrete_outcome_constant_mean,
    combine_indep_dstns,
    expected,
)

from HARK.utilities import (
    construct_assets_grid,
    gen_tran_matrix_1D,
    gen_tran_matrix_2D,
    jump_to_grid_1D,
    jump_to_grid_2D,
    make_grid_exp_mult,
)

init_TwoSectorConsMarkov = dict(
    init_idiosyncratic_shocks,
    **{         
        # assets above grid parameters
        "mCount": 100,  # Number of gridpoints in market resources grid
        "mMax": 1000,  # Maximum value of market resources grid
        "mMin": 0.001,  # Minimum value of market resources grid
        "mFac": 3,  # Factor by which to multiply mCount to get number of gridpoints in market resources grid

        ### Markov Parameters
        "MrkvArray": [[0.9, 0.1], [0.1, 0.9]],  # Transition Matrix for Markov Process
        "global_markov": False,  # If True, then the Markov Process is the same for all agents
        "MrkvPrbsInit": [0.8, 0.2],
        })

class TwoSectorConsMarkovSolver(ConsMarkovSolver):

    ### Allow for the possibility of different artificial borrowing constraints
    def def_boundary(self):
        """
        Find the borrowing constraint for each current state and save it as an
        attribute of self for use by other methods.

        Parameters
        ----------
        none

        Returns
        -------
        none
        """
        self.BoroCnstNatAll = np.zeros(self.StateCount) + np.nan
        # Find the natural borrowing constraint conditional on next period's state
        for j in range(self.StateCount):
            PermShkMinNext = np.min(self.IncShkDstn_list[j].atoms[0])
            TranShkMinNext = np.min(self.IncShkDstn_list[j].atoms[1])
            self.BoroCnstNatAll[j] = (
                (self.solution_next.mNrmMin[j] - TranShkMinNext)
                * (self.PermGroFac_list[j] * PermShkMinNext)
                # / self.Rfree_list[0][j] # Rfree can be time carying. Use the first one.
                # / self.Rfree_list[j] # ORIGINAL ONE
            )

        self.BoroCnstNat_list = np.zeros(self.StateCount) + np.nan
        self.mNrmMin_list = np.zeros(self.StateCount) + np.nan
        self.BoroCnstDependency = np.zeros((self.StateCount, self.StateCount)) + np.nan
        # The natural borrowing constraint in each current state is the *highest*
        # among next-state-conditional natural borrowing constraints that could
        # occur from this current state.
        for i in range(self.StateCount):
            possible_next_states = self.MrkvArray[i, :] > 0
            self.BoroCnstNat_list[i] = np.max(self.BoroCnstNatAll[possible_next_states])

            # Explicitly handle the "None" case:
            if self.BoroCnstArt[i] is None:
                self.mNrmMin_list[i] = self.BoroCnstNat_list[i]
            else:
                self.mNrmMin_list[i] = np.max(
                    [self.BoroCnstNat_list[i], self.BoroCnstArt[i]]
                )
            self.BoroCnstDependency[i, :] = (
                self.BoroCnstNat_list[i] == self.BoroCnstNatAll
            )
        # Also creates a Boolean array indicating whether the natural borrowing
        # constraint *could* be hit when transitioning from i to j.



def _solve_TwoSectorConsMarkovSolver(
    solution_next,
    IncShkDstn,
    LivPrb,
    DiscFac,
    CRRA,
    Rfree,
    PermGroFac,
    MrkvArray,
    BoroCnstArt,
    aXtraGrid,
    vFuncBool,
    CubicBool,
):
    """
    Solves a single period consumption-saving problem with risky income and
    stochastic transitions between discrete states, in a Markov fashion.  Has
    identical inputs as solveConsIndShock, except for a discrete
    Markov transitionrule MrkvArray.  Markov states can differ in their interest
    factor, permanent growth factor, and income distribution, so the inputs Rfree,
    PermGroFac, and IncShkDstn are arrays or lists specifying those values in each
    (succeeding) Markov state.

    Parameters
    ----------
    solution_next : ConsumerSolution
        The solution to next period's one period problem.
    IncShkDstn_list : [distribution.Distribution]
        A length N list of income distributions in each succeeding Markov
        state.  Each income distribution is
        a discrete approximation to the income process at the
        beginning of the succeeding period.
    LivPrb : float
        Survival probability; likelihood of being alive at the beginning of
        the succeeding period.
    DiscFac : float
        Intertemporal discount factor for future utility.
    CRRA : float
        Coefficient of relative risk aversion.
    Rfree_list : np.array
        Risk free interest factor on end-of-period assets for each Markov
        state in the succeeding period.
    PermGroGac_list : float
        Expected permanent income growth factor at the end of this period
        for each Markov state in the succeeding period.
    MrkvArray : numpy.array
        An NxN array representing a Markov transition matrix between discrete
        states.  The i,j-th element of MrkvArray is the probability of
        moving from state i in period t to state j in period t+1.
    BoroCnstArt: float or None
        Borrowing constraint for the minimum allowable assets to end the
        period with.  If it is less than the natural borrowing constraint,
        then it is irrelevant; BoroCnstArt=None indicates no artificial bor-
        rowing constraint.
    aXtraGrid: np.array
        Array of "extra" end-of-period asset values-- assets above the
        absolute minimum acceptable level.
    vFuncBool: boolean
        An indicator for whether the value function should be computed and
        included in the reported solution.
    CubicBool: boolean
        An indicator for whether the solver should use cubic or linear inter-
        polation.

    Returns
    -------
    solution : ConsumerSolution
        The solution to the single period consumption-saving problem. Includes
        a consumption function cFunc (using cubic or linear splines), a marg-
        inal value function vPfunc, a minimum acceptable level of normalized
        market resources mNrmMin, normalized human wealth hNrm, and bounding
        MPCs MPCmin and MPCmax.  It might also have a value function vFunc
        and marginal marginal value function vPPfunc.  All of these attributes
        are lists or arrays, with elements corresponding to the current
        Markov state.  E.g. solution.cFunc[0] is the consumption function
        when in the i=0 Markov state this period.
    """
    solver = TwoSectorConsMarkovSolver(
        solution_next,
        IncShkDstn,
        LivPrb,
        DiscFac,
        CRRA,
        Rfree,
        PermGroFac,
        MrkvArray,
        BoroCnstArt,
        aXtraGrid,
        vFuncBool,
        CubicBool,
    )
    solution_now = solver.solve()
    return solution_now


class TwoSectorMarkovConsumerType(MarkovConsumerType):
    """
    An agent in the Markov consumption-saving model.  His problem is defined by a sequence
    of income distributions, survival probabilities, discount factors, and permanent
    income growth rates, as well as time invariant values for risk aversion, the
    interest rate, the grid of end-of-period assets, and how he is borrowing constrained.
    """ 

    def __init__(self, **kwds):
        params = init_TwoSectorConsMarkov.copy()
        params.update(kwds)
    
        MarkovConsumerType.__init__(self, **kwds)
        self.solve_one_period = _solve_TwoSectorConsMarkovSolver

        if not hasattr(self, "global_markov"):
            self.global_markov = False


    def update(self):
        """
        Update the income process, the assets grid, and the terminal solution.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.update_income_process_Markov()
        self.update_assets_grid()
        self.update_solution_terminal()

    ### Overwrite checking inputs as error message for IndcShkDistribution is wrong
    def check_markov_inputs(self):
        """
        Many parameters used by MarkovConsumerType are arrays.  Make sure those arrays are the
        right shape.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        StateCount = self.MrkvArray[0].shape[0]
        
        # Check that arrays in lists are the right shape
        for MrkvArray_t in self.MrkvArray:
            if not isinstance(MrkvArray_t, np.ndarray) or MrkvArray_t.shape != (
                StateCount,
                StateCount,
            ):
                raise ValueError(
                    "MrkvArray not the right shape, it should be of the size states*statres."
                )
        for LivPrb_t in self.LivPrb:
            if not isinstance(LivPrb_t, np.ndarray) or LivPrb_t.shape != (StateCount,):
                raise ValueError(
                    "Array in LivPrb is not the right shape, it should be an array of length equal to number of states"
                )
        for PermGroFac_t in self.PermGroFac:
            if not isinstance(PermGroFac_t, np.ndarray) or PermGroFac_t.shape != (
                StateCount,
            ):
                raise ValueError(
                    "Array in PermGroFac is not the right shape, it should be an array of length equal to number of states"
                )

        # Now check the income distribution.
        # Note IncShkDstn is (potentially) time-varying, so it is in time_vary.
        # Therefore it is a list, and each element of that list responds to the income distribution
        # at a particular point in time.  Each income distribution at a point in time should itself
        # be a list, with each element corresponding to the income distribution
        # conditional on a particular Markov state.
        # TODO: should this be a numpy array too?

        ### CHANGED
        for IncShkDstn_t in self.IncShkDstn:
            if not isinstance(IncShkDstn_t, list):
                raise ValueError(
                    "self.IncShkDstn is time varying and so must be a list"
                    + "of lists of Distributions, one per Markov State. Found "
                    + f"{self.IncShkDstn} instead"
                )
            elif len(IncShkDstn_t) != StateCount:
                raise ValueError(
                    "List in IncShkDstn is not the right length, it should be length equal to number of states"
                )

    ### Markov Income Process
    def update_income_process_Markov(self):
        """
        Updates income Process for each Markov State and saves it as a list of Distributions.
        """
        ### Check if parameter have the right format. They should be lists with as many entries as Markov States and each element should have an array with length T_cycle.       
        T_cycle =self.T_cycle



        IncShkDstn = IndexDistribution(
            engine=BufferStockIncShkDstn,
            conditional={
                "sigma_Perm": [0.06],
                "sigma_Tran": [0.2],
                "n_approx_Perm": [7],
                "n_approx_Tran": [7],
                "neutral_measure": [False],
                "UnempPrb": [0.0],
                "IncUnemp": [0.0],
            },
            RNG=self.RNG,
        )

        PermShkDstn = IndexDistribution(
            engine=LognormPermIncShk,
            conditional={
                "sigma": [0.06],
                "n_approx": [7],
                "neutral_measure": [False],
            },
        )

        TranShkDstn = IndexDistribution(
            engine=MixtureTranIncShk,
            conditional={
                "sigma": [0.2],
                "UnempPrb": [0.0],
                "IncUnemp": [0.0],
                "n_approx": [7],
            },
        )
        self.add_to_time_vary("IncShkDstn", "PermShkDstn", "TranShkDstn")
        self.PermShkDstn = PermShkDstn
        self.TranShkDstn = TranShkDstn

        ### Reshape the parameters to fit the IndexDistribution.
        ### Make sure that parameters are in the right format. If not, correct them.
        ### We should define parameters which are lists with the length of T_cycle and each element is either a list or an array.
        ### The parameters for the IndShockDistribution needs to be a list with of length T_cycle.

        params = [self.PermShkStd, self.TranShkStd, self.UnempPrb, self.IncUnemp, self.wage, self.taxrate, self.labor]
        reformatted_params = reformat_parameters(params)


        IncShkDstn_List = []
        for i in range(len(self.MrkvPrbsInit)):
            IncShkDstn_MrkV = IndexDistribution(
                engine=HANKIncShkDstn,
                conditional={
                    "sigma_Perm": reformatted_params[i][0],
                    "sigma_Tran": reformatted_params[i][1],
                    "n_approx_Perm": [self.PermShkCount] * T_cycle,
                    "n_approx_Tran": [self.TranShkCount] * T_cycle,
                    "neutral_measure": [self.neutral_measure] * T_cycle,
                    "UnempPrb": reformatted_params[i][2],
                    "IncUnemp": reformatted_params[i][3],
                    "wage": reformatted_params[i][4],                    "taxrate": reformatted_params[i][5],
                    "labor": reformatted_params[i][6],
                    "TranShkMean_Func": [self.TranShkMean_Func[0]] * T_cycle,
                },
                RNG=self.RNG,
            )        
            IncShkDstn_List.append(IncShkDstn_MrkV)


        ### Re-arrange to make a list with T_elements in which each element is a list with the Markov States
        time_list = []

        for i in range(self.T_cycle):
            time_list.append([
                IncShkDstn_List[0][i], IncShkDstn_List[1][i]
                ])
    

        self.IncShkDstn = time_list

    ### Unchanged
    def define_distribution_grid(
        self,
        dist_mGrid=None,
        dist_pGrid=None,
        m_density=0,
        num_pointsM=None,
        timestonest=None,
        num_pointsP=55,
        max_p_fac=30.0,
    ):
        """
        Defines the grid on which the distribution is defined. Stores the grid of market resources and permanent income as attributes of self.
        Grid for normalized market resources and permanent income may be prespecified
        as dist_mGrid and dist_pGrid, respectively. If not then default grid is computed based off given parameters.

        Parameters
        ----------
        dist_mGrid : np.array
                Prespecified grid for distribution over normalized market resources

        dist_pGrid : np.array
                Prespecified grid for distribution over permanent income.

        m_density: float
                Density of normalized market resources grid. Default value is mdensity = 0.
                Only affects grid of market resources if dist_mGrid=None.

        num_pointsM: float
                Number of gridpoints for market resources grid.

        num_pointsP: float
                 Number of gridpoints for permanent income.
                 This grid will be exponentiated by the function make_grid_exp_mult.

        max_p_fac : float
                Factor that scales the maximum value of permanent income grid.
                Larger values increases the maximum value of permanent income grid.

        Returns
        -------
        None
        """

        # If true Use Harmenberg 2021's Neutral Measure. For more information, see https://econ-ark.org/materials/harmenberg-aggregation?launch
        if not hasattr(self, "neutral_measure"):
            self.neutral_measure = False

        if num_pointsM == None:
            m_points = self.mCount
        else:
            m_points = num_pointsM

        if not isinstance(timestonest, int):
            timestonest = self.mFac
        else:
            timestonest = timestonest

        if self.cycles == 0:
            if not hasattr(dist_mGrid, "__len__"):
                mGrid = make_grid_exp_mult(
                    ming=self.mMin,
                    maxg=self.mMax,
                    ng=m_points,
                    timestonest=timestonest,
                )  # Generate Market resources grid given density and number of points

                for i in range(m_density):
                    m_shifted = np.delete(mGrid, -1)
                    m_shifted = np.insert(m_shifted, 0, 1.00000000e-04)
                    dist_betw_pts = mGrid - m_shifted
                    dist_betw_pts_half = dist_betw_pts / 2
                    new_A_grid = m_shifted + dist_betw_pts_half
                    mGrid = np.concatenate((mGrid, new_A_grid))
                    mGrid = np.sort(mGrid)

                self.dist_mGrid = mGrid

            else:
                # If grid of market resources prespecified then use as mgrid
                self.dist_mGrid = dist_mGrid

            if not hasattr(dist_pGrid, "__len__"):
                num_points = num_pointsP  # Number of permanent income gridpoints
                # Dist_pGrid is taken to cover most of the ergodic distribution
                # set variance of permanent income shocks
                p_variance = self.PermShkStd[0] ** 2
                # Maximum Permanent income value
                max_p = max_p_fac * (p_variance / (1 - self.LivPrb[0])) ** 0.5
                one_sided_grid = make_grid_exp_mult(
                    1.05 + 1e-3, np.exp(max_p), num_points, 3
                )
                self.dist_pGrid = np.append(
                    np.append(1.0 / np.fliplr([one_sided_grid])[0], np.ones(1)),
                    one_sided_grid,
                )  # Compute permanent income grid
                
            else:
                # If grid of permanent income prespecified then use it as pgrid
                self.dist_pGrid = dist_pGrid

            if (
                self.neutral_measure == True
            ):  # If true Use Harmenberg 2021's Neutral Measure. For more information, see https://econ-ark.org/materials/harmenberg-aggregation?launch
                self.dist_pGrid = np.array([1])

        elif self.cycles > 1:
            raise Exception(
                "define_distribution_grid requires cycles = 0 or cycles = 1"
            )

        elif self.T_cycle != 0:
            if num_pointsM == None:
                m_points = self.mCount
            else:
                m_points = num_pointsM

            if not hasattr(dist_mGrid, "__len__"):
                mGrid = make_grid_exp_mult(
                    ming=self.mMin,
                    maxg=self.mMax,
                    ng=m_points,
                    timestonest=timestonest,
                )  # Generate Market resources grid given density and number of points

                for i in range(m_density):
                    m_shifted = np.delete(mGrid, -1)
                    m_shifted = np.insert(m_shifted, 0, 1.00000000e-04)
                    dist_betw_pts = mGrid - m_shifted
                    dist_betw_pts_half = dist_betw_pts / 2
                    new_A_grid = m_shifted + dist_betw_pts_half
                    mGrid = np.concatenate((mGrid, new_A_grid))
                    mGrid = np.sort(mGrid)

                self.dist_mGrid = mGrid

            else:
                # If grid of market resources prespecified then use as mgrid
                self.dist_mGrid = dist_mGrid

            if not hasattr(dist_pGrid, "__len__"):
                self.dist_pGrid = []  # list of grids of permanent income

                for i in range(self.T_cycle):
                    num_points = num_pointsP
                    # Dist_pGrid is taken to cover most of the ergodic distribution
                    # set variance of permanent income shocks this period
                    p_variance = self.PermShkStd[i] ** 2
                    # Consider probability of staying alive this period
                    max_p = max_p_fac * (p_variance / (1 - self.LivPrb[i])) ** 0.5
                    one_sided_grid = make_grid_exp_mult(
                        1.05 + 1e-3, np.exp(max_p), num_points, 2
                    )

                    # Compute permanent income grid this period. Grid of permanent income may differ dependent on PermShkStd
                    dist_pGrid = np.append(
                        np.append(1.0 / np.fliplr([one_sided_grid])[0], np.ones(1)),
                        one_sided_grid,
                    )
                    self.dist_pGrid.append(dist_pGrid)

            else:
                # If grid of permanent income prespecified then use as pgrid
                self.dist_pGrid = dist_pGrid

            if (
                self.neutral_measure == True
            ):  # If true Use Harmenberg 2021's Neutral Measure. For more information, see https://econ-ark.org/materials/harmenberg-aggregation?launch
                self.dist_pGrid = self.T_cycle * [np.array([1])]
            
    def calc_transition_matrix_Markov(self, shk_dstn=None):
        """
        Calculates how the distribution of agents across market resources
        transitions from one period to the next. If finite horizon problem, then calculates
        a list of transition matrices, consumption and asset policy grids for each period of the problem.
        The transition matrix/matrices and consumption and asset policy grid(s) are stored as attributes of self.


        Parameters
        ----------
            shk_dstn: list
                list of income shock distributions. Each Income Shock Distribution should be a DiscreteDistribution Object (see Distribution.py)
        Returns
        -------
        None

        """

        if self.cycles == 0:  # Infinite Horizon Problem
            if not hasattr(shk_dstn, "pmv"):
                shk_dstn = self.IncShkDstn

            dist_mGrid = self.dist_mGrid  # Grid of market resources
            dist_pGrid = self.dist_pGrid  # Grid of permanent incomes

            MrkvPrbs = self.MrkvArray[0]  # Transition Matrix for Markov Process
            MrkvPrbsInit = self.MrkvPrbsInit # Initial Markov State Probabilities
            MrkvPrbsInit_array = np.array(MrkvPrbsInit)

            # Loop over Markov States
            PolGridShape = (len(MrkvPrbsInit), len(dist_mGrid))  # Shape of policy grids
            self.cPol_Grid = np.zeros(PolGridShape)  # Array of consumption policy grids for each Markov state
            self.aPol_Grid = np.zeros(PolGridShape)  # Array of asset policy grids for each Markov state
            bNext = np.zeros(PolGridShape)  # Array of bank balance grids for each period in T_cycle

            # Variables which change by Markov State
            ShockShape = (len(MrkvPrbsInit), len(shk_dstn[0][0].pmv))
            shk_prbs = np.zeros(ShockShape)  # Array of shock probabilities for each Markov state
            tran_shks = np.zeros(ShockShape)  # Array of transitory shocks for each Markov state
            perm_shks = np.zeros(ShockShape)  # Array of permanent shocks for each Markov state

            LivPrb = self.LivPrb[0]
            # LivPrb = []            

            for i in range(len(MrkvPrbsInit)):
                self.cPol_Grid[i] = self.solution[0].cFunc[i](dist_mGrid)
                self.aPol_Grid[i] = dist_mGrid - self.cPol_Grid[i]  # Asset policy grid in period k
                        
                if type(self.Rfree) == list:
                    bNext[i] = self.Rfree[0][i] * self.aPol_Grid[i]
                else:
                    bNext[i] = self.Rfree[i] * self.aPol_Grid[i]


                # Obtain shocks and shock probabilities from income distribution this period
                shk_prbs[i] = shk_dstn[0][i].pmv   # Probability of shocks this period
                # Transitory shocks this period
                tran_shks[i] = shk_dstn[0][i].atoms[1]
                # Permanent shocks this period
                perm_shks[i] = shk_dstn[0][i].atoms[0]

                # # Update probability of staying alive this period
                # LivPrb.append(self.LivPrb[0][i])

            if len(dist_pGrid) == 1:
                # New borns have this distribution (assumes start with no assets and permanent income=1)
                NewBornDist = np.zeros((len(MrkvPrbsInit), len(dist_mGrid)))  # To store modified NewBornDist arrays

                # Iterate over each element in MrkvPrbsInit and modify NewBornDist
                for i, prob in enumerate(MrkvPrbsInit):
                    NewBornDist[i] = jump_to_grid_1D(
                                        np.ones_like(tran_shks[i]),
                                        shk_prbs[i],
                                        dist_mGrid,) #* (MrkvPrbs[i,i]/prob)
                                        # np.zeros_like(tran_shks[i]), # No shocks at the beginning. Therefore, does not matter which shocks I take

                # Compute Transition Matrix given shocks and grids.
                TranMatrix_M = gen_tran_matrix_1D_Markov(
                    dist_mGrid,
                    MrkvPrbs,
                    bNext,
                    shk_prbs,
                    perm_shks,
                    tran_shks,
                    LivPrb,
                    NewBornDist,
                    MrkvPrbsInit_array,
                )
                
                self.tran_matrix = TranMatrix_M

            else:
                print('2D not done yet')

        elif self.cycles > 1:
            raise Exception("calc_transition_matrix requires cycles = 0 or cycles = 1")

        elif self.T_cycle != 0:  # finite horizon problem
            if not hasattr(shk_dstn, "pmv"):
                shk_dstn = self.IncShkDstn

            self.cPol_Grid = (
                [] )  # List of consumption policy grids for each period in T_cycle
            self.aPol_Grid = []  # List of asset policy grids for each period in T_cycle
            self.tran_matrix = []  # List of transition matrices

            dist_mGrid = self.dist_mGrid

            MrkvPrbsInit = self.MrkvPrbsInit # Initial Markov State Probabilities
            MrkvPrbsInit_array = np.array(MrkvPrbsInit)
            for k in range(self.T_cycle):
                
                if type(self.dist_pGrid) == list:
                    # Permanent income grid this period
                    dist_pGrid = self.dist_pGrid[k]
                else:
                    dist_pGrid = (
                        self.dist_pGrid
                    )  # If here then use prespecified permanent income grid

                # Markov Transition Matrix this period (Can be time varying as well)
                if len(self.MrkvArray) == 1:
                    # Markov Transition Matrix this period
                    MrkvPrbs = self.MrkvArray[0]
                else:
                    MrkvPrbs = self.MrkvArray[k]

                # Loop over Markov States
                PolGridShape = (len(MrkvPrbsInit), len(dist_mGrid))  # Shape of policy grids
                Cnow = np.zeros(PolGridShape)  # Array of consumption policy grids for each Markov state
                aNext = np.zeros(PolGridShape)  # Array of asset policy grids for each Markov state
                bNext = np.zeros(PolGridShape)  # Array of bank balance grids for each period in T_cycle

                # Variables which change by Markov State
                ShockShape = (len(MrkvPrbsInit), len(shk_dstn[k][0].pmv))
                shk_prbs = np.zeros(ShockShape)  # Array of shock probabilities for each Markov state
                tran_shks = np.zeros(ShockShape)  # Array of transitory shocks for each Markov state
                perm_shks = np.zeros(ShockShape)  # Array of permanent shocks for each Markov state

                LivPrb = self.LivPrb[k]

                for i in range(len(MrkvPrbsInit)):
                    Cnow[i] = self.solution[k].cFunc[i](dist_mGrid)
                    aNext[i] = dist_mGrid - Cnow[i]  # Asset policy grid in period k
                            
                    if type(self.Rfree) == list:
                        bNext[i] = self.Rfree[k][i] * aNext[i]
                    else:
                        bNext[i] = self.Rfree[i] * aNext[i]

                    # Obtain shocks and shock probabilities from income distribution this period
                    shk_prbs[i] = shk_dstn[k][i].pmv   # Probability of shocks this period
                    # Transitory shocks this period
                    tran_shks[i] = shk_dstn[k][i].atoms[1]
                    # Permanent shocks this period
                    perm_shks[i] = shk_dstn[k][i].atoms[0]

                self.cPol_Grid.append(Cnow)  # Add to list
                self.aPol_Grid.append(aNext)  # Add to list

                if len(dist_pGrid) == 1:
                    # New borns have this distribution (assumes start with no assets and permanent income=1)
                    NewBornDist = np.zeros((len(MrkvPrbsInit), len(dist_mGrid)))  # To store modified NewBornDist arrays

                    # Iterate over each element in MrkvPrbsInit and modify NewBornDist
                    for i, prob in enumerate(MrkvPrbsInit):
                        NewBornDist[i] = jump_to_grid_1D(
                                            np.ones_like(tran_shks[i]),
                                            shk_prbs[i],
                                            dist_mGrid,) #* (MrkvPrbs[i,i]/prob)
                                            # np.zeros_like(tran_shks[i]), # No shocks at the beginning. Therefore, does not matter which shocks I take
                    
                    # Compute Transition Matrix given shocks and grids.
                    TranMatrix_M = gen_tran_matrix_1D_Markov(
                        dist_mGrid,
                        MrkvPrbs,
                        bNext,
                        shk_prbs,
                        perm_shks,
                        tran_shks,
                        LivPrb,
                        NewBornDist,
                        MrkvPrbsInit_array,
                    )
                    self.tran_matrix.append(TranMatrix_M)

                else:
                    print('2D not done yet')


    def calc_ergodic_dist_Markov(self, transition_matrix=None):
        """
        Calculates the ergodic distribution across normalized market resources and
        permanent income as the eigenvector associated with the eigenvalue 1.
        The distribution is stored as attributes of self both as a vector and as a reshaped array with the ij'th element representing
        the probability of being at the i'th point on the mGrid and the j'th
        point on the pGrid.

        Parameters
        ----------
        transition_matrix: List
                    list with one transition matrix whose ergordic distribution is to be solved
        Returns
        -------
        None
        """

        if not isinstance(transition_matrix, list):
            transition_matrix = [self.tran_matrix]

        eigen, ergodic_distr = sp.linalg.eigs(
            transition_matrix[0], v0=np.ones(len(transition_matrix[0])), k=1, which="LM"
        )  # Solve for ergodic distribution
        ergodic_distr = ergodic_distr.real / np.sum(ergodic_distr.real)

        self.vec_erg_dstn = ergodic_distr  # distribution as a vector
        # distribution reshaped into len(mgrid) by len(pgrid) array
        self.erg_dstn = ergodic_distr.reshape(
            (len(self.MrkvPrbsInit) * len(self.dist_mGrid), len(self.dist_pGrid))
        )

    def calc_ergodic_dist_Markov_weighted(self, transition_matrix=None):
        """
        Calculates the ergodic distribution across normalized market resources and
        permanent income as the eigenvector associated with the eigenvalue 1.
        The distribution is stored as attributes of self both as a vector and as a reshaped array with the ij'th element representing
        the probability of being at the i'th point on the mGrid and the j'th
        point on the pGrid.

        Parameters
        ----------
        transition_matrix: List
                    list with one transition matrix whose ergordic distribution is to be solved
        Returns
        -------
        None
        """

        if not isinstance(transition_matrix, list):
            transition_matrix = [self.tran_matrix]

        ### Adapt transition_matrix given size of each Sector
        eigen, ergodic_distr = sp.linalg.eigs(
            transition_matrix[0], v0=np.ones(len(transition_matrix[0])), k=1, which="LM"
        )  # Solve for ergodic distribution
        ergodic_distr = ergodic_distr.real / np.sum(ergodic_distr.real)

        ergodic_distr_weighted = np.zeros_like(ergodic_distr)
        # weight by size of sector
        ergodic_distr_weighted[:len(self.dist_mGrid)] = ergodic_distr[:len(self.dist_mGrid)] * self.MrkvPrbsInit[0] * 2
        ergodic_distr_weighted[len(self.dist_mGrid):] = ergodic_distr[len(self.dist_mGrid):] * self.MrkvPrbsInit[1] * 2

        self.vec_erg_dstn = ergodic_distr_weighted  # distribution as a vector
        self.vec_erg_dstn_unweighted = ergodic_distr

        # distribution reshaped into len(mgrid) by len(pgrid) array
        self.erg_dstn = ergodic_distr_weighted.reshape(
            (len(self.MrkvPrbsInit) * len(self.dist_mGrid), len(self.dist_pGrid))
        )

    def compute_steady_state(self):
        # Compute steady state to perturb around
        # Only solve again if we haven't already done it or we solved a lifecycle model
        if not hasattr(self, 'solution') or self.cycles>0:
            self.cycles = 0
            self.solve()

        # Use Harmenberg Measure
        self.neutral_measure = True
        self.update_income_process_Markov()

        # Non stochastic simuation
        self.define_distribution_grid()
        self.calc_transition_matrix_Markov()

        # Policy Functions are now arrays with dimension (Nr of States x len(dist_mGrid)). We need to stack them to get a vector.
        self.c_ss = np.concatenate(self.cPol_Grid) # Normalized Consumption Policy grid
        self.a_ss = np.concatenate(self.aPol_Grid)  # Normalized Asset Policy grid

        #self.calc_ergodic_dist_Markov()  # Calculate ergodic distribution
        self.calc_ergodic_dist_Markov_weighted()  # Calculate ergodic distribution
        # Steady State Distribution as a vector (m*p x 1) where m is the number of gridpoints on the market resources grid
        ss_dstn = self.vec_erg_dstn

        ### Calculate Steady States for each Sector (weight them to make it as 100% of the population). Alternatively, do not divide them by size of sectors and sum them up.
        self.A_ss_Markv = np.zeros(len(self.MrkvPrbsInit))
        self.C_ss_Markv = np.zeros(len(self.MrkvPrbsInit))

        for i in range(len(self.MrkvPrbsInit)):
            self.A_ss_Markv[i] = np.dot(self.a_ss[i * len(self.dist_mGrid) : (i + 1) * len(self.dist_mGrid)], ss_dstn[i * len(self.dist_mGrid) : (i + 1) * len(self.dist_mGrid)])[0] /self.MrkvPrbsInit[i] #*len(self.MrkvPrbsInit)
            self.C_ss_Markv[i] = np.dot(self.c_ss[i * len(self.dist_mGrid) : (i + 1) * len(self.dist_mGrid)], ss_dstn[i * len(self.dist_mGrid) : (i + 1) * len(self.dist_mGrid)])[0] /self.MrkvPrbsInit[i] #*len(self.MrkvPrbsInit)                          
        
        ### Average over sectors with size of Sector
        self.A_ss = np.dot(self.A_ss_Markv, self.MrkvPrbsInit)
        self.C_ss = np.dot(self.C_ss_Markv, self.MrkvPrbsInit)


        return self.A_ss, self.C_ss, self.A_ss_Markv, self.C_ss_Markv

    def calc_jacobian(self, shk_param, position, T):
        """
        Calculates the Jacobians of aggregate consumption and aggregate assets. Parameters that can be shocked are
        LivPrb, PermShkStd,TranShkStd, DiscFac, UnempPrb, Rfree, IncUnemp, DiscFac .

        Parameters:
        ----------

        shk_param: string
            name of variable to be shocked

        position: int
            position of parameter to be shocked. 0 = both Markov Types.
        
        T: int
            dimension of Jacobian Matrix. Jacobian Matrix is a TxT square Matrix


        Returns
        ----------
        CJAC: numpy.array
            TxT Jacobian Matrix of Aggregate Consumption with respect to shk_param

        AJAC: numpy.array
            TxT Jacobian Matrix of Aggregate Assets with respect to shk_param

        """

        # Set up finite Horizon dictionary
        params = deepcopy(self.__dict__["parameters"])
        params["T_cycle"] = T  # Dimension of Jacobian Matrix
        params["cycles"] = 1  # required

        # Specify a dictionary of lists because problem we are solving is technically finite horizon so variables can be time varying (see section on fake news algorithm in https://onlinelibrary.wiley.com/doi/abs/10.3982/ECTA17434 )
        params["LivPrb"] = params["T_cycle"] * [self.LivPrb[0]]
        params["PermGroFac"] = params["T_cycle"] * [self.PermGroFac[0]]
        params["PermShkStd"] = params["T_cycle"] * [self.PermShkStd[0]]
        params["TranShkStd"] = params["T_cycle"] * [self.TranShkStd[0]]
        if type(self.Rfree)==np.ndarray:
            params["Rfree"] = params["T_cycle"] * [self.Rfree]
        else:
            params["Rfree"] = params["T_cycle"] * [self.Rfree[0]]
        params["UnempPrb"] = params["T_cycle"] * [self.UnempPrb[0]]
        params["IncUnemp"] = params["T_cycle"] * [self.IncUnemp[0]]
        
        params['wage'] = params['T_cycle']*[self.wage[0]]
        params['taxrate'] = params['T_cycle']*[self.taxrate[0]]
        params['labor'] = params['T_cycle']*[self.labor[0]]
        params['TranShkMean_Func'] = params['T_cycle']*[self.TranShkMean_Func[0]]

        params["MrkvArray"] = params["T_cycle"] * [self.MrkvArray[0]]
        # Create instance of a finite horizon agent
        FinHorizonAgent = TwoSectorMarkovConsumerType(**params)
        # FinHorizonAgent.cycles = 1  # required done above

        # delete Rfree from time invariant list since it varies overtime
        FinHorizonAgent.del_from_time_inv("Rfree")
        # Add Rfree to time varying list to be able to introduce time varying interest rates
        FinHorizonAgent.add_to_time_vary("Rfree")

        # Set Terminal Solution as Steady State Solutiuon
        FinHorizonAgent.solution_terminal = deepcopy(self.solution[0])

        dx = 0.0001  # Size of perturbation
        # Period in which the change in the interest rate occurs (second to last period)
        i = params["T_cycle"] - 1

        FinHorizonAgent.IncShkDstn = params["T_cycle"] * [self.IncShkDstn[0]]

        # If parameter is in time invariant list then add it to time vary list
        FinHorizonAgent.del_from_time_inv(shk_param)
        FinHorizonAgent.add_to_time_vary(shk_param)

        # this condition is because some attributes are specified as lists while other as floats
        # Based on input 'position', we can determine which parameter is shocked (all or only one of the Markov types)
        if type(getattr(self, shk_param)) == list:
            if position == 0:
                peturbed_list = (
                    (i) * [getattr(self, shk_param)[0]]
                    + [getattr(self, shk_param)[0] + dx]
                    + (params["T_cycle"] - i - 1) * [getattr(self, shk_param)[0]]
                )  # Sequence of interest rates the agent faces


            elif position ==1:
                peturbed_list = (
                    (i) * [getattr(self, shk_param)[0]]
                    + [np.array([getattr(self, shk_param)[0][0] + dx, getattr(self, shk_param)[0][1]])]
                    + (params["T_cycle"] - i - 1) * [getattr(self, shk_param)[0]]
                )                

            elif position == 2:
                peturbed_list = (
                    (i) * [getattr(self, shk_param)[0]]
                    + [np.array([getattr(self, shk_param)[0][0], getattr(self, shk_param)[0][1] + dx ])]
                    + (params["T_cycle"] - i - 1) * [getattr(self, shk_param)[0]]
                )  
            else:
                print('Position is greater than number of Markov States')                
        else:
            if position == 0:
                peturbed_list = (
                    (i) * [getattr(self, shk_param)]
                    + [getattr(self, shk_param) + dx]
                    + (params["T_cycle"] - i - 1) * [getattr(self, shk_param)]
                )

            elif position ==1:
                peturbed_list = (
                    (i) * [getattr(self, shk_param)]
                    + [np.array([getattr(self, shk_param)[0] + dx, getattr(self, shk_param)[1]])]
                    + (params["T_cycle"] - i - 1) * [getattr(self, shk_param)]
                )                

            elif position == 2:
                peturbed_list = (
                    (i) * [getattr(self, shk_param)]
                    + [np.array([getattr(self, shk_param)[0], getattr(self, shk_param)[1] + dx])]
                    + (params["T_cycle"] - i - 1) * [getattr(self, shk_param)]
                )  
            else:
                print('Position is greater than number of Markov States')
            
        setattr(FinHorizonAgent, shk_param, peturbed_list)

        # Update income process if perturbed parameter enters the income shock distribution
        # FinHorizonAgent.update_income_process()
        FinHorizonAgent.update_income_process_Markov()
        # Solve
        FinHorizonAgent.solve()

        #FinHorizonAgent.Rfree = params["T_cycle"] * [self.Rfree]
        # Use Harmenberg Neutral Measure
        FinHorizonAgent.neutral_measure = True
        FinHorizonAgent.update_income_process_Markov()

        # Calculate Transition Matrices
        FinHorizonAgent.define_distribution_grid()
        FinHorizonAgent.calc_transition_matrix_Markov()

        # Fake News Algorithm begins below ( To find fake news algorithm See page 2388 of https://onlinelibrary.wiley.com/doi/abs/10.3982/ECTA17434  )

        ##########
        # STEP 1 # of fake news algorithm, As in the paper for Curly Y and Curly D. Here the policies are over assets and consumption so we denote them as curly C and curly D.
        ##########
        # Policy Functions are now arrays with dimension (Nr of States x len(dist_mGrid)). We need to stack them to get a vector.
        a_ss = np.concatenate(self.aPol_Grid)  # steady state Asset Policy
        c_ss = np.concatenate(self.cPol_Grid)  # steady state Consumption Policy
        tranmat_ss = self.tran_matrix  # Steady State Transition Matrix

        # List of asset policies grids where households expect the shock to occur in the second to last Period
        ## Need to concatenate for each T
        a_t = []
        for t in range(T):
            a_t.append(np.concatenate(FinHorizonAgent.aPol_Grid[t]))
        # add steady state assets to list as it does not get appended in calc_transition_matrix method
        a_t.append(self.a_ss)

        # List of consumption policies grids where households expect the shock to occur in the second to last Period
        # c_t = [np.concatenate(FinHorizonAgent.cPol_Grid)]
        c_t = []
        for t in range(T):
            c_t.append(np.concatenate(FinHorizonAgent.cPol_Grid[t]))
        # add steady state consumption to list as it does not get appended in calc_transition_matrix method
        c_t.append(self.c_ss)

        da0_s = []  # Deviation of asset policy from steady state policy
        dc0_s = []  # Deviation of Consumption policy from steady state policy
        for i in range(T):
            da0_s.append(a_t[T - i] - a_ss)
            dc0_s.append(c_t[T - i] - c_ss)

        da0_s = np.array(da0_s)
        dc0_s = np.array(dc0_s)

        # Steady state distribution of market resources (permanent income weighted distribution)
        D_ss = self.vec_erg_dstn.T[0]
        dA0_s = []
        dC0_s = []
        for i in range(T):
            dA0_s.append(np.dot(da0_s[i], D_ss))
            dC0_s.append(np.dot(dc0_s[i], D_ss))

        dA0_s = np.array(dA0_s)
        # This is equivalent to the curly Y scalar detailed in the first step of the algorithm
        A_curl_s = dA0_s / dx

        dC0_s = np.array(dC0_s)
        C_curl_s = dC0_s / dx

        # List of computed transition matrices for each period
        tranmat_t = FinHorizonAgent.tran_matrix
        tranmat_t.append(tranmat_ss)

        # List of change in transition matrix relative to the steady state transition matrix
        dlambda0_s = []
        for i in range(T):
            dlambda0_s.append(tranmat_t[T  -   i] - tranmat_ss)

        dlambda0_s = np.array(dlambda0_s)

        dD0_s = []
        for i in range(T):
            dD0_s.append(np.dot(dlambda0_s[i], D_ss))

        dD0_s = np.array(dD0_s)
        D_curl_s = dD0_s / dx  # Curly D in the sequence space jacobian

        ########
        # STEP2 # of fake news algorithm
        ########

        # Expectation Vectors
        exp_vecs_a = []
        exp_vecs_c = []

        # First expectation vector is the steady state policy
        exp_vec_a = a_ss
        exp_vec_c = c_ss
        for i in range(T):
            exp_vecs_a.append(exp_vec_a)
            exp_vec_a = np.dot(tranmat_ss.T, exp_vec_a)

            exp_vecs_c.append(exp_vec_c)
            exp_vec_c = np.dot(tranmat_ss.T, exp_vec_c)

        # Turn expectation vectors into arrays
        exp_vecs_a = np.array(exp_vecs_a)
        exp_vecs_c = np.array(exp_vecs_c)

        #########
        # STEP3 # of the algorithm. In particular equation 26 of the published paper.
        #########
        # Fake news matrices
        Curl_F_A = np.zeros((T, T))  # Fake news matrix for assets
        Curl_F_C = np.zeros((T, T))  # Fake news matrix for consumption

        # First row of Fake News Matrix
        Curl_F_A[0] = A_curl_s
        Curl_F_C[0] = C_curl_s
        

        for i in range(T - 1):
            for j in range(T):
                Curl_F_A[i + 1][j] = np.dot(exp_vecs_a[i], D_curl_s[j])
                Curl_F_C[i + 1][j] = np.dot(exp_vecs_c[i], D_curl_s[j])

        ########
        # STEP4 #  of the algorithm
        ########
        
        # Function to compute jacobian matrix from fake news matrix
        def J_from_F(F):
            J = F.copy()
            for t in range(1, F.shape[0]):
                J[1:, t] += J[:-1, t-1]
            return J
        
        J_A = J_from_F(Curl_F_A)
        J_C = J_from_F(Curl_F_C)
        
        ########
        # Additional step due to compute Zeroth Column of the Jacobian
        ########   
         
        params = deepcopy(self.__dict__["parameters"])
        params["T_cycle"] = 2 # Dimension of Jacobian Matrix
        params["cycles"] = 1  # required
        
        # Specify a dictionary of lists because problem we are solving is technically finite horizon so variables can be time varying (see section on fake news algorithm in https://onlinelibrary.wiley.com/doi/abs/10.3982/ECTA17434 )
        params["LivPrb"] = params["T_cycle"] * [self.LivPrb[0]]
        params["PermGroFac"] = params["T_cycle"] * [self.PermGroFac[0]]
        params["PermShkStd"] = params["T_cycle"] * [self.PermShkStd[0]]
        params["TranShkStd"] = params["T_cycle"] * [self.TranShkStd[0]]
        if type(self.Rfree)==np.ndarray:
            params["Rfree"] = params["T_cycle"] * [self.Rfree]
        else:
            params["Rfree"] = params["T_cycle"] * [self.Rfree[0]]
        # params["Rfree"] = params["T_cycle"] * [self.Rfree[0]]
        params["UnempPrb"] = params["T_cycle"] * [self.UnempPrb[0]]
        params["IncUnemp"] = params["T_cycle"] * [self.IncUnemp[0]]
        
        params['wage'] = params['T_cycle']*[self.wage[0]]
        params['taxrate'] = params['T_cycle']*[self.taxrate[0]]
        params['labor'] = params['T_cycle']*[self.labor[0]]
        params['TranShkMean_Func'] = params['T_cycle']*[self.TranShkMean_Func[0]]
        # params['IncShkDstn'] = params['T_cycle']* [self.IncShkDstn[0]]
        params["MrkvArray"] = params["T_cycle"] * [self.MrkvArray[0]]
        
        if shk_param == 'DiscFac':
            
            params['DiscFac'] = params['T_cycle']*[self.DiscFac]

        # Create instance of a finite horizon agent for calculation of zeroth
        ZerothColAgent = TwoSectorMarkovConsumerType(**params)
        # Set Terminal Solution as Steady State Solutiuon
        ZerothColAgent.solution_terminal = deepcopy(self.solution[0])
      
        # If parameter is in time invariant list then add it to time vary list
        ZerothColAgent.del_from_time_inv(shk_param)
        ZerothColAgent.add_to_time_vary(shk_param)
        
        if type(getattr(self, shk_param)) == list:
            ZerothColAgent.shk_param = params['T_cycle'] * [getattr(self, shk_param)[0]]
        else:
            ZerothColAgent.shk_param = params['T_cycle'] * [getattr(self, shk_param)]

        # Update income process if perturbed parameter enters the income shock distribution
        ZerothColAgent.update_income_process_Markov()

        # Solve
        ZerothColAgent.solve()
        
        # this condition is because some attributes are specified as lists while other as floats
        # Based on input 'position', we can determine which parameter is shocked (all or only one of the Markov types)
        # this condition is because some attributes are specified as lists while other as floats
        if type(getattr(self, shk_param)) == list:
            if position == 0:
                peturbed_list = (
                    [getattr(self, shk_param)[0] + dx]
                    + (params["T_cycle"] - 1) * [getattr(self, shk_param)[0]]
                )  # Sequence of interest rates the agent faces


            elif position ==1:
                peturbed_list = (
                    [np.array([getattr(self, shk_param)[0][0] + dx, getattr(self, shk_param)[0][1]])]
                    + (params["T_cycle"] - 1) * [getattr(self, shk_param)[0]]
                )                

            elif position == 2:
                peturbed_list = (
                    [np.array([getattr(self, shk_param)[0][0], getattr(self, shk_param)[0][1] + dx ])]
                    + (params["T_cycle"] - 1) * [getattr(self, shk_param)[0]]
                )  
            else:
                print('Position is greater than number of Markov States')                
        else:
            if position == 0:
                peturbed_list = (
                    [getattr(self, shk_param) + dx]
                    + (params["T_cycle"] - 1) * [getattr(self, shk_param)]
                )

            elif position ==1:
                peturbed_list = (
                    [np.array([getattr(self, shk_param)[0] + dx, getattr(self, shk_param)[1]])]
                    + (params["T_cycle"] - 1) * [getattr(self, shk_param)]
                )                

            elif position == 2:
                peturbed_list = (
                    [np.array([getattr(self, shk_param)[0], getattr(self, shk_param)[1] + dx])]
                    + (params["T_cycle"] - 1) * [getattr(self, shk_param)]
                )  
            else:
                print('Position is greater than number of Markov States')

            
        setattr(ZerothColAgent, shk_param, peturbed_list) # Set attribute to agent

        # Use Harmenberg Neutral Measure
        ZerothColAgent.neutral_measure = True
        ZerothColAgent.update_income_process_Markov()

        # Calculate Transition Matrices
        ZerothColAgent.define_distribution_grid()
        ZerothColAgent.calc_transition_matrix_Markov()
        
        tranmat_t_zeroth_col = ZerothColAgent.tran_matrix
        # dstn_t_zeroth_col = self.vec_erg_dstn.T[0]
        dstn_t_zeroth_col = self.vec_erg_dstn_unweighted.T[0]
        dstn_t_zeroth_col_weighted = np.zeros_like(dstn_t_zeroth_col)
        C_t_no_sim = np.zeros(T)
        A_t_no_sim = np.zeros(T)

        C_t_no_sim_Mrkv = np.zeros(len(self.MrkvPrbsInit)) 
        A_t_no_sim_Mrkv = np.zeros(len(self.MrkvPrbsInit))
        for i in range(T):
            if i ==0:
                dstn_t_zeroth_col = np.dot(tranmat_t_zeroth_col[i], dstn_t_zeroth_col)
            else:
                dstn_t_zeroth_col = np.dot(tranmat_ss, dstn_t_zeroth_col)

            ### Need to weight again

            # weight by size of sector
            dstn_t_zeroth_col_weighted[:len(self.dist_mGrid)] = dstn_t_zeroth_col[:len(self.dist_mGrid)] * self.MrkvPrbsInit[0] * 2
            dstn_t_zeroth_col_weighted[len(self.dist_mGrid):] = dstn_t_zeroth_col[len(self.dist_mGrid):] * self.MrkvPrbsInit[1] * 2
            #
            # 
            for j in range(len(self.MrkvPrbsInit)):
                C_t_no_sim_Mrkv[j] =  np.dot( self.cPol_Grid[j], dstn_t_zeroth_col_weighted[j * len(self.dist_mGrid) : (j + 1) * len(self.dist_mGrid)]) /self.MrkvPrbsInit[j]
                A_t_no_sim_Mrkv[j] =  np.dot( self.aPol_Grid[j], dstn_t_zeroth_col_weighted[j * len(self.dist_mGrid) : (j + 1) * len(self.dist_mGrid)]) /self.MrkvPrbsInit[j]
                # C_t_no_sim_Mrkv[j] =  np.dot( self.cPol_Grid[j] * 2, dstn_t_zeroth_col_weighted[j * len(self.dist_mGrid) : (j + 1) * len(self.dist_mGrid)]) #/self.MrkvPrbsInit[j]
                # A_t_no_sim_Mrkv[j] =  np.dot( self.aPol_Grid[j] * 2, dstn_t_zeroth_col_weighted[j * len(self.dist_mGrid) : (j + 1) * len(self.dist_mGrid)]) #/self.MrkvPrbsInit[j]
            C_t_no_sim[i] = np.dot(C_t_no_sim_Mrkv, self.MrkvPrbsInit)
            A_t_no_sim[i] = np.dot(A_t_no_sim_Mrkv, self.MrkvPrbsInit)            

        J_A.T[0] = (A_t_no_sim - self.A_ss)/dx
        J_C.T[0] = (C_t_no_sim - self.C_ss)/dx

        # if position == 0:
        #     J_A.T[0] = (A_t_no_sim - self.A_ss)/dx
        #     J_C.T[0] = (C_t_no_sim - self.C_ss)/dx
        # else:
        #     J_A.T[0] = (A_t_no_sim - self.A_ss)/dx # self.MrkvPrbsInit[position - 1] * 
        #     J_C.T[0] = (C_t_no_sim - self.C_ss)/dx #self.MrkvPrbsInit[position - 1] *                   

        # # # Benchmark
        # J_C_both = 0.04871647994231232#        = J_C.T[0]

        # J_C.T[0][0] - J_C_both/2

        return J_C, J_A


    def construct_lognormal_income_process_unemployment_Markov(self, MrkVState):
        """
        Generates a list of discrete approximations to the income process for each
        life period, from end of life to beginning of life.  Permanent shocks are mean
        one lognormally distributed with standard deviation PermShkStd[t] during the
        working life, and degenerate at 1 in the retirement period.  Transitory shocks
        are mean one lognormally distributed with a point mass at IncUnemp with
        probability UnempPrb while working; they are mean one with a point mass at
        IncUnempRet with probability UnempPrbRet.  Retirement occurs
        after t=T_retire periods of working.

        Note 1: All time in this function runs forward, from t=0 to t=T

        Note 2: All parameters are passed as attributes of the input parameters.

        Parameters (passed as attributes of the input parameters)
        ----------
        PermShkStd : [float]
            List of standard deviations in log permanent income uncertainty during
            the agent's life.
        PermShkCount : int
            The number of approximation points to be used in the discrete approxima-
            tion to the permanent income shock distribution.
        TranShkStd : [float]
            List of standard deviations in log transitory income uncertainty during
            the agent's life.
        TranShkCount : int
            The number of approximation points to be used in the discrete approxima-
            tion to the permanent income shock distribution.
        UnempPrb : float or [float]
            The probability of becoming unemployed during the working period.
        UnempPrbRet : float or None
            The probability of not receiving typical retirement income when retired.
        T_retire : int
            The index value for the final working period in the agent's life.
            If T_retire <= 0 then there is no retirement.
        IncUnemp : float or [float]
            Transitory income received when unemployed.
        IncUnempRet : float or None
            Transitory income received while "unemployed" when retired.
        T_cycle :  int
            Total number of non-terminal periods in the consumer's sequence of periods.

        Returns
        -------
        IncShkDstn :  [distribution.Distribution]
            A list with T_cycle elements, each of which is a
            discrete approximation to the income process in a period.
        PermShkDstn : [[distribution.Distributiony]]
            A list with T_cycle elements, each of which
            a discrete approximation to the permanent income shocks.
        TranShkDstn : [[distribution.Distribution]]
            A list with T_cycle elements, each of which
            a discrete approximation to the transitory income shocks.
        """
        # Unpack the parameters from the input
        T_cycle = self.T_cycle
        PermShkStd = self.PermShkStd[MrkVState][0]
        PermShkCount = self.PermShkCount
        TranShkStd = self.TranShkStd[MrkVState][0]
        TranShkCount = self.TranShkCount
        T_retire = self.T_retire
        UnempPrb = self.UnempPrb[MrkVState][0]
        IncUnemp = self.IncUnemp[MrkVState][0]
        UnempPrbRet = self.UnempPrbRet
        IncUnempRet = self.IncUnempRet
        
        taxrate = self.taxrate[MrkVState][0]
        TranShkMean_Func = self.TranShkMean_Func[0]
        labor = self.labor[MrkVState][0]
        wage = self.wage[MrkVState][0]

        # formal_income_dist_HANK = HANKIncShkDstn(PermShkStd, TranShkStd, PermShkCount, TranShkCount, UnempPrb, IncUnemp, taxrate, TranShkMean_Func, labor, wage)
        ### HANKIncShkDstn works. The others not yet.
        
        if T_retire > 0:
            normal_length = T_retire
            retire_length = T_cycle - T_retire
        else:
            normal_length = T_cycle
            retire_length = 0

        if all(
            [
                isinstance(x, (float, int)) or (x is None)
                for x in [UnempPrb, IncUnemp, UnempPrbRet, IncUnempRet]
            ]
        ):
            UnempPrb_list = [UnempPrb] * normal_length + [UnempPrbRet] * retire_length
            IncUnemp_list = [IncUnemp] * normal_length + [IncUnempRet] * retire_length

        elif all([isinstance(x, list) for x in [UnempPrb, IncUnemp]]):
            UnempPrb_list = UnempPrb
            IncUnemp_list = IncUnemp

        else:
            raise Exception(
                "Unemployment must be specified either using floats for UnempPrb,"
                + "IncUnemp, UnempPrbRet, and IncUnempRet, in which case the "
                + "unemployment probability and income change only with retirement, or "
                + "using lists of length T_cycle for UnempPrb and IncUnemp, specifying "
                + "each feature at every age."
            )

        PermShkCount_list = [PermShkCount] * normal_length + [1] * retire_length
        TranShkCount_list = [TranShkCount] * normal_length + [1] * retire_length

        if not hasattr(self, "neutral_measure"):
            self.neutral_measure = False

        neutral_measure_list = [self.neutral_measure] * len(PermShkCount_list)

        PermShkDstn = IndexDistribution(
            engine=LognormPermIncShk,
            conditional={
                "sigma": PermShkStd,
                "n_approx": PermShkCount_list,
                "neutral_measure": neutral_measure_list,
            },
        )
        
        if self.HANK == True:
            IncShkDstn = IndexDistribution(
              engine=HANKIncShkDstn,
              conditional={
                  "sigma_Perm": PermShkStd,
                  "sigma_Tran": TranShkStd,
                  "n_approx_Perm": PermShkCount_list,
                  "n_approx_Tran": TranShkCount_list,
                  "neutral_measure": neutral_measure_list,
                  "UnempPrb": UnempPrb_list,
                  "IncUnemp": IncUnemp_list,
                  "wage": wage,
                  "taxrate": taxrate,
                  "labor": labor,
                  "TranShkMean_Func": TranShkMean_Func,
              },
              RNG=self.RNG,
          )
            
            
            TranShkDstn = IndexDistribution(
                engine=MixtureTranIncShk_HANK,
                conditional={
                    "sigma": TranShkStd,
                    "UnempPrb": UnempPrb_list,
                    "IncUnemp": IncUnemp_list,
                    "n_approx": TranShkCount_list,
                    "wage": wage,
                    "taxrate": taxrate,
                    "labor": labor,
                    "TranShkMean_Func": TranShkMean_Func,
                    
                    
                },
            )
    
        else:
                
            IncShkDstn = IndexDistribution(
                engine=BufferStockIncShkDstn,
                conditional={
                    "sigma_Perm": PermShkStd,
                    "sigma_Tran": TranShkStd,
                    "n_approx_Perm": PermShkCount_list,
                    "n_approx_Tran": TranShkCount_list,
                    "neutral_measure": neutral_measure_list,
                    "UnempPrb": UnempPrb_list,
                    "IncUnemp": IncUnemp_list,
                },
                RNG=self.RNG,
            )
            
            TranShkDstn = IndexDistribution(
                engine=MixtureTranIncShk,
                conditional={
                    "sigma": TranShkStd,
                    "UnempPrb": UnempPrb_list,
                    "IncUnemp": IncUnemp_list,
                    "n_approx": TranShkCount_list,
                },
            )   
        

        return IncShkDstn, PermShkDstn, TranShkDstn


class BufferStockIncShkDstn(DiscreteDistributionLabeled):
    """
    A one-period distribution object for the joint distribution of income
    shocks (permanent and transitory), as modeled in the Buffer Stock Theory
    paper:
        - Lognormal, discretized permanent income shocks.
        - Transitory shocks that are a mixture of:
            - A lognormal distribution in normal times.
            - An "unemployment" shock.

    Parameters
    ----------
    sigma_Perm : float
        Standard deviation of the log- permanent shock.
    sigma_Tran : float
        Standard deviation of the log- transitory shock.
    n_approx_Perm : int
        Number of points to use in the discrete approximation of the permanent shock.
    n_approx_Tran : int
        Number of points to use in the discrete approximation of the transitory shock.
    UnempPrb : float
        Probability of the "unemployment" shock.
    IncUnemp : float
        Income shock in the "unemployment" state.
    neutral_measure : Bool, optional
        Whether to use Hamenberg's permanent-income-neutral measure. The default is False.
    seed : int, optional
        Random seed. The default is 0.

    Returns
    -------
    IncShkDstn : DiscreteDistribution
        Income shock distribution.

    """

    def __init__(
        self,
        sigma_Perm,
        sigma_Tran,
        n_approx_Perm,
        n_approx_Tran,
        UnempPrb,
        IncUnemp,
        neutral_measure=False,
        seed=0,
    ):
        perm_dstn = LognormPermIncShk(
            sigma=sigma_Perm, n_approx=n_approx_Perm, neutral_measure=neutral_measure
        )
        tran_dstn = MixtureTranIncShk(
            sigma=sigma_Tran,
            UnempPrb=UnempPrb,
            IncUnemp=IncUnemp,
            n_approx=n_approx_Tran,
        )

        joint_dstn = combine_indep_dstns(perm_dstn, tran_dstn)

        super().__init__(
            name="Joint distribution of permanent and transitory shocks to income",
            var_names=["PermShk", "TranShk"],
            pmv=joint_dstn.pmv,
            atoms=joint_dstn.atoms,
            seed=seed,
        )



class HANKIncShkDstn(DiscreteDistributionLabeled):
    """
    A one-period distribution object for the joint distribution of income
    shocks (permanent and transitory), as modeled in the Buffer Stock Theory
    paper:
        - Lognormal, discretized permanent income shocks.
        - Transitory shocks that are a mixture of:
            - A lognormal distribution in normal times.
            - An "unemployment" shock.
    Parameters
    ----------
    sigma_Perm : float
        Standard deviation of the log- permanent shock.
    sigma_Tran : float
        Standard deviation of the log- transitory shock.
    n_approx_Perm : int
        Number of points to use in the discrete approximation of the permanent shock.
    n_approx_Tran : int
        Number of points to use in the discrete approximation of the transitory shock.
    UnempPrb : float
        Probability of the "unemployment" shock.
    IncUnemp : float
        Income shock in the "unemployment" state.
    neutral_measure : Bool, optional
        Whether to use Harmenberg's permanent-income-neutral measure. The default is False.
    seed : int, optional
        Random seed. The default is 0.
    Returns
    -------
    IncShkDstn : DiscreteDistribution
        Income shock distribution.
    """

    def __init__(
        self,
        sigma_Perm,
        sigma_Tran,
        n_approx_Perm,
        n_approx_Tran,
        UnempPrb,
        IncUnemp,
        taxrate,
        TranShkMean_Func,
        labor,
        wage,
        
        neutral_measure=False,
        seed=0,
    ):
        perm_dstn = LognormPermIncShk(
            sigma=sigma_Perm, n_approx=n_approx_Perm, neutral_measure=neutral_measure
        )
        tran_dstn = MixtureTranIncShk_HANK(
            sigma=sigma_Tran,
            UnempPrb=UnempPrb,
            IncUnemp=IncUnemp,
            n_approx=n_approx_Tran,
            wage = wage,
            labor = labor,
            taxrate = taxrate,
            TranShkMean_Func = TranShkMean_Func
        )

        joint_dstn = combine_indep_dstns(perm_dstn, tran_dstn)


        super().__init__(
            name="HANK",
            var_names=["PermShk", "TranShk"],
            pmv=joint_dstn.pmv,
            atoms=joint_dstn.atoms,
            seed=seed,
        )
        

class LognormPermIncShk(DiscreteDistribution):
    """
    A one-period distribution of a multiplicative lognormal permanent income shock.

    Parameters
    ----------
    sigma : float
        Standard deviation of the log-shock.
    n_approx : int
        Number of points to use in the discrete approximation.
    neutral_measure : Bool, optional
        Whether to use Hamenberg's permanent-income-neutral measure. The default is False.
    seed : int, optional
        Random seed. The default is 0.

    Returns
    -------
    PermShkDstn : DiscreteDistribution
        Permanent income shock distribution.

    """

    def __init__(self, sigma, n_approx, neutral_measure=False, seed=0):
        # Construct an auxiliary discretized normal
        logn_approx = MeanOneLogNormal(sigma).discretize(
            n_approx if sigma > 0.0 else 1, method="equiprobable", tail_N=0
        )
        # Change the pmv if necessary
        if neutral_measure:
            logn_approx.pmv = (logn_approx.atoms * logn_approx.pmv).flatten()

        super().__init__(pmv=logn_approx.pmv, atoms=logn_approx.atoms, seed=seed)


class MixtureTranIncShk(DiscreteDistribution):
    """
    A one-period distribution for transitory income shocks that are a mixture
    between a log-normal and a single-value unemployment shock.

    Parameters
    ----------
    sigma : float
        Standard deviation of the log-shock.
    UnempPrb : float
        Probability of the "unemployment" shock.
    IncUnemp : float
        Income shock in the "unemployment" state.
    n_approx : int
        Number of points to use in the discrete approximation.
    seed : int, optional
        Random seed. The default is 0.

    Returns
    -------
    TranShkDstn : DiscreteDistribution
        Transitory income shock distribution.

    """

    def __init__(self, sigma, UnempPrb, IncUnemp, n_approx, seed=0):
        dstn_approx = MeanOneLogNormal(sigma).discretize(
            n_approx if sigma > 0.0 else 1, method="equiprobable", tail_N=0
        )
        if UnempPrb > 0.0:
            dstn_approx = add_discrete_outcome_constant_mean(
                dstn_approx, p=UnempPrb, x=IncUnemp
            )

        super().__init__(pmv=dstn_approx.pmv, atoms=dstn_approx.atoms, seed=seed)



class MixtureTranIncShk_HANK(DiscreteDistribution):
    """
    A one-period distribution for transitory income shocks that are a mixture
    between a log-normal and a single-value unemployment shock.
    Parameters
    ----------
    sigma : float
        Standard deviation of the log-shock.
    UnempPrb : float
        Probability of the "unemployment" shock.
    IncUnemp : float
        Income shock in the "unemployment" state.
    n_approx : int
        Number of points to use in the discrete approximation.
    seed : int, optional
        Random seed. The default is 0.
    Returns
    -------
    TranShkDstn : DiscreteDistribution
        Transitory income shock distribution.
    """

    def __init__(self, sigma, UnempPrb, IncUnemp, n_approx,    wage,  labor, taxrate, TranShkMean_Func, seed=0):
        dstn_approx = MeanOneLogNormal(sigma).discretize(
            n_approx if sigma > 0.0 else 1,  method="equiprobable",tail_N=0
        )
        
        if UnempPrb > 0.0:
            dstn_approx = add_discrete_outcome_constant_mean(
                dstn_approx, p=UnempPrb, x=IncUnemp
            )
            
        dstn_approx.atoms = dstn_approx.atoms*TranShkMean_Func(taxrate,labor,wage)


        super().__init__(pmv=dstn_approx.pmv, atoms=dstn_approx.atoms, seed=seed)


# def ensure_correct_format(variables, markv, T_cycle):
#     """
#     Ensure that the variables are in the correct format.
#     Each variable should be a list of length T_cycle, where each element is an array of length markv.
#     """
#     corrected_variables = []

#     for var in variables:
#         if isinstance(var, list):
#             if len(var) == T_cycle:
#                 # If var is already a list of the correct length
#                 corrected_var = [np.array(sub_var) if isinstance(sub_var, (list, np.ndarray)) else np.array([sub_var]*markv) for sub_var in var]
#             else:
#                 # If var is a list but not of the correct length, extend it
#                 corrected_var = [np.array(var[i % len(var)]) if isinstance(var[i % len(var)], (list, np.ndarray)) else np.array([var[i % len(var)]]*markv) for i in range(T_cycle)]
#         elif isinstance(var, (float, int)):
#             # If var is a float or int, create a list with T_cycle elements, each being an array with markv entries
#             corrected_var = [np.array([var]*markv) for _ in range(T_cycle)]
#         elif isinstance(var, np.ndarray):
#             if var.shape == (T_cycle, markv):
#                 corrected_var = [var[i] for i in range(T_cycle)]
#             elif var.shape[0] == T_cycle:
#                 corrected_var = [np.array(var[i]) if var.shape[1] == markv else np.array([var[i]]*markv) for i in range(T_cycle)]
#             else:
#                 corrected_var = [np.array(var)]*T_cycle
#         else:
#             raise ValueError(f"Unsupported variable type: {type(var)}")

#         corrected_variables.append(corrected_var)

#     return corrected_variables


# def reverse_format(variables, T_cycle, markv):
#     """
#     Reverse the format from a list of length T_cycle with each element being an array of length markv
#     to a list of length markv with each element being an array of length T_cycle.
#     """
#     reversed_variables = []
    
#     for var in variables:
#         if isinstance(var, list) and all(isinstance(sub_var, np.ndarray) for sub_var in var):
#             if len(var) == T_cycle and all(sub_var.size == markv for sub_var in var):
#                 # Transpose the list of arrays
#                 transposed_var = np.array(var).T.tolist()
#                 reversed_var = [np.array(transposed_var[i]) for i in range(markv)]
#             else:
#                 raise ValueError("Input variable list length or array sizes do not match T_cycle and markv")
#         else:
#             raise ValueError("Unsupported variable type or structure")

#         reversed_variables.append(reversed_var)

#     return reversed_variables


# Function to reformat the lists
def reformat_parameters(params):
    markv = len(params[0][0])  # Determine markv from the first element's length
    T_cycle = len(params[0])   # Determine T_cycle from the length of the parameter list

    # Initialize the reformatted parameters list
    reformatted_params = [[] for _ in range(markv)]

    for param in params:
        for j in range(markv):
            reformatted_params[j].append([param[i][j] for i in range(T_cycle)])

    return reformatted_params

