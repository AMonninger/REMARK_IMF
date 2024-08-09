import numpy as np
import numba
from copy import copy, deepcopy

from HARK.utilities import (
    construct_assets_grid,
    gen_tran_matrix_1D,
    gen_tran_matrix_2D,
    jump_to_grid_1D,
    jump_to_grid_2D,
    make_grid_exp_mult,
)


@numba.njit(parallel=True)
def gen_tran_matrix_1D_Markov(
    dist_mGrid, MrkvPrbs, bNext, shk_prbs, perm_shks, tran_shks, LivPrb, NewBornDist, MrkvPrbsInit
):
    """
    Computes Transition Matrix across normalized market resources.
    This function is built to non-stochastic simulate the IndShockConsumerType.
    This function is used exclusively when Harmenberg Neutral Measure is applied and/or if permanent income is not a state variable
    For more information, see https://econ-ark.org/materials/harmenberg-aggregation?launch

    Parameters
    ----------
    dist_mGrid : np.array
        Grid over normalized market resources

    MrkvPrbs : np.array
        Probabilities of transitioning between different states of the Markov process

    bNext : np.array
        Grid over bank balances

    shk_prbs : np.array
        Array of shock probabilities over combinations of permanent and transitory shocks

    perm_shks : np.array
        Array of shocks to permanent income. Shocks should follow Harmenberg neutral measure

    tran_shks : np.array
        Array of shocks to transitory

    LivPrb : float
        Probability of not dying

    NewBornDist : np.array
        array representing distribution of newborns across grid of normalized market resources and grid of permanent income.

    Returns
    -------
    TranMatrix : np.array
        Transition Matrix over normalized market resources grid.


    """
    Size_sector_1 = MrkvPrbsInit[0]/MrkvPrbsInit[1] ### Ratio between sizes of sectors
    TranMatrix = np.zeros((len(MrkvPrbs) * len(dist_mGrid), len(MrkvPrbs) * len(dist_mGrid)))
    TranMatrix_NoMrkv_array = np.zeros((len(MrkvPrbs), len(dist_mGrid), len(dist_mGrid)))

    for Mrkv in numba.prange(len(MrkvPrbs)): #For each state of the Markov process (formal and informal sector)
        for i in numba.prange(len(dist_mGrid)):
            mNext_ij = (
                bNext[Mrkv][i] / perm_shks[Mrkv] + tran_shks[Mrkv]
            )  # Compute next period's market resources given todays bank balances bnext[i]

            TranMatrix_NoMrkv_array[Mrkv, :, i] += (
                LivPrb[Mrkv] * jump_to_grid_1D(mNext_ij, shk_prbs[Mrkv], dist_mGrid)
                + (1.0 - LivPrb[Mrkv]) * NewBornDist[Mrkv]) # Given state of the Markov Process   

    # Fill the upper left corner of TranMatrix
    TranMatrix[:len(dist_mGrid), :len(dist_mGrid)] = TranMatrix_NoMrkv_array[0] * MrkvPrbs[0, 0] #* Size_sector_1# Start Markov State 1 End in Markov State 1

    # Fill the upper right corner of TranMatrix
    TranMatrix[:len(dist_mGrid), len(dist_mGrid):] = TranMatrix_NoMrkv_array[0] * MrkvPrbs[0, 1] #* Size_sector_1# Start Markov State 1 End in Markov State 2

    # Fill the lower left corner of TranMatrix
    TranMatrix[len(dist_mGrid):, :len(dist_mGrid)] = TranMatrix_NoMrkv_array[1] * MrkvPrbs[1, 0] #/Size_sector_1 # Start Markov State 2 End in Markov State 1

    # Fill the lower right corner of TranMatrix
    TranMatrix[len(dist_mGrid):, len(dist_mGrid):] = TranMatrix_NoMrkv_array[1] * MrkvPrbs[1, 1] #/Size_sector_1# Start Markov State 2 End in Markov State 2

    return TranMatrix

# @numba.njit(parallel=True)
# def gen_tran_matrix_1D_Markov_weighted(
#     dist_mGrid, MrkvPrbs, bNext, shk_prbs, perm_shks, tran_shks, LivPrb, NewBornDist, MrkvPrbsInit
# ):
#     """
#     Computes Transition Matrix across normalized market resources.
#     This function is built to non-stochastic simulate the IndShockConsumerType.
#     This function is used exclusively when Harmenberg Neutral Measure is applied and/or if permanent income is not a state variable
#     For more information, see https://econ-ark.org/materials/harmenberg-aggregation?launch

#     Parameters
#     ----------
#     dist_mGrid : np.array
#         Grid over normalized market resources

#     MrkvPrbs : np.array
#         Probabilities of transitioning between different states of the Markov process

#     bNext : np.array
#         Grid over bank balances

#     shk_prbs : np.array
#         Array of shock probabilities over combinations of permanent and transitory shocks

#     perm_shks : np.array
#         Array of shocks to permanent income. Shocks should follow Harmenberg neutral measure

#     tran_shks : np.array
#         Array of shocks to transitory

#     LivPrb : float
#         Probability of not dying

#     NewBornDist : np.array
#         array representing distribution of newborns across grid of normalized market resources and grid of permanent income.

#     Returns
#     -------
#     TranMatrix : np.array
#         Transition Matrix over normalized market resources grid.


#     """
#     Size_sector_1 = MrkvPrbsInit[0]/MrkvPrbsInit[1] ### Ratio between sizes of sectors
#     TranMatrix = np.zeros((len(MrkvPrbs) * len(dist_mGrid), len(MrkvPrbs) * len(dist_mGrid)))
#     TranMatrix_NoMrkv_array = np.zeros((len(MrkvPrbs), len(dist_mGrid), len(dist_mGrid)))
#     weighted_tran_matrix = np.zeros((len(MrkvPrbs) * len(dist_mGrid), len(MrkvPrbs) * len(dist_mGrid)))
    
#     for Mrkv in numba.prange(len(MrkvPrbs)): #For each state of the Markov process (formal and informal sector)
#         for i in numba.prange(len(dist_mGrid)):
#             mNext_ij = (
#                 bNext[Mrkv][i] / perm_shks[Mrkv] + tran_shks[Mrkv]
#             )  # Compute next period's market resources given todays bank balances bnext[i]

#             TranMatrix_NoMrkv_array[Mrkv, :, i] += (
#                 LivPrb[Mrkv] * jump_to_grid_1D(mNext_ij, shk_prbs[Mrkv], dist_mGrid)
#                 + (1.0 - LivPrb[Mrkv]) * NewBornDist[Mrkv]) # Given state of the Markov Process   

#     # Fill the upper left corner of TranMatrix
#     TranMatrix[:len(dist_mGrid), :len(dist_mGrid)] = TranMatrix_NoMrkv_array[0] * MrkvPrbs[0, 0] #* MrkvPrbsInit[0] * 2 #* Size_sector_1# Start Markov State 1 End in Markov State 1

#     # Fill the upper right corner of TranMatrix
#     TranMatrix[:len(dist_mGrid), len(dist_mGrid):] = TranMatrix_NoMrkv_array[0] * MrkvPrbs[0, 1] #* MrkvPrbsInit[0] * 2#* Size_sector_1# Start Markov State 1 End in Markov State 2

#     # Fill the lower left corner of TranMatrix
#     TranMatrix[len(dist_mGrid):, :len(dist_mGrid)] = TranMatrix_NoMrkv_array[1] * MrkvPrbs[1, 0] #* MrkvPrbsInit[1] * 2#/Size_sector_1 # Start Markov State 2 End in Markov State 1

#     # Fill the lower right corner of TranMatrix
#     TranMatrix[len(dist_mGrid):, len(dist_mGrid):] = TranMatrix_NoMrkv_array[1] * MrkvPrbs[1, 1] #* MrkvPrbsInit[1] * 2#/Size_sector_1# Start Markov State 2 End in Markov State 2

#     ### multiply first 400 rows by 0.8 and multiply the last 400 rows by 0.2
#     for i in range(len(dist_mGrid)):
#         weighted_tran_matrix[i] = MrkvPrbsInit[0] * TranMatrix[i] * 2
#     for i in range(len(dist_mGrid), len(dist_mGrid) * 2):
#         weighted_tran_matrix[i] = MrkvPrbsInit[1] * TranMatrix[i] * 2

#     return weighted_tran_matrix

# @numba.njit(parallel=True)
def gen_tran_matrix_1D_Markov_OLD(
    dist_mGrid, MrkvPrbs, bNext_Grid, shk_prbs, perm_shks, tran_shks, LivPrb, NewBornDist
):
    """
    Computes Transition Matrix across normalized market resources.
    This function is built to non-stochastic simulate the IndShockConsumerType.
    This function is used exclusively when Harmenberg Neutral Measure is applied and/or if permanent income is not a state variable
    For more information, see https://econ-ark.org/materials/harmenberg-aggregation?launch

    Parameters
    ----------
    dist_mGrid : np.array
        Grid over normalized market resources

    MrkvPrbs : np.array
        Probabilities of transitioning between different states of the Markov process

    bNext_Grid : Lif of np.array
        Grid over bank balances

    shk_prbs : np.array
        Array of shock probabilities over combinations of permanent and transitory shocks

    perm_shks : np.array
        Array of shocks to permanent income. Shocks should follow Harmenberg neutral measure

    tran_shks : np.array
        Array of shocks to transitory

    LivPrb : float
        Probability of not dying

    NewBornDist : np.array
        array representing distribution of newborns across grid of normalized market resources and grid of permanent income.

    Returns
    -------
    TranMatrix : np.array
        Transition Matrix over normalized market resources grid.


    """

    TranMatrix = np.zeros((len(MrkvPrbs) * len(dist_mGrid), len(MrkvPrbs) * len(dist_mGrid)))
    TranMatrix_NoMrkv_List = []

    for Mrkv in numba.prange(len(MrkvPrbs)): #For each state of the Markov process (formal and informal sector)
        TranMatrix_NoMrkv_array = np.zeros((len(dist_mGrid), len(dist_mGrid)))
        for i in numba.prange(len(dist_mGrid)):
            mNext_ij = (
                bNext_Grid[Mrkv][i] / perm_shks[Mrkv] + tran_shks[Mrkv]
            )  # Compute next period's market resources given todays bank balances bnext[i]

            TranMatrix_NoMrkv_array[:, i] += (
                LivPrb[Mrkv] * jump_to_grid_1D(mNext_ij, shk_prbs[Mrkv], dist_mGrid)
                + (1.0 - LivPrb[Mrkv]) * NewBornDist[Mrkv]) # Given state of the Markov Process

        TranMatrix_NoMrkv_List.append(TranMatrix_NoMrkv_array)


    # Fill the upper left corner of TranMatrix
    TranMatrix[:len(dist_mGrid), :len(dist_mGrid)] = TranMatrix_NoMrkv_List[0] * MrkvPrbs[0, 0] # Start Markov State 1 End in Markov State 1

    # Fill the upper right corner of TranMatrix
    TranMatrix[:len(dist_mGrid), len(dist_mGrid):] = TranMatrix_NoMrkv_List[0] * MrkvPrbs[0, 1] # Start Markov State 1 End in Markov State 2

    # Fill the lower left corner of TranMatrix
    TranMatrix[len(dist_mGrid):, :len(dist_mGrid)] = TranMatrix_NoMrkv_List[1] * MrkvPrbs[1, 0] # Start Markov State 2 End in Markov State 1

    # Fill the lower right corner of TranMatrix
    TranMatrix[len(dist_mGrid):, len(dist_mGrid):] = TranMatrix_NoMrkv_List[1] * MrkvPrbs[1, 1] # Start Markov State 2 End in Markov State 2

    return TranMatrix
