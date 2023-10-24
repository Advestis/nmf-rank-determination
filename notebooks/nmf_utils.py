import os
import numpy as np
import pandas as pd


def read_dataset(path_data, dataset_name, show_data=False):
    """Function that reads a dataset.

    Parameters
    ----------
    path_data: str
        Path to the dataset
    dataset_name: str
        The dataset name
    show_data: bool
        Show data if True

    Returns
    -------
    pd.DataFrame
        Dataframe of the dataset
    np.array
        Description of 'm0' ?

    """

    # Reading the data
    df_out = pd.read_csv(os.path.join(path_data, dataset_name))
    if show_data:
        print(df_out)

    # Data processing
    if dataset_name == "swimmer_2.csv":
        m0 = df_out.values[:, 2:].astype(float)
    elif dataset_name == 'Sausage Raw NIR.csv':
        m0 = df_out.values[:, 8:].astype(float)
    elif dataset_name == 'ALL-AML Brunet.csv':
        # m0 = df.values[:, 2:].astype(float)
        m0 = np.log2(df_out.values[:, 2:].astype(float))
        # m0 -= np.repeat(np.min(m0, axis=0)[:, np.newaxis].T, n, axis=0)
        m0 -= np.repeat(np.min(m0, axis=0)[:, np.newaxis].T, np.shape(m0)[0], axis=0)
    else:
        m0 = None

    return df_out, m0


def cusum_calculation(comp_min, comp_max, w_test, h_test):
    """Function that calculates the cusum.

    Parameters
    ----------
    comp_min: int
        Minimum number of components
    comp_max: int
        Maximum number of components
    w_test: np.array
        W Test matrix
    h_test: np.array
        H Test matrix

    Returns
    -------
    np.array
        Cusum

    """

    cusum = np.zeros(comp_max)
    test = np.sqrt(w_test * h_test)
    cusum[comp_min - 2] = (test[comp_min - 2] - test[comp_min - 1] > 0)

    for nc in range(comp_min, comp_max):
        if test[nc - 1] - test[nc] > 0:
            deltax = 1
        else:
            deltax = -1
        cusum[nc - 1] = max(cusum[nc - 2] + deltax, 0)
    cusum[comp_max - 1] = cusum[comp_max - 2]

    return cusum


def rank_estimation(cusum, comp_max):
    """Function that estimates the rank.

    Parameters
    ----------
    cusum: np.array
        Cusum
    comp_max: int
        Maximum number of components

    Returns
    -------
    float
        Estimated rank

    """

    n_comp_est = 999

    for nc in range(1, comp_max - 2):
        if cusum[nc - 1] * cusum[nc] * cusum[nc + 1] > 0:
            n_comp_est = nc
            break

    print(f"[INFO] Estimated rank = {n_comp_est}")
    return n_comp_est
