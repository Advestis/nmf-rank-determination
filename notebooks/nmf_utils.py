import os
import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from sklearn.preprocessing import normalize
from scipy.cluster.hierarchy import linkage, cophenet
from tqdm import tqdm

def read_dataset(path_data, dataset_name):
    """Function that reads a dataset.

    Parameters
    ----------
    path_data: str
        Path to the dataset
    dataset_name: str
        The dataset name

    Returns
    -------
    pd.DataFrame
        Dataframe of the dataset
    np.array
        The dataset

    """

    # Reading the data
    df_out = pd.read_csv(os.path.join(path_data, dataset_name))

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

def perform_cophenetic_test(data, comp_min, comp_max, iter_max, n_runs):
    """Function that performs the cophenetic test for the NMF model.

    Parameters
    ----------
    data: np.array
        The dataset
    comp_min: int
        Minimum number of components
    comp_max: int
        Maximum number of components
    iter_max: int
        Maximum number of iterations
    n_runs: int
        Number of runs

    Returns
    -------
    np.array
        W Test matrix
    np.array
        H Test matrix

    """

    # Matrices Initialization
    (n, p) = np.shape(data)
    c_w = np.zeros(n)
    c_h = np.zeros(p)
    iln1 = np.triu_indices(n, 1)
    ilp1 = np.triu_indices(p, 1)
    test_w = np.zeros(comp_max)
    test_h = np.zeros(comp_max)

    for n_comp in tqdm(range(comp_min, comp_max + 1)):
        nmf_model = NMF(n_components=n_comp, init='nndsvda', solver='cd', beta_loss='frobenius', max_iter=iter_max,
                        random_state=0)
        w4 = nmf_model.fit_transform(data)
        h4 = nmf_model.components_.T
        error = np.linalg.norm(data - w4 @ h4.T)
        co_w = np.zeros(int(n * (n - 1) / 2))
        co_h = np.zeros(int(p * (p - 1) / 2))
        nmf_model_random = NMF(n_components=n_comp, init='custom', solver='cd', beta_loss='frobenius',
                               max_iter=iter_max, random_state=0)

        for _ in tqdm(range(0, n_runs)):
            w4_init = np.random.rand(n, n_comp); h4_init = np.random.rand(p, n_comp).T
            w4 = nmf_model_random.fit_transform(data, W=w4_init, H=h4_init)
            h4 = nmf_model_random.components_.T
            c_w = np.argmax(normalize(w4, axis=0), axis=1)
            c_h = np.argmax(normalize(h4, axis=0), axis=1)
            # co_w += np.array([c_w[i] == c_w[j] for i in range(0,n-1) for j in range(i+1,n)])
            # co_h += np.array([c_h[i] == c_h[j] for i in range(0,p-1) for j in range(i+1,p)])
            co_w += np.equal.outer(c_w, c_w)[iln1]
            co_h += np.equal.outer(c_h, c_h)[ilp1]

        co_w = 1 - co_w / n_runs
        co_h = 1 - co_h / n_runs

        cpc_w, cp_w = cophenet(linkage(co_w, method='ward'), co_w)
        cpc_h, cp_h = cophenet(linkage(co_h, method='ward'), co_h)

        test_w[n_comp - 1] = cpc_w / error
        test_h[n_comp - 1] = cpc_h / error

    return test_w, test_h, co_h


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
