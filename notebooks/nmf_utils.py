import os
import numpy as np
import pandas as pd
import time
from sklearn.decomposition import NMF
from sklearn.preprocessing import normalize
from scipy.cluster.hierarchy import linkage, cophenet
from tqdm import tqdm

OUTPUTS_PATH = r'../outputs'


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
    np.array
        co_H matrix

    """

    # Matrices Initialization
    (n, p) = np.shape(data)
    #c_w = np.zeros(n)
    #c_h = np.zeros(p)
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
            w4_init = np.random.rand(n, n_comp)
            h4_init = np.random.rand(p, n_comp).T
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

    return test_w, test_h


def perform_concordance_test(data, comp_min, comp_max, iter_max, n_runs):
    """Function that performs the concordance test for the NMF model.

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
    test_w = np.zeros(comp_max)
    test_h = np.zeros(comp_max)

    for n_comp in tqdm(range(comp_min, comp_max + 1)):
        nmf_model = NMF(n_components=n_comp, init='nndsvda', solver='cd', beta_loss='frobenius', max_iter=iter_max, random_state=0)
        w4 = nmf_model.fit_transform(data)
        h4 = nmf_model.components_.T
        error = np.linalg.norm(data - w4 @ h4.T)
        nmf_model_random = NMF(n_components=n_comp, init='custom', solver='cd', beta_loss='frobenius', max_iter=iter_max, random_state=0)
        np.random.seed(234)

        for i_run in tqdm(range(0, n_runs)):
            w4_init = np.random.rand(n, n_comp)
            h4_init = np.random.rand(p, n_comp).T
            w4_random = nmf_model_random.fit_transform(data, W=w4_init.copy(), H=h4_init.copy())
            h4_random = nmf_model_random.components_.T
            mcorr = np.corrcoef(w4, w4_random, rowvar=False)[0:n_comp, n_comp:2 * n_comp]
            mcorr[np.isnan(mcorr)] = -1
            idx = mcorr.argmax(axis=1)
            mean_corr = np.mean([mcorr[i, idx[i]] for i in range(0, n_comp)])
            x = np.unique(idx).shape[0]
            # Move concordance into [0,1] range instead of [1/n_comp, 1] to cancel bias for low ranks
            # and correct for mean correlation level in matches
            test_w[n_comp - 1] += ((x * (x - 1)) / ((n_comp - 1) * n_comp)) * mean_corr
            mcorr = np.corrcoef(h4, h4_random, rowvar=False)[0:n_comp, n_comp:2 * n_comp]
            mcorr[np.isnan(mcorr)] = -1
            idx = mcorr.argmax(axis=1)
            mean_corr = np.mean([mcorr[i, idx[i]] for i in range(0, n_comp)])
            x = np.unique(idx).shape[0]
            test_h[n_comp - 1] += ((x * (x - 1)) / ((n_comp - 1) * n_comp)) * mean_corr

        test_w[n_comp - 1] /= (n_runs * error)
        test_h[n_comp - 1] /= (n_runs * error)

    return test_w, test_h


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


def perform_rank_determination(data, comp_min, comp_max, iter_max, n_runs, method="cophenetic", save_res=False):
    """Function that performs the rank determination for the NMF model.

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
    method: str
        Method to use for the rank determination
    save_res: bool
        Saves the results if True

    Returns
    -------
    np.array
        W Test matrix
    np.array
        H Test matrix
    np.array
        Cusum
    float
        Estimated rank

    """

    time_start = time.time()

    # Performing the test
    if method == 'cophenetic':
        print(f"[INFO] Performing Cophenetic Test...")
        test_w, test_h = perform_cophenetic_test(data.copy(), comp_min, comp_max, iter_max, n_runs)
        filename_to_save = f"WH_cophenetic_test.csv"
    elif method == "concordance":
        print(f"[INFO] Performing Concordance Test...")
        test_w, test_h = perform_concordance_test(data.copy(), comp_min, comp_max, iter_max, n_runs)
        filename_to_save = f"WH_concordance_test.csv"
    else:
        test_w = None
        test_h = None
        filename_to_save = ""

    # Cusum Calculation
    print("[INFO] Calculating Cusum...")
    cusum = cusum_calculation(comp_min, comp_max, test_w, test_h)

    # Rank Estimation
    print("[INFO] Computing Rank Estimation...")
    estimated_rank = rank_estimation(cusum, comp_max)

    time_elapsed = (time.time() - time_start)

    print(f"[INFO] Elapsed time: {time_elapsed} seconds")

    # Saving results
    if save_res:
        data_to_save = np.concatenate((test_w, test_h), axis=0)
        np.savetxt(os.path.join(OUTPUTS_PATH, filename_to_save), data_to_save, delimiter=',')

    return test_w, test_h, cusum, estimated_rank
