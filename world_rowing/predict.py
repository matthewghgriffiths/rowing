

import numpy as np 
import pandas as pd


def calc_boat_pace(live_data, distances=None):
    distances = distances or np.linspace(0, 2000, 401).astype(int)
    boat_pace = pd.DataFrame(
        np.vstack(
            [
                np.interp(
                    distances, 
                    live_data.distanceTravelled[cnt],
                    500 / live_data.metrePerSecond[cnt]
                )
                for cnt in live_data.metrePerSecond.columns
            ]
        ),
        index=live_data.metrePerSecond.columns,
        columns=distances
    )
    boat_pace.columns.name = 'distance'
    boat_pace.index.name = 'country'
    return boat_pace


def calc_all_boat_pace(race_live_data, distances=None, set_last=True):
    distances = distances or np.linspace(0, 2000, 401).astype(int)
    boat_pace = pd.concat({
        race_id: calc_boat_pace(live_data)
        for race_id, live_data in race_live_data.items()
    }) 
    if set_last:
        boat_pace.loc[:, distances[-1]] = boat_pace.loc[:, distances[-2]]

    return boat_pace


def fit_factor_analysis_regularised(X, d, F=None, W=None, delta=1, psi=1, niter=100):
    """
    d: number of factors
    delta: derivative regularisation factor
    psi: Gaussian magnitude regularisation factor
    """
    n, m = X.shape
    #initialise factors
    if F is None:
        U, s, Vt = np.linalg.svd(X)
        scale = np.diag(np.sqrt(s[:d]))
        W = U[:, :d].dot(scale)
        F = scale.dot(Vt[:d, :])
    if W is None:
        W = np.linalg.lstsq(F.T, X.T, rcond=None)[0].T

    D = np.zeros((m, m-1))
    ind = np.diag_indices(m-1)
    D[ind] = - 1
    D[ind[0] + 1, ind[1]] = 1

    # Do Expectation Maximisation
    Psi = psi * np.eye(n)  # Gaussian regularisation
    B = delta * D.dot(D.T) # Derivative regularisation
    Fp = F.copy()
    Wp = W.copy()
    for i in range(niter):
        A = Wp.T.dot(Psi.dot(Wp))
        S = A.diagonal()[:, None] +  B.diagonal()[None, :]
        C = 2 * W.T.dot(Psi.dot(X))
        D = (C - A.dot(Fp) - Fp.dot(B))/S/2
        Fp += D
        Wp = np.linalg.lstsq(Fp.T, X.T, rcond=None)[0].T

    return Wp, Fp
