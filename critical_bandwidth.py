import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import argrelextrema

from ApproxGaussianKDE import ApproxGaussianKDE as KDE



def critical_bandwidth(data, I=(-np.inf, np.inf), htol=1e-3):
    '''
    I is interval over which density is tested for unimodality
    '''
    hmax = (np.max(data)-np.min(data))/2.0
    return bisection_search_unimodal(0, hmax, htol, data, I)


def critical_bandwidth_m_modes(data, m, I=(-np.inf, np.inf), htol=1e-3):
        # I is interval over which density is tested for unimodality
    hmax = (np.max(data)-np.min(data))/2.0
    return bisection_search_most_m_modes(0, hmax, htol, data, m, I)


def bisection_search_unimodal(hmin, hmax, htol, data, I):
    '''
    Assuming fun(xmax) < 0.
    '''
    return bisection_search_most_m_modes(hmin, hmax, htol, data, 1, I)


def bisection_search_most_m_modes(hmin, hmax, htol, data, m, I):
    '''
    Assuming fun(xmax) < 0.
    '''
    if hmax-hmin < htol:
        return (hmin + hmax)/2.0
    hnew = (hmin + hmax)/2.0
    #print "hnew = {}".format(hnew)
    if kde_has_at_most_m_modes(hnew, data, m, I):  # upper bound for bandwidth
        return bisection_search_most_m_modes(hmin, hnew, htol, data, m, I)
    return bisection_search_most_m_modes(hnew, hmax, htol, data, m, I)


def is_unimodal_kde(h, data, I=(-np.inf, np.inf)):
    return kde_has_at_most_m_modes(h, data, 1, I)


def kde_has_at_most_m_modes(h, data, m, I=(-np.inf, np.inf)):
    # I is interval over which density is tested for unimodality
    xtol = h*0.05 # TODO: Compute error given xtol.
    kde = KDE(data, h)
    x_new = np.linspace(max(I[0], np.min(data)), min(I[1], np.max(data)), 10)
    x = np.zeros(0,)
    y = np.zeros(0,)
    while True:
        y_new = kde.evaluate_prop(x_new)
        x = merge_into(x_new, x)
        y = merge_into(y_new, y)
        # fig, ax = plt.subplots()
        # ax.plot(x, y)
        # ax.scatter(x, y, marker='+')
        # ax.scatter(x_new, y_new, marker='+', color='red')
        if len(argrelextrema(np.hstack([[0], y, [0]]), np.greater)[0]) > m:
            return False
        if x[1] - x[0] < xtol:
            return True
        x_new = (x[:-1]+x[1:])/2.0


def merge_into(z_new, z):
    if len(z) == 0:
        return z_new
    z_merged = np.zeros((2*len(z)-1,))
    z_merged[np.arange(0, len(z_merged), 2)] = z
    z_merged[np.arange(1, len(z_merged), 2)] = z_new
    return z_merged