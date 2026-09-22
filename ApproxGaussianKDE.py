import numpy as np
from scipy.stats import norm


class ApproxGaussianKDE(object):
    """
        Approximate Gaussian kernel density estimate, using binned data. 
        For each bin, the data in the bin is replaced by its mean. 
        The bin width is chosen so that the maximal absolute error is 
        bounded by tol/(h*sqrt(2*pi). This is shown in the accompanying
        documentation "Approxmation of kernel density estimator".
        Apart from binning, the implementation mimics that of 
        KernelDensity from sklearn.neighbors.
    """

    def __init__(self, data, bandwidth, tol=1e-4):
        self.h = bandwidth
        self.kernel = lambda u: np.exp(-u**2/2.0)
        self.data_orig = data
        data = np.sort(data)
        bin_width = np.sqrt(2*tol)*self.h
        bins = np.arange(data[0]-bin_width/2.0, data[-1]+3*bin_width/2.0, bin_width)
        ind_bins = np.searchsorted(data, bins)
        bin_sizes = np.diff(ind_bins)
        empty = bin_sizes == 0
        self.weights = bin_sizes[~empty]
        #print "len(self.weights) = {}".format(len(self.weights))
        self.datah = np.array([np.mean(ary) for (ary, empty_) in zip(np.split(data, ind_bins[1:-1]), empty) if not empty_])
        self.datah /= self.h
        self._norm_factor = np.sqrt(2*np.pi)*self.h*sum(self.weights)

    def evaluate_prop(self, x):  # returns values proportional to kde.
        xh = np.array(x).reshape(-1)/self.h
        res = np.zeros(len(xh))
        if len(xh) > len(self.datah):  # loop over data
            for data_, weight in zip(self.datah, self.weights):
                res += weight*self.kernel(data_-xh)
        else:  # loop over x
            for i, x_ in enumerate(xh):
                res[i] = np.sum(self.weights*self.kernel(self.datah-x_))
        return res

    def evaluate(self, x):
        res = self.evaluate_prop(x)
        return res / self._norm_factor

    def score_samples(self, x):
        return np.log(self.evaluate(x))

    def distr(self, x):
        return np.mean(norm.cdf(x - self.data_orig, scale=self.h))