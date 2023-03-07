from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
import numpy as np
import matplotlib.pyplot as plt


class Interpolator:

    # Method can be "linear" or "nearest"
    def __init__(self, method='linear'):
        self.interp = None
        self.method = method

    # Wt, xt, r are 1D list
    def train(self, Wt, xt, r):
        self.Wt = Wt
        self.xt = xt
        self.w_min = min(Wt)
        self.w_max = max(Wt)
        self.x_min = min(xt)
        self.x_max = max(xt)
        self.segment = (self.x_max - self.x_min) / 200
        if self.method == 'linear':
            self.interp = LinearNDInterpolator(list(zip(Wt, xt)), r, fill_value=min(r))
        if self.method == 'nearest':
            self.interp = NearestNDInterpolator(list(zip(Wt, xt)), r, fill_value=min(r))

    # Wt, xt can be scalar or list
    def estimate(self, Wt, xt):
        if self.interp is None:
            raise Exception('Interpolator not yet trained')
        else:
            return self.interp(Wt, xt)
        
    # Input Wt to get x that maximize reward and the corresponding reward
    def find_max(self, Wt):
        r_max = float('-inf')
        x_max = 0
        for i in range(200):
            x = self.x_min + i * self.segment
            r = self.estimate(Wt, x)
            if r > r_max:
                r_max = r
                x_max = x
        return x_max, r_max
        
    def plot(self):
        if self.interp is None:
            raise Exception('Interpolator not yet trained')
        else:
            W = np.linspace(self.w_min, self.w_max)
            X = np.linspace(self.x_min, self.x_max)
            W, X = np.meshgrid(W, X)
            R = self.interp(W, X)
            plt.pcolormesh(W, X, R, shading='auto')
            plt.plot(self.Wt, self.xt, "ok", label='input point')
            plt.legend()
            plt.colorbar()
            plt.axis("equal")
            plt.show()


if __name__ == '__main__':
    rng = np.random.default_rng()
    w = rng.random(20) - 0.5
    x = rng.random(20) - 0.5
    r = 1 - np.hypot(w, x)

    interpo = Interpolator('linear')
    interpo.train(w, x, r)
    interpo.plot()
    print(interpo.find_max(0))