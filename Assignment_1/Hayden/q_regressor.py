from sklearn.neighbors import KNeighborsRegressor
import numpy as np
import matplotlib.pyplot as plt


class QRegressor:

    def __init__(self, k=5, weights='distance'):
        self.model = KNeighborsRegressor(k, weights)

    # Wt, xt, r are 1D list
    def train(self, W, x, r):
        self.w_min = min(W)
        self.w_max = max(W)
        self.x_min = min(x)
        self.x_max = max(x)
        self.w_std = np.std(W)
        self.x_std = np.std(x)
        self.W = np.array(W) / self.w_std
        self.x = np.array(x) / self.x_std
        self.segment = (self.x_max - self.x_min) / 200
        self.model.fit(np.array([W, x]).transpose(), r)


    def estimate(self, Wt, xt):
        Wt = np.array(Wt) / self.w_std
        xt = np.array(xt) / self.x_std
        return self.model.predict(np.array([Wt, xt]).transpose())
        
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

    interpo = QRegressor()
    interpo.train(w, x, r)
    interpo.plot()
    # print(interpo.find_max(0))