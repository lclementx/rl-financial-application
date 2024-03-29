from sklearn.neighbors import KNeighborsRegressor
import numpy as np
import matplotlib.pyplot as plt

'''
This is a support file that that contain a single class QRegressor which use KNN regression to estimate the Q value
'''

class QRegressor:

    # Larger k to smooth out the large variance from monte carlo
    def __init__(self, k=50, weights='distance'):
        self.model = KNeighborsRegressor(n_neighbors=k, weights=weights)

    # Input Wt, xt, r to train the model, the three input variables should be list of same length
    def train(self, W, x, r):
        self.w_min = min(W)
        self.w_max = max(W)
        self.x_min = min(x)
        self.x_max = max(x)
        self.W = W
        self.x = x
        self.segment = (self.x_max - self.x_min) / 100
        self.model.fit(np.array([W, x]).transpose(), r)

    # Estimate the Q value given Wt, xt as input and they should be of the same shape
    def estimate(self, Wt, xt):
        pred = self.model.predict(np.array([Wt, xt]).reshape(2, -1).transpose())
        if len(pred) == 1:
            return pred[0]
        else:
            return pred
        
    # Input Wt to get the x that maximize reward and the corresponding reward
    def find_max(self, Wt):
        r_max = float('-inf')
        x_max = 0
        for i in range(100):
            x = self.x_min + i * self.segment
            r = self.estimate(Wt, x)
            if r > r_max:
                r_max = r
                x_max = x
        return x_max, r_max
    
    # Plot the learned Q function
    def plot(self):
        W = np.linspace(self.w_min, self.w_max)
        X = np.linspace(self.x_min, self.x_max)
        W, X = np.meshgrid(W, X)
        # R = 1- np.hypot(W, X)
        R = self.estimate(W.reshape(-1), X.reshape(-1))
        R = R.reshape(50, 50)
        plt.pcolormesh(W, X, R, shading='auto')
        plt.plot(self.W, self.x, "ok", label='input point')
        plt.legend()
        plt.colorbar()
        plt.axis("equal")
        plt.show()

    # Plot the learned Q function sliced on a particular W
    def plotSlice(self, w):
        X = np.linspace(self.x_min, self.x_max)
        W = [w for _ in range(len(X))]
        R = self.estimate(W, X)
        plt.plot(X, R)
        plt.show()


# Learn and plot a function for testing
if __name__ == '__main__':
    rng = np.random.default_rng()
    w = rng.random(300) - 0.5
    x = rng.random(300) - 0.5
    r = 1 - np.hypot(w, x)

    model = QRegressor()
    model.train(w, x, r)
    model.plot()
    print(model.find_max(0))