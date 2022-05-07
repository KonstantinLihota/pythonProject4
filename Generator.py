import numpy as np


class DataGenerating:

    def __init__(self, B1, B2, sigma, N, v=1000):

        self.q1 = np.array(B1)
        self.q2 = np.array(B2)
        self.sigma = sigma
        self.v = v
        self.N = N
        self.X = [1] * len(self.q2)

    def generating_data(self):

        e = np.random.normal(0, 1, size=self.N)
        self.X.append(np.dot(self.q1, self.X) + self.sigma * e[0])

        for i in range(len(self.q1), self.v):
            X_1 = np.dot(self.q1, self.X[i - len(self.q1):i]) + self.sigma * e[i - len(self.q1)]

            self.X.append(X_1)

        for i in range(self.N - self.v):
            X_1 = np.dot(self.q2, self.X[self.v + i - len(self.q2):i + self.v]) + self.sigma * e[
                self.v + i - len(self.q2)]
            self.X.append(X_1)
        return self.X[len(self.q1):]

'''
    if plot:
        plt.figure(figsize=(10, 7))
        plt.title('Data')
        plt.plot(self.X)
    return self.X
'''
