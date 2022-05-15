
import matplotlib.pyplot as plt
import numpy as np
class Plot():
    def __init__(self, model):
        self.model  = model


    def time_step(self):
        plt.figure(figsize=(20,5))
        plt.scatter(self.model.t, np.zeros(len(self.model.t)))
    def sum_step(self):
        plt.figure(figsize=(20,10))
        plt.step(np.array(range(len(self.model.r))),self.model.r)

    def estimate_parametrs(self):
        for i in range(self.model.order):
            plt.figure(figsize=(20, 5))
            plt.plot(self.model.t[:-1],np.array(self.model.L).reshape(-1,2)[:,i])

    def data(self):
            plt.figure(figsize=(20, 5))
            plt.plot(self.model.X)
    def change_detection(self):
            plt.figure(figsize=(20, 5))
            plt.plot(self.model.t[1:-1],np.array(self.model.list_J).reshape(-1,1))
    def estimate_var(self):
            plt.figure(figsize=(20, 5))
            plt.plot(np.array(self.model.sigma_estimate_list).reshape(-1,1))
    def plot(self):
        self.data()
        self.time_step()
        self.sum_step()
        self.estimate_var()
        self.change_detection()




