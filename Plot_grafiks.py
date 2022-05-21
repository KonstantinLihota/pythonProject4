
import matplotlib.pyplot as plt
import numpy as np
class Plot():
    def __init__(self, model):
        self.model  = model


    def time_step(self):
        plt.figure(figsize=(20,5))
        plt.scatter(self.model.t, np.zeros(len(self.model.t)))
        plt.title("Отрезки времени")
    def sum_step(self):
        plt.figure(figsize=(20,10))
        plt.step(np.array(range(len(self.model.r))),self.model.r)
        plt.title("лесенка сумм")

    def estimate_parametrs(self):
        for i in range(self.model.order):
            plt.figure(figsize=(20, 5))
            plt.plot(self.model.t[:-1],np.array(self.model.L).reshape(-1,self.model.order)[:,i])
            plt.title("Оценки параметров")

    def data(self, diff = True):
            plt.figure(figsize=(20, 5))
            if diff:
                plt.plot(list(range(9000,11000)),self.model.X[9000:11000])
            else:
                plt.plot(self.model.X)
            plt.title("Данные")
    def change_detection(self):
            plt.figure(figsize=(20, 5))
            plt.plot(self.model.t[1:-self.model.offset],np.array(self.model.list_J).reshape(-1,1))
            plt.title("Точки разладки")
    def estimate_var(self):
            plt.figure(figsize=(20, 5))
            plt.plot(np.array(self.model.sigma_estimate_list).reshape(-1,1))
            plt.title("Оценка дисперсии")
    def plot(self):
        self.data()
        self.time_step()
        self.sum_step()
        self.estimate_var()
        self.estimate_parametrs()
        self.change_detection()




