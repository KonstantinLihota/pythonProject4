from Model import Model
import matplotlib.pyplot as plt
from Generator import DataGenerating
from Plot_grafiks import Plot
import numpy as np
if __name__ == '__main__':
    #data = Model()

    gen = DataGenerating([-0.3,0.1], [0.3, -0.2], sigma=1, N=20000, v=10000)
    data = gen.generating_data()
    model = Model(h=50, data = data, order = 2)

    #print(model.Estimate_var(4,40,80))

    #v,c = model.V_slove(2, 90)
    #print(v, c)

    model.r_t(30, 30)

    model.Lambd()

    print(model.L)
    model.J()
    drow = Plot(model)
    #drow.time_step()
    #drow.sum_step()
    #drow.estimate_parametrs()
    #drow.data()
    #drow.change_detection()
    #drow.estimate_var()
    drow.plot()
    #plt.plot()


