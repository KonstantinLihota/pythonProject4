from Model import Model
import matplotlib.pyplot as plt
from Generator import DataGenerating
from Plot_grafiks import Plot
import numpy as np
import pandas as pd
if __name__ == '__main__':
    list_date = np.array([])
    for i in range(13, 22):
        data = pd.read_csv(f'{i + 1}-{i}.csv')
        data =  np.array(([float(i.replace(",", "")) for i in data['Close']]))[::-1]
        #plt.plot(data)

        list_date = np.concatenate((list_date,data))
    plt.plot(list_date)
    for i in range(len(list_date)-1):
        list_date[i] = list_date[i+1]- list_date[i]
    print(len(list_date))
    data = list(list_date[:-1])
    #gen = DataGenerating([-0.2,0.1], [0.3, -0.2], sigma=1, N=20000, v=10000)
    #data = gen.generating_data()
    model = Model(h=10, data = data, order = 2)

    #print(model.Estimate_var(4,40,80))

    #v,c = model.V_slove(2, 90)
    #print(v, c)

    model.r_t(20, 20)

    model.Lambd()

    #print(model.L)
    model.J(2)
    drow = Plot(model)
    #drow.time_step()
    #drow.sum_step()
    #drow.estimate_parametrs()
    #drow.data()
    #drow.change_detection()
    #drow.estimate_var()
    drow.plot()


