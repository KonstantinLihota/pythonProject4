from Model import Model
from Generator import DataGenerating
if __name__ == '__main__':
    #data = Model()

    gen = DataGenerating([-0.1, 0.3], [0.3, 0.5], sigma=1, N=2000, v=1000)
    data = gen.generating_data()
    model = Model(h=15, data = data)

    print(model.Estimate_var(3,40,80))

    #v,c = model.V_slove(2, 90)
    #print(v, c)

    model.r_t(30, 30, 1)

    model.Lambd()
    print('---------')
    print(model.L)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
