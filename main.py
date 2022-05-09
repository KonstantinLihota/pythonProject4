from Model import Model
from Generator import DataGenerating
if __name__ == '__main__':
    #data = Model()

    gen = DataGenerating([-0.5, -0.3], [0.3, 0.5], sigma=10, N=20000, v=10000)
    data = gen.generating_data()
    model = Model(h=10, data = data)

    model.Estimate_var(3,400,800)

    v,c = model.V_slove(2, 100)
    print(v, c)

    #model.r_t(40, 40)
    # print(data.Estimate_var(10, 100, 200))
    # print(data.MNK(10, 50))
    # print(data.Estimate_var(10, 300, 600))
    #data.Lambd()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
