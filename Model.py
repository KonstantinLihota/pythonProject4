import numpy as np
import matplotlib.pyplot as plt
import copy


class Model:

    def __init__(self, h, data, order=2):

        self.h = h
        self.order = order
        self.N = len(data)
        self.t = []
        self.r = []
        self.list_v = []
        self.L = []
        self.list_J = []
        self.X = data
        self.sigma = 1

    def _MNK(self, N0, N1):

        A = np.array([self.X[N0 - i:N1 - i] for i in range(self.order, 2 * self.order)])
        b = np.array(self.X[N0:N1]).reshape(-1, 1)

        return np.dot(
            np.dot(
                np.linalg.inv(
                    np.dot(A, A.transpose())), A), b).transpose()[0][::-1]

    def Estimate_var(self, N0, N1, N2):
        sum = 0
        mnk = self._MNK(N0, N1)

        for i in range(N1, N2 - 1):
            sum += (np.array(self.X[i + 1]) -
                    np.dot(mnk, np.array(self.X[i - self.order: i]).reshape(-1, 1))) ** 2

        var = (sum / (N2 - N1 - 2)) ** (1 / 2)
        return var[0]

    def _C(self, i, v, C_sum):

        A = np.array(self.X[i - self.order:i]).reshape(-1, self.order)

        C_sum = C_sum + v * np.dot(A.transpose(), A)

        return min(np.linalg.eig(C_sum)[0]), C_sum

    def _Sum_right(self, i, v, sum):

        A = np.array(self.X[i - self.order:i]).reshape(-1, self.order)

        sum = sum + (v ** 2) * np.dot(A, A.transpose())[0]

        return sum

    def _start_param(self, x0):
        list_v = []
        v_min = np.zeros((self.order, self.order))
        C_sum = 0
        for i in range(x0, self.order + x0):
            v = 1 / (self.sigma * (np.dot(np.array(self.X[i: i + self.order]), np.array(self.X[i: i + self.order]).transpose())) ** (1 / 2))
            if v>1:
                v=0

            list_v.append(v)

            A = np.array(self.X[i - self.order:i]).reshape(-1, self.order)
            v_min += list_v[-1] * np.dot(A.transpose(), A)
            C_sum += (list_v[-1] ** 2) * np.dot(A, A.transpose())[0][0]

        return list_v, v_min, C_sum

    def V_slove(self, k, N):

        v_interval, v_min, C_sum = self._start_param(k)

        for i in range(k + self.order, N):
            v_interval.append(0.5)

            diff = self._Sum_right(i, v_interval[-1], C_sum) - self._C(i, v_interval[-1], v_min)[0] / (self.sigma ** 2)
            p = 2

            min = [100000000000, 0]
            while (abs(diff) > 0.0001):

                p = p * 2

                if diff < 0:
                    v_interval[-1] = v_interval[-1] + 1 / p
                else:
                    v_interval[-1] = v_interval[-1] - 1 / p
                if 1 / p < 1e-10:
                    # v_interval[-1] = min[1]
                    break

                C_, mat = self._C(i, v_interval[-1], v_min)
                Sum_ = self._Sum_right(i, v_interval[-1], C_sum)
                #print(Sum_ , C_ / (self.sigma ** 2), v_interval[-1])
                diff = Sum_ - C_ / (self.sigma ** 2)
                # if diff<min[0]:
                #   min[0] = diff
                #  min[1] = copy.copy(v_interval[-1])

            v_min = mat
            C_sum = Sum_

        return v_interval, v_min

    def r_t(self, MNK_size, D_size, var = None):
        v = []
        t_last_index = self.order

        while (self.N - self.order > t_last_index+ MNK_size + D_size+ 2):

            k0 = t_last_index
            if var is None:
                self.sigma = self.Estimate_var(k0+ 2, k0 + MNK_size+ 2, k0 + MNK_size + D_size+ 2)
            else:
                self.sigma = var

            k0 = k0 + MNK_size + D_size
            k_last = k0 + 1
            S = 0

            while S < self.h and k_last < self.N - self.order:

                v, S = self.V_slove(k0, k_last)
                #print(k0, k_last, self.V_slove(82, 90))
                k_last = k_last + 1
                S = min(np.linalg.eig(S)[0])
                #print(S,v)
                self.r.append(S)

            self.list_v.append(v)
            self.r.append(S)

            t_last_index = k_last + 1
            print(t_last_index)
            self.t.append(t_last_index)

    def Lambd(self):

        for i in range(self.order - 2):
            _, cc = self.C(self.t[i], self.t[i + 1], self.list_v)

            c = np.linalg.inv(cc)

            S = np.zeros((self.order, 1))

            for j in range(self.t[i] - self.order, self.t[i + 1]):
                A = np.array(self.X[j - self.order:j]).reshape(-1, self.order)
                S += A.transpose() * self.list_v[j] * self.X[j + 1]

            self.L.append(np.dot(c, S))

    def J(self):
        for l in range(1, len(self.L)):
            k = (self.L[l] - self.L[l - 1])

            self.list_J.append(np.dot(k.transpose(), k))
