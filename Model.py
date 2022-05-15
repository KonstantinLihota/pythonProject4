import numpy as np

import copy


class Model:

    def __init__(self, h, data, order=2):

        self.h = h
        self.order = order
        self.N = len(data)
        self.t = [0]
        self.r = []
        self.list_v = []
        self.L = []
        self.list_J = []
        self.X = data
        self.sigma = 1
        self.С_border = []
        self.indent = 0
        self.sigma_estimate_list = []


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
        #self._ = (A, v)
        return sum

    def _start_param(self, x0):
        list_v = []
        v_min = np.zeros((self.order, self.order))
        C_sum = 0
        if x0 < self.order:
            return [0] * self.order, 0, 0

        for i in range(x0 - self.order, x0):
            v = 1 / (self.sigma * (
                np.dot(np.array(self.X[i  + 2- self.order :i +2]), np.array(self.X[i+2 - self.order:i+2]).transpose())) ** (
                             1 / 2))
            # if v>1:
            #    v = 1
            v = 0

            list_v.append(v)

            A = np.array(self.X[i - self.order:i]).reshape(-1, self.order)
            v_min += list_v[-1] * np.dot(A.transpose(), A)
            C_sum += (list_v[-1] ** 2) * np.dot(A, A.transpose())[0][0]

        return list_v, v_min, C_sum

    def V_slove(self, k=0, v_interval=0, v_min=None, C_sum=0):
        if v_min is None:
            v_interval, v_min, C_sum = self._start_param(k)

        v_interval.append(0.5)

        C_, mat = self._C(k, v_interval[-1], v_min)
        Sum_ = self._Sum_right(k, v_interval[-1], C_sum)
        p = 2
        diff = Sum_ - C_ / (self.sigma ** 2)
        while (abs(diff) > 1e-6):

            p = p * 2

            if diff < 0:
                v_interval[-1] = v_interval[-1] + 1 / p
            else:
                v_interval[-1] = v_interval[-1] - 1 / p
            if 1 / p < 1e-10:
                break

            C_, mat = self._C(k, v_interval[-1], v_min)
            Sum_ = self._Sum_right(k, v_interval[-1], C_sum)

            diff = Sum_ - C_ / (self.sigma ** 2)

        v_min = mat
        C_sum = Sum_
        print(v_interval[-1])
        return v_interval, v_min, C_sum

    def r_t(self, MNK_size, D_size, var=None):

        t_last_index = self.order
        self.indent = MNK_size + D_size

        while (self.N - self.order > t_last_index + MNK_size + D_size + 2):

            k0 = t_last_index
            if var is None:
                self.sigma = self.Estimate_var(k0 + 2, k0 + MNK_size + 2, k0 + MNK_size + D_size + 2)
                self.sigma_estimate_list.append(self.sigma)
            else:
                self.sigma = var

            k0 = k0 + MNK_size + D_size
            k_last = k0 + 1
            v, S, C_sum = self.V_slove(k=k0)

            while min(np.linalg.eig(S)[0]) < self.h and k_last < self.N - self.order:
                v, S, C_sum = self.V_slove(k_last, v, S, C_sum)
                k_last = k_last + 1

                self.r.append(min(np.linalg.eig(S)[0]))

            self.list_v.append(v)
            self.С_border.append(S)
            #print(v[-1])
            t_last_index = k_last
            self.t.append(t_last_index - 1)

    def Lambd(self):

        for i in range(len(self.t) - 1):
            c = np.linalg.inv(self.С_border[i])
            S = np.zeros((self.order, 1))

            for j in range(len(self.list_v[i])):
                T = self.t[i] + j + self.indent
                A = np.array(self.X[T - self.order - 1:T - 1]).reshape(-1, self.order)
                S += A.transpose() * self.list_v[i][j] * self.X[T]

            self.L.append(np.dot(c, S))

    def J(self):
        for l in range(1, len(self.L)):
            k = (self.L[l] - self.L[l - 1])

            self.list_J.append(np.dot(k.transpose(), k))
