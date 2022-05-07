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
        print([self.X[N0 - i:N1 - i] for i in range(self.order, 2 * self.order)])
        A = np.array([self.X[N0 - i:N1 - i] for i in range(self.order, 2 * self.order)])
        b = np.array(self.X[N0:N1]).reshape(-1, 1)

        return np.dot(
            np.dot(
                np.linalg.inv(
                    np.dot(A, A.transpose())), A), b).transpose()[0][::-1]

    def Estimate_var(self, N0, N1, N2):
        sum = 0
        mnk = self._MNK(N0, N1)
        print(mnk)
        for i in range(N1, N2 - 1):
            sum += (np.array(self.X[i + 1]) -
                    np.dot(mnk, np.array(self.X[i - self.order: i]).reshape(-1, 1))) ** 2

        var = (sum / (N2 - N1 - 2)) ** (1 / 2)
        return var

    def _C(self, x0, N, v):
        C_sum = np.zeros((self.order, self.order))

        for i in range(x0, N):
            A = np.array(self.X[i - self.order:i]).reshape(-1, self.order)

            C_sum = C_sum + v[i - x0] * np.dot(A.transpose(), A)

        return min(np.linalg.eig(C_sum)[0]), C_sum

    def _Sum_right(self, x0, N, v):
        sum = 0

        for i in range(x0, N):
            A = np.array(self.X[i - self.order:i]).reshape(-1, self.order)
            sum = sum + v[i - x0] ** 2 * np.dot(A, A.transpose())[0]

        return sum

    def V_slove(self, k, N, v):

        self.start_v()
        v_interval = v
        for i in range(k, N):
            v_interval.append(0.5)

            diff = self._Sum_right(k, i, v_interval) - self._C(k, i, v_interval)[0] / (self.sigma ** 2)
            p = 2

            while (abs(diff) > 0.0001):

                p = p * 2
                if diff < 0:
                    v_interval[-1] = v_interval[-1] + 1 / p
                else:
                    v_interval[-1] = v_interval[-1] - 1 / p
                if 1 / p < 1e-10:
                    break

                C_ = self._C(k, i, v_interval)[0]
                diff = self._Sum_right(k, i, v_interval) - C_ / (self.sigma ** 2)

        return v_interval, C_

    def _start_v(self):
        for i in range(self.order):
            self.list_v.append(1 / (self.sigma * (
                np.dot(np.array(self.X[i: i + self.order]), np.array(self.X[i: i + self.order]).transpose())) ** (
                                            1 / 2)))

    def r_t(self, MNK_size, D_size):
        v = []
        t_last_index = self.order

        while (self.N - self.order > t_last_index + 2):

            S = 0
            k0 = t_last_index + MNK_size + D_size
            k_last = t_last_index + MNK_size + D_size + 1
            if k0 + MNK_size + D_size < self.N:
                self.sigma = self.Estimate_var(k0, k0 + MNK_size, k0 + MNK_size + D_size)[0]

            print(self.sigma)

            while S < self.h and k_last < self.N - self.order:
                wghts = copy.copy(self.list_v)
                print(S, k0, k_last, len(v))
                v, S = self.V_slove(k0, k_last, wghts)

                k_last = k_last + 1
                self.r.append(S)

            self.list_v.append(copy.copy(v))

            self.r.append(S)

            t_last_index = k_last - 1
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
