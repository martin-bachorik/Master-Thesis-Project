# Standard libraries
import control.matlab
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# Local libraries
from SourceCode.simulation.templates import templates as tmp
from SourceCode.simulation.plotter import Data


class NormalStep:
    def __init__(self, t=None, u=None, y=None):
        self.t = t
        self.u = u
        self.y = y
        self.ys = None

        # Normalize instance variables by calling Normalizing Method
        self.normalStepResponse()

    def normalStepResponse(self):
        # The beauty is, we don't have to know current steady state value of the system, all we need to do is set the system
        # to some initial u_init and leave enough time to reach its steady state, therefore u_init becomes u^s. After, reaching
        # steady state, we perform step value u_fin = u^s + k
        idx = None
        for i in range(len(self.u) - 1):
            if self.u[i + 1] != self.u[
                i]:  # we find index at the step time, where all the values are in the steady state ("if the system is stable")
                idx = i + 1
                break
        # normalized BY (u[-1]-u[0])
        # y --> yn; t-->tn
        self.y = self.y[idx:]  # cut the data from the steady state point where the step response takes place
        self.ys = self.y[0]
        self.y = self.y - self.y[0]  # first data point of yn must be 0 .. because we need deviation form ... y[0] is SS
        self.y = self.y / (self.u[-1] - self.u[0])
        self.t = self.t[idx:] - self.t[idx]


class Strejc:
    @staticmethod
    def strejc(path=None):
        # load all the data 20x
        data = {}
        for i in range(20):
            data[i] = Data.load_data(path=path.format(i + 1))  # LOAD

        # NORMALIZE pH by F2 [mL/s] for all data
        normObjectX1 = {}
        for i in range(len(data)):
            normObjectX1[i] = NormalStep(t=data[i]['t'], u=data[i]['u'], y=data[i]['pH'])

        # store pH vector from all object to an array
        all_u = [normObjectX1[x].u for x in range(len(normObjectX1))]
        all_pH = [normObjectX1[x].y for x in range(len(normObjectX1))]
        all_us = [normObjectX1[x].u[0] for x in range(len(normObjectX1))]
        all_ys = [normObjectX1[x].ys for x in range(len(normObjectX1))]

        avag_us = sum(all_us[:]) / len(all_us)
        avag_ys = sum(all_ys[:]) / len(all_ys)
        avag_y = sum(all_pH[:]) / len(all_pH)

        tnx1 = normObjectX1[0].t

        # Strejc Method
        yn = avag_y
        tn = tnx1
        K = yn[-1]

        dy = np.diff(yn)
        dt = np.diff(tn)
        yd = dy / dt        # derivative
        max_yd = max(yd)  # find maximum value of derivative
        idx = [idx for idx, val in enumerate(yd) if val == max_yd]
        idx = idx[0]

        t1 = tn[idx]
        y1 = yn[idx]

        t2 = tn[idx + 1]
        y2 = yn[idx + 1]

        a = (y1 - y2) / (t1 - t2)
        b = y1 - a * t1

        tz = -b / a
        tk = (K - b) / a

        Do = 0        # delay D

        Tu = tz - Do
        Tn = tk - tz
        print("a: {}".format(a))
        print("b: {}".format(b))
        print("tz: {}".format(tz))
        print("tk: {}".format(tk))
        print("Tu: {}".format(Tu))
        print("Tn: {}".format(Tn))

        fs = Tu / Tn

        print("fs: {}".format(fs))

        n = 1
        fn0 = 0
        gn0 = 1

        T = Tn * gn0
        Dv = (fs - fn0) * Tn
        D = Do + Dv

        print("K: {}".format(K))
        print("T: {}".format(T))
        print("D: {}".format(D))

        s = control.tf('s')
        x = 1
        k = 1
        product = 1
        while x <= n:
            # STREJC
            product = product * (T * s + 1)
            k = k + 1
            x = x + 1
        print('\nDelay: {}'.format(D))
        G_Strejc = K / product
        structure = {'t': tnx1, 'allpH': all_pH, 'avpH': avag_y, 'all_u': all_u, 'G': G_Strejc,
                     'avag_ys': avag_ys, 'avag_us': avag_us}
        return structure

    @staticmethod
    def drawFigures(data, save='no', test_path=None):
        t = data['t']
        all_pH = data['allpH']
        avagpH = data['avpH']
        avag_ys = data['avag_ys']
        avag_us = data['avag_us']
        all_u = data['all_u']
        G = data['G']

        fig01, axes = plt.subplots(nrows=1, ncols=1, figsize=tmp.set_size())
        for i in range(len(all_pH)):
            axes.plot(t, all_pH[i])
        axes.set(ylabel=r"$pH$ $[-]$")
        axes.set(xlabel=r"$t$ $[s]$")
        axes.grid(True)
        axes.set_xlim(0, t[-1])

        str = list()
        for i in range(len(all_u)):
            str = str + ['{}--$>${}'.format(all_u[i][0], all_u[i][-1])]

        ############################################ Average ###########################################################
        fig02, axes = plt.subplots(nrows=1, ncols=1, figsize=tmp.set_size())
        axes.plot(t, avagpH)
        axes.set(ylabel=r"$pH$ $[-]$")
        axes.set(xlabel=r"$t$ $[s]$")
        axes.grid(True)
        axes.set_xlim(0, t[-1])
        
        # Compare
        fig00, axes = plt.subplots(nrows=1, ncols=1, figsize=tmp.set_size_long())
        T, yout = control.step_response(G, t)
        axes.plot(t, avagpH, "g-", )
        axes.plot(T, yout, "b-",)
        fig00.legend(['Raw Data', 'Transfer Function'],
                    loc='upper center', fancybox=None, ncol=6, borderaxespad=0.1, edgecolor='black', fontsize=8,
                    bbox_to_anchor=(0.5, 1), borderpad=0.3)
        axes.set(ylabel=r"$pH$ $[-]$")
        axes.set(xlabel=r"$t$ $[s]$")
        axes.grid(True)
        axes.set_xlim(0, T[-1])
        axes.set_ylim(0, 1.79)

        # if save == 'yes':
            # inpArr = np.hstack((T.reshape(-1, 1), yout.reshape(-1, 1)))
            # df = pd.DataFrame(inpArr, columns=['t', 'pH'])
            # df.to_csv('Strejc.csv', index=False)
            # fig00.savefig('Strejc_compare.eps', format="eps")
            # fig01.savefig('Strejc_train.eps', format="eps")
            # fig02.savefig('Strejc_average.eps', format="eps")
            # fig1.savefig('Strejc_test_25steps_y.eps', format="eps")
            # fig2.savefig('Strejc_test_25steps_u.eps', format="eps")
