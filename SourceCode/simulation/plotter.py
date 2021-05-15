# Standard libraries
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

# Local libraries
from .templates import templates as tmp


class Data:
    def __init__(self, data):
        """

        Args:
            data (dict):
        """
        self.data = data
        self.t = data['t']
        self.x = data['x']
        self.y = data['y']
        self.u = data['u']

        # Check dimensionality
        if self.x is not None:
            if self.x.ndim > 2:
                self.nx = np.size(self.x, 2)  # 3D
            else:
                self.nx = np.size(self.x, 1)  # 2D

        if self.y is not None:
            if self.y.ndim > 2:
                self.ny = np.size(self.y, 2)
            else:
                self.ny = np.size(self.y, 1)

        if self.u is not None:
            if self.u.ndim > 2:
                self.nu = np.size(self.u, 2)
            else:
                self.nu = np.size(self.u, 1)

        # Check existence of some keys values which may not be always present.
        if self.check_key(data, "method_key") is True:
            self.default_legend = data['method_key']
            self.n_method = len(data['method_key'])
        else:
            self.default_legend = None

        if self.check_key(data, 'true_solution') is True:
            self.x_true = data['true_solution']
        else:
            self.x_true = None

        if self.check_key(data, 'method_benchmark') is True:
            self.time_method = data['method_benchmark']
        else:
            self.time_method = None

    @staticmethod
    def check_key(data, search_key):
        for key in data.keys():
            if search_key == key:
                return True

    def rmse_error(self):
        """Evaluates precision of numeric methods against the true solution if exists.

        """
        if self.x_true is None:
            raise ValueError("True solution does not exist, therefore no comparison available")
        else:
            error = np.zeros(shape=(self.n_method, 2))
            for i in range(self.n_method):
                for j in range(self.nx):
                    error[i, j] = np.sqrt(1 / len(self.x[i, :, j]) * (sum((self.x_true[:, j] - self.x[i, :, j]) ** 2)))

            for i in range(len(error)):
                print('\nMethod No.{} time computation: {}'.format(i + 1, self.time_method[i]))
                for j in range(len(error[i])):
                    print('Equation No.{} error: {}'.format(j, error[i][j]))

    def plot_abs_error(self, color, save_fig=None, show=True):
        error = (np.subtract(self.x_true, self.x))

        ################################################# abs error ################################################
        fig1, axes1 = plt.subplots(nrows=1, ncols=1, figsize=(tmp.set_size()))
        legend = self.default_legend
        if self.n_method > 1:
            for i in range(self.n_method):
                for j in range(self.nx):
                    if j % 2 == 0:
                        axes1.plot(self.t, error[i, :, j], color[i][j], label=legend[i])
                    else:
                        axes1.plot(self.t, error[i, :, j], color[i][j], label="__nolegend__")
        else:
            for j in range(self.nx):
                if j % 2 == 0:
                    axes1.plot(self.t, error[:, j], color[0][j], label=legend[0])
                else:
                    axes1.plot(self.t, error[:, j], color[0][j], label="__nolegend__")


        y_label = r"$Error$"
        axes1.set(ylabel=y_label)
        AxisLabelOffset(axes1, label=y_label, ax="y")
        axes1.set(xlabel=r"$t$ $[s]$")
        axes1.grid(True)
        axes1.set_xlim(0, self.t[-1])
        # fig1.legend(loc='upper center', fancybox=None, ncol=5, borderaxespad=0.1, edgecolor='black', fontsize=8,
        #             bbox_to_anchor=(0.5, 1), borderpad=0.3)
        if save_fig is not None:
            fig1.savefig('{}.eps'.format(save_fig), format="eps")
        if show is not False:
            plt.show()

    def save_data(self, path=None):
        t = list(self.t)
        u = list(self.u[:, 0])
        x1 = list(self.x[:, 0])
        x2 = list(self.x[:, 1])
        y = list(self.y[:, 0])
        if path is None:
            raise NotADirectoryError("Entered path not found.")
        else:
            df = pd.DataFrame({'t': t, 'x1': x1, 'x2': x2, 'y': y, 'u': u})
            df.to_csv(path, index=False)

    @staticmethod
    def load_data(path=None):
        if path is None:
            raise FileNotFoundError("Data file not found.")
        else:
            read_data = pd.read_csv(path)
            structure = {}
            # Fill the dictionary with data vectors
            for i in range(len(read_data.columns)):
                structure[read_data.columns[i]] = read_data[read_data.columns[i]].values
            return structure


class Graph(Data):
    def __init__(self, data):
        super().__init__(data)

    def draw_system_figures(self, color, legend=None,
                            true_solution=True,
                            intersection=True,
                            save_fig=None,
                            show=True,
                            save_csv=None):
        """ Interpretation of the simulation data in separate figure for each x,y,u or optional true solution.

        Args:
            save_csv:
            color (list): Color set-up for lines.
            legend (list, optional):
            true_solution (bool, optional): Shows with x lines in the figure.
            intersection (bool, optional): Emphasize the point of equivalence in the figure.
            save_fig (bool, optional): Save the figures in .eps format.
            show (bool, optional): Show the figures on the display screen.

         Notes:
            [
               [method1-->[eq1-->[..sample0.,..sample1.,..sample2.,..sample4.]
                          [eq2-->[..sample0.,..sample1.,..sample2.,..sample4.]],
               [method2-->[eq1-->[..sample0.,..sample1.,..sample2.,..sample4.]
                          [eq2-->[..sample0.,..sample1.,..sample2.,..sample4.]],
               [method3-->[eq1-->[..sample0.,..sample1.,..sample2.,..sample4.]
                          [eq2-->[..sample0.,..sample1.,..sample2.,..sample4.]],
            ]
        """

        if legend is None:
            legend = self.default_legend
        # TODO solve this nastiness
        self.u = np.squeeze(self.u)
        ################################################# x ########################################################
        fig1, axes1 = plt.subplots(nrows=1, ncols=1, figsize=(tmp.set_size_long()))
        if self.x is not None:
            if self.n_method > 1:
                for i in range(self.n_method):
                    for j in range(self.nx):
                        if j % 2 == 0:
                            axes1.plot(self.t, self.x[i, :, j], color[i][j], label=legend[i])
                        else:
                            axes1.plot(self.t, self.x[i, :, j], color[i][j], label="__no_legend__")
            else:
                for j in range(self.nx):
                    if j % 2 == 0:
                        axes1.step(self.t, self.x[:, j], color[0][j], label="$x_1$")
                    else:
                        axes1.step(self.t, self.x[:, j], color[0][j],  label="$x_2$")

            y_label = r"$c$ $[mol.mL^{-1}]$"
            axes1.set(ylabel=y_label)
            AxisLabelOffset(axes1, label=y_label, ax="y")  # Special Case
            axes1.set(xlabel=r"$t$ $[s]$")
            axes1.grid(True)
            axes1.set_xlim(0, self.t[-1])
            # Additional plotting
            if true_solution is not False:
                axes1.plot(self.t, self.x_true[:, 0], 'y-', label="True")
                axes1.plot(self.t, self.x_true[:, 1], 'y--')

        ################################################### y ########################################################
        fig2, axes2 = plt.subplots(nrows=1, ncols=1, figsize=(tmp.set_size_long()))
        if self.y is not None:
            if self.n_method > 1:
                for i in range(self.n_method):
                    for j in range(self.ny):
                        if self.data["r"] is not None:
                            axes2.plot(self.t, self.data["r"][i, :, j], "k-", where='post')
                        axes2.step(self.t, self.y[i, :, j], "y-", where='post')
            else:
                for i in range(self.ny):
                    if self.data["r"] is not None:
                        axes2.step(self.t, self.data["r"][:, i], "k-", where='post')
                    axes2.step(self.t, self.y[:, i], "b-", where='post')

            axes2.set(ylabel=r"$pH$ $[-]$")
            axes2.set(xlabel=r"$t$ $[s]$")
            axes2.grid(True)
            axes2.set_xlim(0, self.t[-1])
        ################################################### u #########################################################
        fig3, axes3 = plt.subplots(nrows=1, ncols=1, figsize=(tmp.set_size_long()))
        if self.u is not None:
            axes3.step(self.t, self.u, "b-", where='post')
            axes3.set(ylabel=r"$F_2$ $[mL.s^{-1}]$")
            axes3.set(xlabel=r"$t$ $[s]$")
            axes3.grid(True)
            axes3.set_xlim(0, self.t[-1])

        if intersection is not False:
            # Draw an intersection points after the sign change
            if self.n_method > 1:
                idx = self._intersection_search(self.x[0, :, 0], self.x[0, :, 1])  # Get intersection indices
                interX = [self.t[j] for j in idx]  # Search for the intersection points [x, y] by indices.
                interY = [self.x[0, j, 0] for j in idx]  # Arbitrarily choose y function
                interpH = [self.y[0, j, 0] for j in idx]
            else:
                idx = self._intersection_search(self.x[:, 0], self.x[:, 1])  # Get intersection indices
                interX = [self.t[j] for j in idx]  # Search for the intersection points [x, y] by indices.
                interY = [self.x[j, 0] for j in idx]  # Arbitrarily choose y function
                interpH = [self.y[j, 0] for j in idx]
            axes1.plot(interX, interY, "go", label="Equivalence point")
            axes2.plot(interX, interpH, "go")

        fig1.legend(loc='upper center', fancybox=None, ncol=5, borderaxespad=0.1, edgecolor='black', fontsize=8,
                    bbox_to_anchor=(0.5, 1), borderpad=0.3)

        if save_fig is not None:
            fig1.savefig('{}_x.eps'.format(save_fig), format="eps")
            fig2.savefig('{}_y.eps'.format(save_fig), format="svg")
            fig3.savefig('{}_u.eps'.format(save_fig), format="eps")
        if save_csv is not None:
            if self.x is not None and self.data["r"] is not None:
                data = np.hstack(
                    (self.t.reshape(-1, 1), self.u.reshape(-1, 1), self.x[:, 0].reshape(-1,1), self.x[:, 1].reshape(-1,1), self.y.reshape(-1, 1), self.data['r'].reshape(-1, 1)))
                df = pd.DataFrame(data, columns=['t', 'u', 'x1', 'x2', 'y', 'r'])
            elif self.data["r"] is not None:
                data = np.hstack(
                    (self.t.reshape(-1, 1), self.u.reshape(-1, 1), self.y.reshape(-1, 1), self.data['r'].reshape(-1, 1)))
                df = pd.DataFrame(data, columns=['t', 'u', 'y', 'r'])
            else:
                data = np.hstack(
                    (
                    self.t.reshape(-1, 1), self.u.reshape(-1, 1), self.y.reshape(-1, 1)))
                df = pd.DataFrame(data, columns=['t', 'u', 'y'])

            df.to_csv('{}.csv'.format(save_csv), index=False)
        if show is not False:
            plt.show()

    @staticmethod
    def _intersection_search(y1, y2):
        # Initialize because differentiation is only possible when first is diff known,
        # perform backward differentiation.
        diff = y1[0] - y2[1]
        diffArr = [diff]
        # Find points of intersection where the first step was already done.
        i = 1
        idx = list()
        while i < len(y1) - 1:
            diff = y1[i] - y2[i + 1]
            diffArr.append(diff)
            if np.sign(diffArr[i - 1]) != np.sign(diffArr[i]):
                idx.append(i)
            i += 1
        return idx

    @staticmethod
    def statistics_methods(stat_data):
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(tmp.set_size_long()))
        axes.boxplot(stat_data, labels=['Euler', 'Heun', 'RK-4th', 'RK-5th'])
        plt.show()


class AxisLabelOffset:
    def __init__(self, all_axes, label="", ax="y"):
        self.label = label
        self.axis = {"y": all_axes.yaxis, "x": all_axes.xaxis}[ax]
        all_axes.callbacks.connect(self.axis, self.update)  # Joins axis with update() method.
        all_axes.figure.canvas.draw()
        self.update()

    def update(self):
        scale_format = self.axis.get_major_formatter()
        self.axis.offsetText.set_visible(False)
        self.axis.set_label_text(self.label + " " + scale_format.get_offset())
