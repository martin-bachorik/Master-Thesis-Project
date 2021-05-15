# Standard libraries
from sklearn.metrics import mean_squared_error as mse
from sklearn.linear_model import LinearRegression as Lr
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# Local libraries
from SourceCode.simulation.templates import templates as tmp

pd.options.mode.chained_assignment = None


class ARX:
    input_dim = 2

    def __init__(self, p, n_diff=None, scale_factor=False):
        """

        Args:
            p: Parameter 'p' represents number of lags/shifts of model to achieve
            n_diff: Data set number of differentiations
        """
        self.p = p
        self.n_diff = n_diff

        self.parameters = None
        self.train_rmse = None
        self.test_rmse = None
        self.result = None
        
        self.avagy = None
        self.avagu = None

        if scale_factor is not False:
            self.sc = MinMaxScaler(feature_range=(-1, 1))
            # self.sc = StandardScaler()
        else:
            self.sc = None

    def train(self, data):
        """ Linear Ready-Made model: Auto-regressive model with exogenous input u(t)

        Args:
            data: Training set for ARX model

        Returns:
            tuple: Weights and biases for identified linear model
        """
        if self.p is None:
            raise ValueError("You need to set the number of regress lags.")

        data.index = data['t']
        data = data.drop(['t'], axis=1)

        # subtract mean
        self.avagy = np.mean(data['y'])
        self.avagu = np.mean(data['u'])

        data['u'] = (data['u'] - self.avagu)
        data['y'] = (data['y'] - self.avagy)

        dependent_vars = data[['u', 'y']]  # Slice/extract dataframe
        var_keys = [col for col in dependent_vars.columns]

        # Divide dependent vars to train and test data set
        ratio = int(data.index.shape[0] / 2 * 1.5)
        train_data = pd.DataFrame(dependent_vars.iloc[:ratio, :])
        test_data = pd.DataFrame(dependent_vars.iloc[ratio:, :])

        # KEYS
        if self.n_diff is not None:
            # Firstly, differentiate both data sets to get stationary data
            train_data['u'] = train_data['u'].diff(self.n_diff).values
            train_data['y'] = train_data['y'].diff(self.n_diff).values

        if self.sc is not None:
            train_data[['u', 'y']] = self.sc.fit_transform(train_data[['u', 'y']])

        """" Secondly, lag data by p order for each model parameter.
        Lagged cols are appended from the last model parameter col."""
        for k in range(0, len(var_keys)):  # col index
            for i in range(1, self.p + 1):  # No. shifts
                train_data['{}: Lag {}'.format(var_keys[k], i)] = train_data[var_keys[k]].shift(i)
                test_data['{}: Lag {}'.format(var_keys[k], i)] = test_data[var_keys[k]].shift(i)

        # Remove "nan" garbage
        train_data = train_data.dropna()
        test_data = test_data.dropna()

        # TRAIN DATA
        x_train = train_data.iloc[:, self.input_dim:].values  # take only lagged values!
        y_train_target = train_data['y'].values

        # TEST DATA
        x_test = test_data.iloc[:, self.input_dim:].values
        y_test_target = test_data['y'].values

        # Optimize linear regression parameters
        lr = Lr(fit_intercept=False, normalize=False)
        lr.fit(x_train, y_train_target)  # no intercept
        self.parameters = {'weights': lr.coef_, 'bias': lr.intercept_}

        # Predict and save into the table for root mean squared error
        y_train_predict = x_train.dot(self.parameters['weights']) + self.parameters['bias']
        y_test_predict = x_test.dot(self.parameters['weights']) + self.parameters['bias']

        self.train_rmse = np.sqrt(mse(y_train_target, y_train_predict))
        self.test_rmse = np.sqrt(mse(y_test_target, y_test_predict))

    def test_online(self, data, save=False, save_csv= None):
        if self.parameters is None:
            raise ValueError("ARX model parameters are missing.")

        orig_data = data.copy()  # keep the original data frame
        t = data['t'].values

        data.index = data['t']
        data = data.drop(['t'], axis=1)
        data = data[['u', 'y']]  # Slice/extract dataframe

        # subtract mean
        means = {'u': np.mean(data['u']), 'y': np.mean(data['y'])}
        data['u'] = data['u'] - self.avagu
        data['y'] = data['y'] - self.avagy

        var_keys = [col for col in data.columns]

        if self.n_diff is not None:
            rolling_index = (self.p - 1) + self.n_diff
            y_diff_key = data['y'].values[rolling_index]  # KEY
            # Differentiate by the first order
            data['u'] = data['u'].diff(self.n_diff).values
            data['y'] = data['y'].diff(self.n_diff).values

        if self.sc is not None:
            data[['u', 'y']] = self.sc.transform(data[['u', 'y']])

        for k in range(0, len(var_keys)):
            for i in range(1, self.p + 1):
                data['{}: Lag {}'.format(var_keys[k], i)] = data[var_keys[k]].shift(i)

        # append u and y before dropout
        u = data['u'].values.reshape(-1, 1)
        y_target = data['y'].values

        data = data.dropna()  # remove garbage

        # Collect data
        x0 = data.iloc[:, self.input_dim:].values[0, :]
        # Extract and reverse y back start with p_lag at index 0
        y0 = x0[self.p::]
        y0 = y0[::-1]

        y = np.zeros(shape=(len(t), 1))  # Create new vector for online predictions

        y[:self.p, 0] = y0  # append initial y values

        for i in range(len(t) - self.p):
            if i == 0:
                x_in = x0
            else:
                x_in = np.concatenate((u[i:(i + self.p), :][::-1], y[i:(i + self.p), :][::-1]),  # reverse the sequence because of the way it was trained on
                                      axis=0).T
            yi = x_in.dot(self.parameters['weights']) + self.parameters['bias']
            y[(i + self.p), :] = yi


        if self.n_diff is not None:
            y = np.hstack((y_diff_key, y)).cumsum()  # Convert back with cumulative summation with diff keys
        if self.sc is not None:
            # rescale back
            y = self.sc.inverse_transform(
                np.hstack((u.reshape(-1, 1), y.reshape(-1, 1))))[:, -1]  # extract only predicted y

        y = y + self.avagy  # add mean back

        self.test_rmse = np.sqrt(mse(y, orig_data['y']))

        fig1, axes = plt.subplots(nrows=1, ncols=1, figsize=tmp.set_size_long())
        axes.step(t, orig_data['y'].values, 'g-')
        axes.step(t, y, 'b-')
        axes.set(ylabel="$pH$ $[-]$")
        axes.set(xlabel="$t$ $[s]$")
        axes.grid(True)
        axes.set_xlim(0, t[-1])
        fig1.legend(['Raw Data', 'Predicted Data'],
                    loc='upper center', fancybox=None, ncol=6, borderaxespad=0.1, edgecolor='black', fontsize=7,
                    bbox_to_anchor=(0.5, 1), borderpad=0.3)

        # Control input
        fig2, axes = plt.subplots(nrows=1, ncols=1, figsize=tmp.set_size_long())
        axes.step(t, orig_data['u'].values, 'b-', where='post')
        axes.set(ylabel="$F_{2}$ $[mL.s^{-1}]$")
        axes.set(xlabel="$t$ $[s]$")
        axes.grid(True)
        axes.set_xlim(0, t[-1])

        if save is True:
            fig1.savefig('ARX_y.eps', format="eps")
            fig2.savefig('ARX_u.eps', format="eps")

        if save_csv is not None:
            data = np.hstack(
                (t.reshape(-1, 1), orig_data['u'].values.reshape(-1, 1),
                 y.reshape(-1, 1)))
            df = pd.DataFrame(data, columns=['t', 'u', 'y'])
            df.to_csv('{}.csv'.format(save_csv), index=False)

    def test(self, data, save=False, save_csv=False):
        if self.parameters is None:
            raise ValueError("ARX model parameters are missing.")

        orig_data = data.copy()  # keep the original data frame
        t = orig_data.index.values

        data.index = data['t']
        data = data.drop(['t'], axis=1)
        data = data[['u', 'y']]  # Slice/extract dataframe

        # subtract mean
        means = {'u': np.mean(data['u']), 'y': np.mean(data['y'])}
        data['u'] = data['u'] - means['u']
        data['y'] = data['y'] - means['y']

        var_keys = [col for col in data.columns]

        if self.n_diff is not None:
            rolling_index = (self.p - 1) + self.n_diff
            y_diff_key = data['y'].values[rolling_index]  # KEY
            u_diff_key = data['u'].values[rolling_index]  # KEY
            # Differentiate by the first order
            data['u'] = data['u'].diff(self.n_diff).values
            data['y'] = data['y'].diff(self.n_diff).values

        if self.sc is not None:
            data[['u', 'y']] = self.sc.transform(data[['u', 'y']])

        for k in range(0, len(var_keys)):
            for i in range(1, self.p + 1):
                data['{}: Lag {}'.format(var_keys[k], i)] = data[var_keys[k]].shift(i)

        y_lags = data['y'].values[:self.p]
        u_lags = data['u'].values[:self.p]

        data = data.dropna()
        x = data.iloc[:, self.input_dim:].values  # lagged input vector

        u = data['u'].values
        y_target = data['y'].values

        y = x.dot(self.parameters['weights']) + self.parameters['bias']  # Predict
        self.test_rmse = np.sqrt(mse(y, y_target))  # Measure rmse in diff form

        y = np.concatenate((y_lags, y))  # concatenate with the initial sequence
        u = np.concatenate((u_lags, u))  # concatenate with the initial sequence

        if self.sc is not None:
            y = self.sc.inverse_transform(
                np.hstack((u.reshape(-1, 1), y.reshape(-1, 1))))[:, -1]  # Rescale back

        if self.n_diff is not None:
            y = np.hstack((y_diff_key, y)).cumsum()  # Convert back with cumulative sum with diff keys
            # u = np.hstack((u_diff_key, u)).cumsum()

        y = y + means['y']  # add mean back

        # Plot
        fig1, axes = plt.subplots(nrows=1, ncols=1, figsize=tmp.set_size_long())
        axes.step(t, orig_data['y'].values, 'g-')
        axes.step(t, y, 'b-')
        axes.set(ylabel="$pH$ $[-]$")
        axes.set(xlabel="$t$ $[s]$")
        axes.grid(True)
        axes.set_xlim(0, t[-1])
        fig1.legend(['Raw Data', 'Predicted Train Data', 'Predicted Test Data', 'Threshold'],
                    loc='upper center', fancybox=None, ncol=6, borderaxespad=0.1, edgecolor='black', fontsize=7,
                    bbox_to_anchor=(0.5, 1), borderpad=0.3)

        # Control input
        fig2, axes = plt.subplots(nrows=1, ncols=1, figsize=tmp.set_size_long())
        axes.step(t, orig_data['u'].values, 'b-', where='post')
        axes.set(ylabel="$F_{2}$ $[mL.s^{-1}]$")
        axes.set(xlabel="$t$ $[s]$")
        axes.grid(True)
        axes.set_xlim(0, t[-1])

        if save is True:
            fig1.savefig('ARX_y.eps', format="eps")
            fig2.savefig('ARX_u.eps', format="eps")

        if save_csv is not None:
            data = np.hstack(
                (t.reshape(-1, 1), orig_data['u'].values.reshape(-1, 1),
                 y.reshape(-1, 1)))
            df = pd.DataFrame(data, columns=['t', 'u', 'y'])
            df.to_csv('{}.csv'.format(save_csv), index=False)
