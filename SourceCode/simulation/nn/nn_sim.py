# Standard libraries
import torch
import math
import numpy as np
from numpy import array
import matplotlib.pyplot as plt
import pandas as pd


# Local libraries
from .preprocessing import data_preparation, \
    split_recurrent_sequence, split_feedforward_sequence, x_size_recurrent, x_size_feedforward
from ..templates import templates as tmp

from .nn_models import FFNN, RNN, LSTMModel

__all__ = ['FFNN', 'RNN', 'LSTMModel', 'NNSimulator', 'NNLoader']


class NNLoader:
    def __init__(self, model_path=None, model_type=None):
        """ Load the model's dictionary in this way instead
            model_structure = torch.load(PATH)  # torch.load() serves for deserializing the structure's parameters
            # Load the model dimension settings
            input_dim = model_structure['input_dim']
            output_dim = model_structure['output_dim']
            seq_length = model_structure['seq_len']
            hidden_dim = model_structure['hidden_dim']
            layer_dim = model_structure['layer_dim']

            # Instantiate a new model class based on the saved dimension
            model = LSTMModel(input_dim, hidden_dim, layer_dim, output_dim)
            # Extract model weights and biases (just parameters)
            model_w_b = model_structure['model_state_dict']
            # Load the weights and biases into the model instance
            model = model.load_state_dict(model_w_b)
            # Call to set dropout and batch normalization layers to evaluation mode before running inference!
            model.eval()
            # - or - in case we want to continue training the model then just set it to the train model..
            model.train()

            See Also:
                # Additional keys
                whole_model = model_structure['whole_model']
                optimizer_w_b = model_structure['optimizer_state_dict']
                num_epochs = model_structure['epochs']

                load_state_dict(, , strict=True/False)
                - strict parameter defines whether the key names of NN should fit the new model or not
                - if false then omitted else pass

            Notes:
                You always have to instantiate model class somewhere otherwise it's not gonna work.
        """
        if model_path is None:
            raise NameError("You need to assign an address of your model.")

        # Load a model dimension settings into variable.
        model_structure = torch.load(model_path)  # torch.load() serves for deserializing structure's parameters

        self.input_dim = model_structure['input_dim']
        self.output_dim = model_structure['output_dim']
        self.seq_length = model_structure['seq_len']
        self.hidden_dim = model_structure['hidden_dim']
        self.layer_dim = model_structure['layer_dim']
        self.delta_t = model_structure['delta_t']
        self.sc = model_structure['scale_factor']
        self.out = None

        self.model_type = model_type
        if self.model_type == "LSTM" or self.model_type == "RNN":
            self.u_dim = self.input_dim - self.output_dim
            self.x_size = x_size_recurrent(self.seq_length, self.input_dim)
            if self.model_type == "LSTM":
                self.model = LSTMModel(self.input_dim, self.hidden_dim, self.layer_dim, self.output_dim)
            elif self.model_type == "RNN":
                self.model = RNN(self.input_dim, self.hidden_dim, self.layer_dim, self.output_dim)
        elif self.model_type == "FFNN":
            self.u_dim = int(self.input_dim / self.seq_length) - self.output_dim
            self.x_size = x_size_feedforward
            self.model = FFNN(self.input_dim, self.hidden_dim, self.output_dim)
        else:
            raise ValueError("Choose type of Neural Network again.")

        # Loading a model to pre-defined machine learning model structure.
        model_w_b = model_structure['model_state_dict']  # Extract model weights and biases (just parameters).
        self.model.load_state_dict(model_w_b)  # Load the weights and biases into the model instance.
        self.model.eval()  # Call to set dropout and batch normalization layers to eval mode before run.

    def online_open_loop(self, test_data_path=None):
        """""ONLINE-OPEN-LOOP NN model" prediction model (with known u(t) forward).

       Args:
            test_data_path: Path or a structure with the data to simulate.
        """
        test_data = data_preparation(test_data_path, delta_t=self.delta_t)
        "Initial conditions with dim[1 sample, seq_len, input_size)], transformed to 2D vectors."
        if self.model_type == "LSTM" or self.model_type == "RNN":
            X, y_out = split_recurrent_sequence(test_data['u_y'], self.seq_length)
            u = X[:, :, [0]]
            y0_in = X[0, :, -1].reshape(-1, 1)
        elif self.model_type == "FFNN":
            X, y_out = split_feedforward_sequence(test_data['u_y'], self.seq_length)  # Return the reversed sequence.
            u = X[:, :self.seq_length]
            y0_in = X[0, self.seq_length::].reshape(-1, 1)
            y0_in = y0_in[::-1]  # Reverse back the slice sequence.
        else:
            return "Empty"

        """Step is 1 too, so far we're doing only 1 step prediction mode.
        # We could also perform 5 step if we had 5 step prediction model. Also the stop-iter is shorter of seq_len."""
        y = np.zeros(shape=(len(test_data['t']), self.output_dim))
        # Add the initial condition sequence for "y", note that "u" is already known forward.
        y[:self.seq_length] = y0_in
        prediction_step = 1
        for i in range(0, len(test_data['t']) - self.seq_length, prediction_step):
            #  y(t) = y(t-n) + ... + y(t-2) + y(t-1), same applies for u.
            if self.model_type == "LSTM" or self.model_type == "RNN":
                X = np.concatenate((u[i, :, :], y[i:(i + self.seq_length), :]), axis=1)
                X = torch.tensor(self.sc.transform(X), dtype=torch.float32)
                yi = self.model(X.view(self.x_size)).detach().numpy()  # (sample, seq, inp_dim)
            # y(t) = y(t-1) + y(t-2) + ... + y(t-n), the same applies for u.
            elif self.model_type == "FFNN":
                yyi = np.flip(y[i:(i + self.seq_length), :], axis=0)  # Reverse y though u is already reversed.
                X = np.concatenate((u[i, :].reshape(-1, 1), yyi), axis=1)
                X = torch.tensor(self.sc.transform(X), dtype=torch.float32)
                yi = self.model(self.x_size(X, self.input_dim)).detach().numpy()
            else:
                return "Empty"
            yi = np.hstack((np.zeros((1, 1)), yi))
            yi = self.sc.inverse_transform(yi)[0, -1]
            y[(i + self.seq_length), :] = yi
        self.out = (test_data, y)

    def validation(self, test_data_path=None):
        """ Validate the models performing one/step ahead predictions with respect to the measured data.

        Args:
            test_data_path: Path or a structure with the data to simulate.

        """
        test_data = data_preparation(test_data_path, scale=None)
        net_data = self.sc.transform(test_data['u_y'])
        if self.model_type == "LSTM" or self.model_type == "RNN":
            X, y_out = split_recurrent_sequence(net_data, self.seq_length)
            X = torch.tensor(X, dtype=torch.float32)
            y = (self.model(X.view(self.x_size))).detach().numpy()
        elif self.model_type == "FFNN":
            X, y_out = split_feedforward_sequence(net_data, self.seq_length)
            X = torch.tensor(X, dtype=torch.float32)
            y = self.model(X).detach().numpy()
        else:
            return "Empty"
        y = np.hstack((np.zeros((len(y), 1)), y))
        y = self.sc.inverse_transform(y)[:, 1].reshape(-1, 1)
        y = np.append(test_data['u_y'][:self.seq_length, 1], y)  # Add the initial sequence at the beginning of the list

        self.out = (test_data, y)

    def draw_compare_figures(self, save_csv=None, save_fig=None):
        test_data, predicted_data = self.out
        fig1, axes = plt.subplots(nrows=1, ncols=1, figsize=tmp.set_size_long())
        # Process variable y
        axes.step(test_data['t'], test_data['u_y'][:, 1], 'g-', alpha=0.5, markersize=5, where='post')
        axes.step(test_data['t'], predicted_data, 'b-', alpha=0.5, markersize=5,
                  where='post')
        axes.set(ylabel=r"$pH$ $[-]$")
        axes.set(xlabel=r"$t$ $[s]$")
        axes.grid(True)
        axes.set_xlim(0, test_data['t'][-1])
        fig1.legend(['Raw Data', 'Predicted Data'],
                   loc='upper center', fancybox=None, ncol=6, borderaxespad=0.1, edgecolor='black', fontsize=8,
                   bbox_to_anchor=(0.5, 1), borderpad=0.3)
        # Control input u
        fig2, axes = plt.subplots(nrows=1, ncols=1, figsize=tmp.set_size_long())
        axes.step(test_data['t'][:], test_data['u_y'][:, 0], "b-", where='post')
        axes.set(ylabel=r"$F_2$ $[mL.s^{-1}]$")
        axes.set(xlabel=r"$t$ $[s]$")
        axes.grid(True)
        axes.set_xlim(0, test_data['t'][-1])

        # MAE
        model_mae = abs(test_data['u_y'][:, 1] - predicted_data.squeeze())

        fig3, axes3 = plt.subplots(nrows=1, ncols=1, figsize=tmp.set_size_long())
        axes3.plot(test_data['t'], model_mae, "b-")
        axes3.set(xlabel=r"$t$ $[s]$")
        axes3.set(ylabel=r"Absolute error")
        axes3.grid(True)
        axes3.set_xlim(0, test_data['t'][-1])

        if save_fig is not None:
            fig1.savefig('{}_y.eps'.format(save_fig), format="eps")
            fig2.savefig('{}_u.eps'.format(save_fig), format="eps")
            fig3.savefig('{}_mae.eps'.format(save_fig), format="eps")
        if save_csv is not None:
            data = np.hstack(
                (test_data['t'].reshape(-1, 1), test_data['u_y'][:, 0].reshape(-1, 1), predicted_data.reshape(-1, 1)))
            df = pd.DataFrame(data, columns=['t', 'u', 'y'])
            df.to_csv('{}.csv'.format(save_csv), index=False)
        plt.show()


class NNSimulator(NNLoader):
    def __init__(self, sim_time, model_path=None, model_type=None):
        super(NNSimulator, self).__init__(model_path, model_type)
        self.sim_time = sim_time
        self.k_samples = int(self.sim_time / self.delta_t)
        "The delta_t step size depends on the used data in a simulation training!"
        self.t = array([(self.delta_t * x + 0) for x in range(self.k_samples)])

    def closed_loop_sim(self, Ts, ss, referencer, u_controller=None):
        """Closed loop simulation for the neural networks with sequential input(auto-regressive).

        Args:
            ss: Initial conditions in a steady state
            Ts (float): Sampling period
            u_controller: Controller returns the external input to the process
            referencer: Reference of the desired value

        Returns:
            dict: Data structure of simulation always containing [t, x, y, u].

        """
        if self.u_dim == 0:
            raise ValueError("There is only 1 or 0 input dimensions, therefore"
                             " not possible to control closed-loop system")
        if Ts % self.delta_t != 0:
            raise ValueError("Remainder Ts to delta_t simulation is not zero.")
        if u_controller is not None:
            u_controller.Ts = Ts

        x0s, u0s, y0s = ss  # Steady state is also an initial condition.

        # Vector declaration for storing the simulation data.
        u = np.zeros(shape=(self.k_samples, self.u_dim))
        u_save_controller = np.zeros(shape=(int(self.sim_time/Ts), self.u_dim))  # Structure only for the controller Ts.
        y = np.zeros(shape=(self.k_samples, self.output_dim))
        ref = referencer.out(self.t, dim=(self.k_samples, self.output_dim))  # Generate a reference sequence.

        # Initial conditions for the simulation considering NN model delta_t
        u[:self.seq_length, :] = np.array([u0s for _ in range(self.seq_length)]).reshape(-1, 1)  # u \in N
        y[:self.seq_length + 1, :] = np.array([y0s for _ in range(self.seq_length + 1)]).reshape(-1, 1)  # y<=>x \in N+1

        # Separate initial conditions for the simulation considering controller Ts (only for storing u)
        # Find how many u per Ts fits init seq time, in case the Ts is not whole take the lower value
        init_ctrl_seq_length = math.ceil(self.seq_length*self.delta_t / Ts)
        u_save_controller[:init_ctrl_seq_length, :] = u0s
        j = init_ctrl_seq_length  # iterator for the controller

        # Create scalar init cond which is the last element of the initial condition sequences.
        ui = u0s
        yi = y0s

        """Step is 1 too, so far we're doing only 1 step prediction mode.
        # Moving window sequence: Each sample of u and y is seq_length long."""
        prediction_step = 1
        stop = self.k_samples - self.seq_length  # shorter by an initial sequence
        for i in range(0, stop, prediction_step):
            # Calculate the control input u at time t.
            if self.t[i + self.seq_length] % Ts == 0:
                e = (ref[i + self.seq_length, :] - y0s) - (yi - y0s)  # Subtract the steady states from current out y.
                if u_controller.tag_spec:
                    we = (ref[i + self.seq_length, :] - y0s)
                    ye = (yi - y0s)
                    ui = u_controller(e, we, ye) + u0s
                else:
                    ui = u_controller(e) + u0s
                ui = max(ui, 0)  # Add saturation to avoid the negative numbers.
                u_save_controller[j, :] = ui  # save the control input in its own array
                j += 1

            u[i + self.seq_length, :] = ui  # save the control input for the simulation

            # Calculate the process variable y or just pH at time t.
            # y(t) = y(t-n) + ... + y(t-2) + y(t-1), the same applies for u.
            if self.model_type == "LSTM" or self.model_type == "RNN":
                X = np.concatenate((u[i + 1:(i + self.seq_length) + 1, :], y[i + 1:(i + self.seq_length) + 1, :]),
                                   axis=1)  # include the new computed u
                X = torch.tensor(self.sc.transform(X), dtype=torch.float32)
                yi = self.model(X.view(self.x_size)).detach().numpy()
            # y(t) = y(t-1) + y(t-2) + ... + y(t-n), the same applies for u.
            elif self.model_type == "FFNN":
                X = np.concatenate((np.flip(u[i + 1:(i + self.seq_length) + 1, :], axis=0),
                                    np.flip(y[i + 1:(i + self.seq_length) + 1, :], axis=0)), axis=1)
                X = torch.tensor(self.sc.transform(X), dtype=torch.float32)
                yi = self.model(self.x_size(X, self.input_dim)).detach().numpy()
            else:
                return "Empty"
            yi = np.hstack((np.zeros((1, 1)), yi))
            yi = self.sc.inverse_transform(yi)[0, -1]
            if i + 1 != stop:  # Do not save the last element x_{t+1} element.
                y[(i + 1 + self.seq_length), :] = yi  # But, save the process variable where yi = x(t+1).

        self.out = {"t": self.t, "x": None, "y": y, "u": u, "u_controller": u_save_controller,
                    "r": ref, "method_key": [self.model_type]}
