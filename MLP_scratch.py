import numpy as np
import math
from math import e
from matplotlib import pyplot as plt

# Local libraries
from SourceCode.simulation.nn.preprocessing import data_preparation, split_feedforward_sequence, x_size_feedforward
from SourceCode import templates

templates.Template()


class NeuralFramework:
    def __init__(self):
        self.current_layer = None  # Pointer to the current/linked layer (HEAD)
        self.learning_rate = None
        self.loss_type = None

    def add(self, dim_m, dim_n, nonlinearity=None):
        new_layer = NodeLayer(dim_m, dim_n, nonlinearity=nonlinearity)  # create a layer
        if self.current_layer is None:
            self.current_layer = new_layer  # add the first layer to the network/linked-list
        else:
            new_layer.prev_layer = self.current_layer  # add to the new layer object, previously formed network/linked-list
            self.current_layer.next_layer = new_layer  # add to the previous layer, newly formed layer
            self.current_layer = new_layer  # set/link the pointer(HEAD)
        return self  # return the actual network

    def forward(self, x_input):
        """ Forward starts from the input-hidden(first) layer <-> current_layer.previous_layer = None
        """
        current_layer = self.current_layer  # HEAD is copied and will not be overwritten
        # Traverse all the layers from the output layer to the input-hidden(first) layer backward
        while current_layer.prev_layer is not None:
            current_layer = current_layer.prev_layer
        # Traverse from the input-hidden(first) layer to the output(last) layer forward, while performing forward()
        while current_layer.next_layer is not None:
            x_input = current_layer.forward(x_input)  # forward for current layer
            current_layer = current_layer.next_layer

        y_output = current_layer.forward(x_input)  # last forward for the output layer
        return y_output

    def update(self):
        while self.current_layer.prev_layer is not None:
            self.current_layer = self.current_layer.prev_layer

        self.gradient_descent()  # First update
        self.current_layer.gradient = None  # empty the gradient from the layer
        while self.current_layer.next_layer is not None:
            self.current_layer = self.current_layer.next_layer  # move layer
            self.gradient_descent()
            self.current_layer.gradient = None

    def loss_f(self, y_target, y_est):
        if self.loss_type == "MSE":
            err = (y_target - y_est) ** 2
            err = 0.5 * np.mean(err)
            # return np.sqrt(err)
            return err

    @staticmethod
    def derivative_loss_MSE(y_target, y_est):
        return (1 / np.size(y_target)) * (y_est - y_target)  # mean
        # return (y_est - y_target)  # reduction

    def gradient_descent(self):
        self.current_layer.weights = self.current_layer.weights - self.learning_rate * self.current_layer.gradient[
            'weights']
        self.current_layer.biases = self.current_layer.biases - self.learning_rate * self.current_layer.gradient[
            'biases']

    def backward(self, y_target=None, y_est=None, acc_next_delta=None):
        """ Backward starts from the output(last) layer <-> current_layer.next_layer = None
            New accumulation of the error from the next_layer:
                -acc_next_delta is accumulated output_flow! :=> [hidd/inp x batch];
            Architecture:
                -In my case(weights): neuron[output_flow, input_flow];
                -Biases have no input_flow => neuron[output_flow, 1];

            Output layer:
                acc_current_delta = dev_cost * dev_nonlin_f(x^l);

            Hidden layer:
                acc_current_delta = ((W^(l+1)^T @ delta^(l+1))) * dev_nonlin_f(x^l);
                acc_current_delta = [hidd/out x hidd/inp]^T x [hidd/out x batch] = [hidd/inp x batch_size];

            delta_w = [hidd/out x batch_size] x [hidd/inp x batch_size]^T = [hidd/out x hidd/inp];
            delta_b = [hid/out x batch_size] x ones[1 x batch] = [hid/out x 1];

        """
        if self.current_layer.next_layer is None:  # output layer
            loss = self.derivative_loss_MSE(y_target, y_est)  # outer derivative of the cost function
            acc_current_delta = loss  # send delta^{l+1} lower

            delta_w = loss @ self.current_layer.layer_inputs.T
            delta_b = loss @ np.ones((1, loss.shape[1])).T
            self.current_layer.gradient = {'weights': delta_w, 'biases': delta_b}

            self.current_layer = self.current_layer.prev_layer  # move a layer lower
            self.backward(y_target=None, y_est=None, acc_next_delta=acc_current_delta)  # call backward recursively

        elif self.current_layer.next_layer is not None:  # hidden layers
            acc_current_delta = (
                                        self.current_layer.next_layer.weights.T @ acc_next_delta) * self.current_layer.dev_nonlinear_f(
                self.current_layer.linear_layer_outputs)

            delta_w = acc_current_delta @ self.current_layer.layer_inputs.T
            delta_b = acc_current_delta @ np.ones((1, acc_current_delta.shape[1])).T
            self.current_layer.gradient = {'weights': delta_w, 'biases': delta_b}

            if self.current_layer.prev_layer is not None:
                self.current_layer = self.current_layer.prev_layer  # move a layer lower
                self.backward(y_target=None, y_est=self.current_layer.next_layer.layer_inputs,
                              acc_next_delta=acc_current_delta)  # call backward propagation again

            # Reset and update to the last output layer
            elif self.current_layer.prev_layer is None:
                self.gradient_descent()  # update parameters of the network
                # self.current_layer.gradient = None  # empty the gradient from the layer
                while self.current_layer.next_layer is not None:
                    self.current_layer = self.current_layer.next_layer
                    self.gradient_descent()
                    # self.current_layer.gradient = None  # empty the gradient from the layer


class NodeLayer:
    def __init__(self, dim_m, dim_n, nonlinearity=None):
        # Activation function
        self.nonlinearity = nonlinearity
        # 2 pointers (bidirectional)
        self.prev_layer = None  # Pointer to the previous layer
        self.next_layer = None  # Pointer to the next layer

        # Random initialization of weights
        self.weights = np.random.normal(0, 0.05, size=(dim_m, dim_n))
        self.biases = np.zeros(shape=(dim_m, 1))

        self.layer_inputs = None
        self.linear_layer_outputs = None
        self.gradient = None

    def forward(self, x):
        """Forward propagation
        Save the inputs x in NodeLayer object during forwardpropagation, which are necessary for backpropagation
        y = Wx + b
        """
        self.layer_inputs = x
        y = self.weights @ x + self.biases

        self.linear_layer_outputs = y
        return self.nonlinear_f(y)  # Linear output sent through the activation function.

    def nonlinear_f(self, x):
        if self.nonlinearity == "sigmoid":
            return self.Sigmoid(x)
        elif self.nonlinearity == "tanh":
            return self.Tanh(x)
        elif self.nonlinearity == "relu":
            return self.ReLU(x)
        elif self.nonlinearity == "lrelu":
            return self.Leaky_ReLU(x)
        elif self.nonlinearity == "none":
            return x
        else:
            return x

    def dev_nonlinear_f(self, x):
        if self.nonlinearity == "sigmoid":
            return self.Sigmoid(x) * (1 - self.Sigmoid(x))
        elif self.nonlinearity == "tanh":
            return 1 - (self.Tanh(x)) ** 2
        elif self.nonlinearity == "relu":
            x[x > 0] = 1
            x[x <= 0] = 0
            return x
        elif self.nonlinearity == "lrelu":
            return np.where(x > 0, x, x * 0.01)
        elif self.nonlinearity == "none":
            return np.ones(x.shape)
        else:
            pass
        return

    @staticmethod
    def Sigmoid(x):
        y = 1 / (1 + e ** (-x))
        return y

    @staticmethod
    def Tanh(x):
        return np.tanh(x)

    @staticmethod
    def ReLU(x):
        y = np.maximum(0, x)
        return y

    @staticmethod
    def Leaky_ReLU(x):
        y = np.maximum(0.01 * x, x)
        return y


class NeuralNetwork:
    def __init__(self, hidden_dim, input_dim, output_dim):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Define the network
        self.net = NeuralFramework()

        # Hidden layers
        self.net.add(hidden_dim, input_dim, nonlinearity='tanh')
        self.net.add(hidden_dim, hidden_dim, nonlinearity='tanh')
        # self.net.add(hidden_dim, hidden_dim, nonlinearity='tanh')
        # self.net.add(hidden_dim, hidden_dim, nonlinearity='tanh')

        # Output layer
        self.net.add(output_dim, hidden_dim, nonlinearity='none')

        self.net.learning_rate = 0.1
        self.net.loss_type = 'MSE'

    def train(self, data, batch_size, num_epochs):
        data = self.batch_cutter(data, batch_size)
        for epoch in range(num_epochs):
            for x_input, y_target in data:  # dim(data) >> [No. batches, [x, y], batch_size(samples), inputs]
                x_input = x_input.T  # dim(x_input) >> [batch_size(samples), inputs]^T
                y_target = y_target.T

                y_est = self.net.forward(x_input)  # Forward propagation
                self.net.backward(y_target, y_est)  # Backward propagation collects gradients and updates parameters
            loss = self.net.loss_f(y_target, y_est)
            print("Epoch: {}, MSE loss: {}".format(epoch, loss))

    def predict(self, x):
        input_layer = np.transpose(x)
        out = self.net.forward(input_layer)
        return np.transpose(out)

    def batch_cutter(self, data, batch_size):
        """ Cut the data into the batches and changes the dimensions
        """
        data = np.array(np.concatenate([data['x'], data['y']], axis=1))  # Nested list: [x/y, sample, in/out]
        k = math.floor(len(data) / batch_size)
        store = list()
        for i in range(0, k):
            X = data[i * batch_size:(batch_size * (i + 1)), 0:self.input_dim]
            y = data[i * batch_size:(batch_size * (i + 1)), self.input_dim:]
            store.append([X, y])
        return store


class Main:
    @staticmethod
    def exec():
        # Define Hyper parameters
        seq_length = 10

        hidden_dim = 24
        input_param = 1
        output_dim = 1
        input_dim = (input_param + output_dim) * seq_length  # auto-regression
        # input_dim = input_param  # no auto-regression
        net = NeuralNetwork(hidden_dim, input_dim, output_dim)

        batch_size = 256
        num_epochs = 20

        scaled_data = data_preparation(
            "SourceCode/data/MultiStep/T10s/random_open_data.csv", scale="MinMax")
        sc1 = scaled_data['scale']
        h = int(len(scaled_data['t']) / 2)

        # Training data
        time1 = scaled_data['t'][h:]
        train_data = scaled_data['u_y'][h:]
        X_train, y_train = split_feedforward_sequence(train_data, seq_length)

        # # Validation data
        # time2 = scaled_data['t'][-h:]
        # valid_data = scaled_data['u_y'][-h:]

        # Testing data
        test_data_all = data_preparation(
            "SourceCode/data/MultiStep/T10s/Step10s_10steps_1_7.csv", scale=None)

        # Train the network
        training_data = {'x': X_train, 'y': y_train}
        net.train(training_data, batch_size, num_epochs)

        # Predict
        tag = 'online'
        if tag == "validation":
            test_data = sc1.transform(test_data_all['u_y'])
            X2, y2 = split_feedforward_sequence(test_data, seq_length)

            y = net.predict(X2)
            y = np.append(y2[:seq_length], y).reshape(-1, 1)
            y = np.hstack((np.zeros((len(y), input_param)), y))
            y = sc1.inverse_transform(y)[:, input_param:]

            y2 = np.hstack((np.zeros((len(test_data[:, -output_dim:]), input_param)), test_data[:, -output_dim:]))
            y2 = sc1.inverse_transform(y2)[:, input_param:]  # Transform original output
        elif tag == "online":
            test_data = test_data_all['u_y']
            X2, y2 = split_feedforward_sequence(test_data, seq_length)
            u = X2[:, :seq_length]
            y0_in = X2[0, seq_length::].reshape(-1, 1)
            y0_in = y0_in[::-1]  # Reverse back the slice sequence.
            y = np.zeros(shape=(len(test_data_all['t']), output_dim))

            y[:seq_length] = y0_in
            prediction_step = 1
            for i in range(0, len(test_data_all['t']) - seq_length, prediction_step):
                # y(t) = y(t-1) + y(t-2) + ... + y(t-n), the same applies for u.
                yyi = np.flip(y[i:(i + seq_length), :], axis=0)  # Reverse y though u is already reversed.
                X = np.concatenate((u[i, :].reshape(-1, 1), yyi), axis=1)
                X = sc1.transform(X)
                yi = net.predict(x_size_feedforward(X, input_dim))

                yi = np.hstack((np.zeros((1, 1)), yi))
                yi = sc1.inverse_transform(yi)[0, -1]
                y[(i + seq_length), :] = yi

            y2 = test_data[:, -output_dim:]

        # Plot
        plt.figure()
        plt.plot(test_data_all['t'], y2)
        plt.plot(test_data_all['t'], y)
        plt.xlabel('$t$ $[s]$')
        plt.ylabel('$pH$ $[-]$')
        plt.legend(['Testing Data', 'Predicted Data'])

        plt.show()


if __name__ == "__main__":
    Main.exec()
