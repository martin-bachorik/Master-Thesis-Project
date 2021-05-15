from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np
import pandas as pd


def split_recurrent_sequence(data, n_steps, output_dim=1):
    X, y = list(), list()
    "Multi sequence case"
    for i in range(len(data)):
        end_idx = i + n_steps  # Find the end of this pattern.
        # Check if we are beyond the dataset or not.
        if end_idx > len(data) - 1:
            break
        """" Extract the input sequence from the original and save to the list.
         Example for seq_len=3 with output_dim=1: seq_x[[1, 1], [2, 2], [3, 3]] and seq_y[4]
         Vectors are sliced except the last one since we start from the 0 index(this is not Matlab!),
         therefore the seq_y is the next ongoing value +1 or +output_dim."""
        seq_x, seq_y = data[i:end_idx, :], data[end_idx, -output_dim::]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)


def split_feedforward_sequence(data, seq):
    # SLIDING WINDOW --> shifts data(both y and external u) by p
    cols = ['u', 'pH']
    data = pd.DataFrame(data, columns=cols)

    for k in range(len(cols)):
        for i in range(1, seq + 1):
            # shifts col vectors
            # data['{}: Lag {}'.format(cols[k], i)] = data[cols[k]].shift(i)
            data['{}: Lag {}'.format(cols[k], i)] = data[cols[k]].shift(i)

    data = data.dropna()

    X = np.array(data.iloc[:, 2::].values)
    y = np.array(data['pH'].values).reshape(-1, 1)  # reshape is need to form (n, 1)

    return X, y


def data_preparation(file, scale=None, delta_t=None):
    """ Decompress csv files into vectors

    Returns:
        dict: Structure contains scale factor, time and data vector by order [u, y].
    """
    if isinstance(file, str) is True:
        read_data = pd.read_csv(file)
        t = np.array(read_data['t'].values.reshape(-1, 1), dtype="float32")
        u_input = np.array(read_data['u'].values.reshape(-1, 1), dtype="float32")
        y_out_input = np.array(read_data['y'].values.reshape(-1, 1), dtype="float32")  # Previous y are the inputs too.
    elif isinstance(file, dict) is True:
        file = {'t': file['t'].squeeze(), 'u': file['u'].squeeze(), 'y': file['y'].squeeze()}
        read_data = pd.DataFrame(file)
        t = read_data['t']
        if delta_t is not None:  # secure the data copy the same sampling time as the model
            if t[1] - t[0] != delta_t:
                read_data = read_data[read_data['t'] % delta_t == 0]
        t = np.array(read_data['t'].values.reshape(-1, 1), dtype="float32")
        u_input = np.array(read_data['u'].values.reshape(-1, 1), dtype="float32")
        y_out_input = np.array(read_data['y'].values.reshape(-1, 1), dtype="float32")  # Previous y are the inputs too.
    else:
        raise ImportError("Wrong data")

    data_inputs = np.hstack((u_input, y_out_input))  # Bunch vector together
    if scale == "Standard":
        sc1 = StandardScaler()
        data_inputs = sc1.fit_transform(data_inputs)
        return {"scale": sc1,
                "t": t,
                "u_y": data_inputs}
    elif scale == "MinMax":
        sc1 = MinMaxScaler(feature_range=(-1, 1))
        data_inputs = sc1.fit_transform(data_inputs)
        return {"scale": sc1,
                "t": t,
                "u_y": data_inputs}
    elif scale is None:
        return {"scale": None,
                "t": t,
                "u_y": data_inputs}
    else:
        raise NameError("Incorrect scale option!")


def x_size_feedforward(x_in, input_dim):
    return np.transpose(x_in).reshape(-1, input_dim)


def x_size_recurrent(seq_length, input_dim):
    return [-1, seq_length, input_dim]
