# DO NOT DELETE THESE IMPORTS
from SourceCode.simulation.nn.nn_sim import *
from SourceCode.simulation.num_sim import *


# Standard libraries
import numpy as np

# Local libraries
from SourceCode import controller
from SourceCode import signal
from SourceCode import NeutralizationReactor as NReactor
from SourceCode import Graph
from SourceCode import templates


templates.Template()


class Main:
    @staticmethod
    def exec():
        """ OPTION 1
        Open-loop simulation(loading .csv) with NN"""
        # load_model = NNLoader(model_path='./SourceCode/models/ff/FFNN_open_loop.pt', model_type='FFNN')
        # load_model = NNLoader(model_path='./SourceCode/models/rnn/RNN_open_loop.pt', model_type='RNN')
        load_model = NNLoader(model_path='./SourceCode/models/lstm/LSTM_open_loop.pt', model_type='LSTM')

        # load_model.validation(test_data_path='./SourceCode/data/MultiStep/T10s/Step10s_10steps_1_7.csv')
        load_model.online_open_loop(test_data_path='./SourceCode/data/MultiStep/T10s/Step10s_10steps_1_7.csv')

        load_model.draw_compare_figures(save_fig=None, save_csv=None)

        """OPTION 2
        Open-loop simulation(data generated with numerical method) with NN"""
        # # PARAMETER SETTINGS for the simulation are provided below. All settings are configured in second units.
        # delta_t = 0.5
        # step_time = 1800
        # n_steps = 5
        # total_time = n_steps * step_time
        # num_method = ['RK-4th']
        #
        # u0s = 4.545
        # x0s, y0s = NReactor.initial_conditions(us=u0s, x1s=True, x2s=True)
        # # u_signal = signal.Step(step_time, step=[4.7, 6])
        # # u_signal = signal.Step(step_time, step=[u0s, 4.5, 4.98, 5.1, 4.99, 4.5, 5.2, 4.4, 5.01, 4.7])
        # u_signal = signal.Step(step_time, step=[u0s, 9.98, 1.02, 6.1])
        # # u_signal = signal.Sawtooth(n_steps, step_time, delta_t, amp=5)
        # # u_signal = signal.RandStep(step_time, step_interval=[0, 10], n_step=n_steps, ss=u0s)
        # # u_signal = signal.GaussStep(step_time, mu=5, sigma=1.5, n_step=n_steps, ss=u0s)
        # # u_signal = signal.SawRandStep(step_time, saw_time=step_time/3, step_interval=[0, 10], n_step=n_steps, ss=u0s)
        # # u_signal = signal.SawGaussStep(step_time, saw_time=step_time, delta_t=delta_t, mu=5, sigma=1.5, n_step=n_steps, ss=u0s)
        #
        # sim_model = Simulator(total_time, model=NReactor, num_method_key=num_method, delta_t=delta_t)
        # sim_model.open_loop_sim(x0s, u_signal=u_signal, true_solution=True)
        # out = sim_model.out
        #
        # # load_model = NNLoader(model_path='./simulation/nn/ff/FFNN_open_loop.pt', model_type='FFNN')
        # # load_model = NNLoader(model_path='./simulation/nn/rnn/RNN_open_loop.pt', model_type='RNN')
        # load_model = NNLoader(model_path='./simulation/nn/lstm/LSTM_open_loop.pt', model_type='LSTM')
        #
        # # load_model.validation(test_data_path=out)
        # load_model.online_open_loop(test_data_path=out)
        #
        # load_model.draw_compare_figures(save_fig=None, save_csv=None)


if __name__ == "__main__":
    Main.exec()
