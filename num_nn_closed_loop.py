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
        # PARAMETER SETTINGS for the simulation are provided below. All settings are configured in second units.
        Ts = 10
        delta_t = 0.5
        step_time = 1800
        n_steps = 5
        total_time = n_steps * step_time
        num_method = ['RK-4th']

        # us:  ph6: 4.545, ph7: 4.950 ,ph8: 4.9952, ph9: 5.0 ,ph10: 5.02, ph11: 5.204
        u0s = 4.545
        x0s, y0s = NReactor.initial_conditions(us=u0s, x1s=True, x2s=True)

        u_c = controller.PIOver(Zr=0.15, Ti=80, beta=0.8)

        # referencer = signal.RandStep(step_time, step_interval=[7, 9], n_step=1000, ss=y0s)
        # referencer = signal.GaussStep(step_time, mu=8.5, sigma=1, n_step=2000, ss=y0s)
        referencer = signal.Step(step_time, step=[y0s, 6.2, 11, 6, 10.5])
        # referencer = signal.Step(step_time, step=[y0s, 7, 6.5, 7, 10.5, 7, 9.5, 7])
        # referencer = signal.Step(step_time, step=[y0s, 10])

        "Close loop simulation with NN"
        nn_sim_model = NNSimulator(total_time, model_path='./SourceCode/models/lstm/LSTM_closed_loop.pt',
                                   model_type="LSTM")
        nn_sim_model.closed_loop_sim(Ts, (x0s, u0s, y0s), referencer, u_controller=u_c)
        u = nn_sim_model.out['u_controller']

        "Closed-loop simulation with numerical methods"
        sim_model = Simulator(total_time, model=NReactor, num_method_key=num_method, delta_t=delta_t)
        sim_model.closed_loop_sim(Ts, (x0s, u0s, y0s), referencer, u_controller=None, u_external=u, r_external=None)
        out = sim_model.out

        "Plotting option for the output"
        color_option = [['b-', 'b--']]
        graph = Graph(out)
        graph.draw_system_figures(color_option, true_solution=False, intersection=False, show=True, save_fig=None, save_csv=None)


if __name__ == "__main__":
    Main.exec()
