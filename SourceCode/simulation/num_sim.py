from numpy import zeros, array
from . import numeric
import time


class Simulator:
    __method_table = numeric.Methods.get_method_table()

    def __init__(self, t_sim, num_method_key=None, model=None, delta_t=None):
        """

        Args:
            model (class_handle): System of differential equation/s
            delta_t (float): Simulation step size
            t_sim (float): Simulation time horizon
            num_method_key (array_like): Chosen numerical method

        Notes:
            x --> states of the system
            u --> external/control inputs of the system
            y --> outputs of the system

        """
        self.t_sim = t_sim
        if num_method_key is not None:
            if len(num_method_key) > 1:
                self.method = {}  # Aggregate the methods by definition of the keys!
                for key in num_method_key:
                    try:
                        self.method.update({"{}".format(key): self.__method_table[key]})
                    except KeyError:
                        num_method_key.remove(key)
                        print("{} method is not available! Check the list of methods again.".format(key))
            else:
                self.method = {"{}".format(num_method_key[0]): self.__method_table[num_method_key[0]]}

        if delta_t is not None:
            self.delta_t = delta_t  # Simulation step
        else:
            raise ValueError("Missing simulation step size delta_t")

        # Aggregate a model object specifics and its dimensions.
        self.model = model
        if model is not None:
            self.nx = model.nx
            self.nu = model.nu
            self.ny = model.ny

        self.out = None

    def open_loop_sim(self, x0, u_signal=None, true_solution=False):
        """ Numerical open-loop simulation with optional control input

        Possible to simulate multiple methods at once

        Args:
            x0: Initial conditions
            u_signal: Process input signal
            true_solution: Analytical solution to the system

        Returns:
            dict: Data structure of simulation always containing [t, x, y, u].

        """
        if self.method is False:
            raise NameError("No method was given!")
        if u_signal is None:
            raise NameError("Either the control signal was not set or the name was misplaced.")

        t, x, y, u, n_samples = self._declare_vector()
        u = u_signal.out(t, dim=(n_samples, self.nu))

        counter = list()
        if len(self.method) > 1:
            for i, key in enumerate(self.method):
                start_count = time.time()
                x[i, :, :] = array(self.method[key](self.model.system_state, t, x0, u, deltaT=self.delta_t))
                counter.append(time.time() - start_count)
                for j in range(n_samples):
                    y[i, j, :] = self.model.system_output(x[i, j, 0], x[i, j, 1])  # Outputs
        else:  # TODO maybe some better method
            start_count = time.time()
            x = array(list(self.method.values())[0](self.model.system_state, t, x0, u, deltaT=self.delta_t))
            counter.append(time.time() - start_count)
            for j in range(n_samples):
                y[j, :] = self.model.system_output(x[j, 0], x[j, 1])

        if true_solution:
            true_x = self.model.analytic(t, x0, u).T
        else:
            true_x = None
        self.out = {"t": t, "x": x, "y": y, "u": u, "r": None,
                    "method_key": [x for x in self.method.keys()],
                    "true_solution": true_x,
                    "method_benchmark": counter}

    def closed_loop_sim(self, Ts, ss, referencer, u_controller=None, u_external=None, r_external=None):  # TODO move Ts to the constructor?
        """System process simulated by numerical method and controlled with discrete controller.

        Args:
            r_external: External reference
            u_external: External control input
            Ts (float): Sampling period
            ss: Initial conditions in the steady state
            u_controller: Controller returns the external input to the process
            referencer: Reference of the desired value

        Returns:
            dict: Data structure of simulation always containing [t, x, y, u].

        Notes:
            -The system process is considered in this case to be "continuous",
             though the simulation step size is constant, but sufficiently small enough.
            -Can be used in the same way as open_loop_sim() function,but more complicated to set up and longer it takes.
        """
        if ((Ts/self.delta_t) % 1 == 0) is not True:
            raise ValueError("Either incorrect simulation or sampling time."
                             " Both has to be an integer, the simulation could proceed correctly.")
        if len(self.method) == 1:
            method = [x for x in self.method.values()][0]
        else:
            raise NotImplementedError(
                "Choose only 1 method for the closed-loop simulation.")
        if u_controller is not None:
            u_controller.Ts = Ts
        elif u_controller is None and u_external is None:
            raise ValueError("No control input was set.")

        t, x, y, u, n_samples = self._declare_vector()
        x0s, u0s, y0s = ss  # Steady state is also an initial condition

        # initial conditions
        x[0, :] = x0s
        ui = u0s  # Typically, in continuous-time controllers initial u(t) is omitted since there is no Ts
        yi = y0s

        if r_external is not None:
            ref = self.__convert_to_continuous(r_external, Ts, n_samples)
        else:
            ref = referencer.out(t, dim=(n_samples, self.ny))

        j = 0
        for i in range(0, n_samples, 1):
            if t[i] % Ts == 0:
                if u_controller is not None:  # u from the controller
                    e = (ref[i] - y0s) - (yi - y0s)
                    if u_controller.tag_spec:
                        we = (ref[i] - y0s)
                        ye = (yi - y0s)
                        ui = u_controller(e, we, ye) + u0s  # gain scheduler
                    else:
                        ui = u_controller(e) + u0s  # add steady state to deviation output from the controller

                    ui = max(ui, 0)  # add saturation
                elif u_external is not None:  # u from the external input
                    ui = u_external[j]  # TODO correct this later
                    j += 1

            if i < len(t)-1:  # shorter because of x_{i+1} overflow
                x[i+1, :] = array(method(self.model.system_state, t[i], x[i, :], ui, deltaT=self.delta_t))

            yi = self.model.system_output(x[i, 0], x[i, 1])

            u[i, :] = ui
            y[i, :] = yi

        self.out = {"t": t, "x": x, "y": y, "u": u, "r": ref,
                    "method_key": [x for x in self.method.keys()]}

    def _declare_vector(self):
        # TODO this could be also in constructor or maybe not.. is it now better variability of sim obj due to sim_t?
        """Initialization of time variables required for the simulation of the system.
        """
        n_method = len(self.method)
        sim_interval = {'start': 0, 'end': self.t_sim}
        # Time discretization, +1 including last point in time interval [0, t].
        n_samples = int((sim_interval['end'] - sim_interval['start']) / self.delta_t)
        t = array([(self.delta_t * x + sim_interval['start']) for x in range(n_samples)])
        if n_method > 1:
            x = zeros(shape=(n_method, n_samples, self.nx))
            y = zeros(shape=(n_method, n_samples, self.ny))
            u = zeros(shape=(n_method, n_samples, self.nu))
        else:
            x = zeros(shape=(n_samples, self.nx))
            y = zeros(shape=(n_samples, self.ny))
            u = zeros(shape=(n_samples, self.nu))
        return t, x, y, u, n_samples

    def __convert_to_continuous(self, r_old, Ts, n_samples):
        r_new = zeros(shape=(n_samples, self.ny))
        t_factor = int(Ts/self.delta_t)
        for i in range(len(r_old)):
            r_new[i*t_factor:(i*t_factor + t_factor), 0] = r_old[i, 0]
        return r_new
