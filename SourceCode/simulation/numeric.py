from numpy import size
from scipy.integrate import odeint


class Methods:
    """Numerical methods for solving ordinary differential equations and systems

        Each method has the same input and output parameters.

            Args
                diff: handle/s differential equation/s
                x: independent variable, time
                Y0: dependent variable, initial conditions of an output value/s
                u: external variable, control input
                deltaT:

            Returns
                Array: Output values of differential equation/s

            Notes
                All method bypasses to unnecessary operation extracting a vector in form [:, i]

                y_(i+1) = y_(i) + f'(x_(i), y_(i))*dt(i)
                X_(i+1) = X_(i) + f'(t_(i), X_(i))*dt(i)
                deltaT = t[i+1] - t[i]
            """
    @classmethod
    def get_method_table(cls):
        """ Serves as a method collector to higher applications.

        Returns:
            dict: Returns the dictionary of all numerical methods for differential equations.
        """
        return {"Euler": cls.__forward_Euler_method,
                "Heun": cls.__Heun_method,
                "RK-4th": cls.__Runge_Kutta_4,
                "RK-5th": cls.__Runge_Kutta5_Butcher,
                "Ode-int": cls.__odeint}

    @classmethod
    def __forward_Euler_method(cls, diff, x, Y0, u=None, deltaT=None):  # Explicit Euler's method
        if size(x) == 1:
            Y = Y0 + diff(x, Y0, u) * deltaT
            return Y
        else:
            Y = [0 for x in range(size(x))]
            Y[0] = Y0  # insert 2D/nD numpy arr to 1D list
            for i in range(size(x) - 1):  # shorter loop because of Y_{i+1}
                Y[i + 1] = Y[i] + diff(x[i], Y[i], u[i]) * deltaT  #
            return Y

    @classmethod
    def __Heun_method(cls, diff, x, Y0, u=None, deltaT=None):
        """This method will only work in open-loop because lack of knowledge u[i+1]"""
        Y = [0 for x in range(len(x))]
        Y[0] = Y0
        for i in range(len(x) - 1):
            Y[i + 1] = Y[i] + (diff(x[i], Y[i], u[i]) + diff(x[i + 1], Y[i] + deltaT * diff(x[i], Y[i], u[i]),
                                                             u[i + 1])) * deltaT / 2
        return Y

    @classmethod
    def __Runge_Kutta_4(cls, diff, x, Y0, u=None, deltaT=None):
        """Combination of slopes"""

        def sFunc(x, Y, u, deltaT):  # Slope function
            k1 = diff(x, Y, u)
            k2 = diff(x + deltaT / 2, Y + (deltaT * k1) / 2, u)
            k3 = diff(x + deltaT / 2, Y + (deltaT * k2) / 2, u)
            k4 = diff(x + deltaT, Y + deltaT * k2, u)
            # sum of slopes slope
            S = 1 / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            return S

        if size(x) == 1:
            S = sFunc(x, Y0, u, deltaT)
            Y = Y0 + deltaT * S
            return Y
        else:
            Y = [0 for x in range(len(x))]
            Y[0] = Y0
            for i in range(len(x) - 1):
                S = sFunc(x[i], Y[i], u[i], deltaT)
                Y[i + 1] = Y[i] + deltaT * S
        return Y

    @classmethod
    def __Runge_Kutta5_Butcher(cls, diff, t, X0, u=None, deltaT=None):
        # higher order modification of RK4
        def sFunc(t, X, u, deltaT):  # slope function
            k1 = diff(t, X, u)
            k2 = diff(t + deltaT / 4, X + (deltaT * k1) / 4, u)
            k3 = diff(t + deltaT / 4, X + (deltaT * k1) / 8 + (deltaT * k2) / 8, u)
            k4 = diff(t + 1 / 2 * deltaT, X - (deltaT * k2) / 2 + deltaT * k3, u)
            k5 = diff(t + 3 / 4 * deltaT, X + 3 / 16 * (deltaT * k1) + 9 / 16 * (deltaT * k4), u)
            k6 = diff(t + deltaT,
                      X - 3 / 7 * (deltaT * k1) + 2 / 7 * (deltaT * k2) + 12 / 7 * (deltaT * k3) - 12 / 7 * (
                              deltaT * k4) + 8 / 7 * (deltaT * k5), u)
            S = 1 / 90 * (7 * k1 + 32 * k3 + 12 * k4 + 32 * k5 + 7 * k6)  # Sum of the slopes slope
            return S

        if size(t) == 1:
            S = sFunc(t, X0, u, deltaT)
            X = X0 + deltaT * S
            return X
        else:
            X = [0 for x in range(len(t))]
            X[0] = X0
            for i in range(len(t) - 1):
                S = sFunc(t[i], X[i], u[i], deltaT)
                X[i + 1] = X[i] + deltaT * S
        return X

    @classmethod
    def __Runge_Kutta45_Fehlberg(cls, diff, t, X0, u=None, deltaT=None):
        def sFunc(t, X, u, deltaT):  # slope function
            k1 = deltaT * diff(t, X, u)
            k2 = deltaT * diff(t + deltaT / 4, X + k1 / 4, u)
            k3 = deltaT * diff(t + 3 / 8 * deltaT, X + 3 / 32 * k1 + 9 / 32 * k2, u)
            k4 = deltaT * diff(t + 12 / 13 * deltaT, X + 1932 / 2197 * k1 - 7200 / 2197 * k2 + 7296 / 2197 * k3, u)
            k5 = deltaT * diff(t + deltaT, X + 439 / 216 * k1 - 8 * k2 + 3680 / 513 * k3 - 845 / 4104 * k4, u)
            k6 = deltaT * diff(t + 1 / 2 * deltaT,
                               X - 8 / 27 * k1 + 2 * k2 - 3544 / 2565 * k3 + 1859 / 4104 * k4 - 11 / 40 * k5, u)
            S = 16 / 135 * k1 + 6656 / 12825 * k3 + 28561 / 56430 * k4 - 9 / 50 * k5 + 2 / 55 * k6
            return S

        if size(t) == 1:
            S = sFunc(t, X0, u, deltaT)
            X = X0 + deltaT * S
            return X
        else:
            X = [0 for x in range(len(t))]
            X[0] = X0
            for i in range(len(t) - 1):
                S = sFunc(t[i], X[i], u[i], deltaT)
                X[i + 1] = X[i] + deltaT * S
        return X

    @classmethod
    def __odeint(cls, diff, t, X0, u=None, deltaT=None):
        sol = odeint(diff, X0, t, args=(u,), tfirst=True, full_output=False)[-1]
        # x[i+1, :] = array(method(self.model.system_state, [t[i]+x*self.delta_t/10 for x in range(0, 10+1)], x[i, :], ui, deltaT=self.delta_t))
        return sol
