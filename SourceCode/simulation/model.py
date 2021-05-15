# from math import log10, exp
from numpy import log10, exp
from numpy.polynomial import Polynomial as Poly
import numpy as np

__all__ = ['NeutralizationReactor']


class NeutralizationReactor:
    # System dimensions
    nx = 2  # No. states
    nu = 1  # No. MV
    ny = 1  # No. PV
    # Constants
    F1 = 5  # Acetic acid flow [mL/s]
    c1 = 0.05 / 10**3  # Acetic acid concentration [mol/mL]
    c2 = 0.05 / 10**3  # Sodium hydroxide concentration [mol/mL]
    V = 1.5 * 10**3  # Volume of continuously stirred tank [mL]
    ka = 10**(-5)  # Dissociation constant of the acetic acid [-]
    kw = 10**(-14)  # Dissociation constant of water [-]

    @classmethod
    def system_state(cls, t, x, u):
        """The system calculates the equations in scalar value format.

            Two modes: 1. external input; 2. no external input

            Args:
                t (float): Independent variable, time
                x (Array): Dependent variable, states of the system
                u (Array): External variable, control input
            Returns:
                Array: A vector composed of numerical solutions to the system of differential equations.
        """
        # Convert to scalar values
        x1 = x[0].item()
        x2 = x[1].item()
        if np.shape(u) != 1:
            F2 = np.squeeze(u)
        else:
            F2 = u

        F1 = cls.F1
        c1 = cls.c1
        c2 = cls.c2
        V = cls.V

        dx_dt1 = (F1*c1 - (F1+F2)*x1) / V
        dx_dt2 = (F2*c2 - (F1+F2)*x2) / V
        dx_dt = np.array([dx_dt1, dx_dt2])
        return dx_dt

    @classmethod
    def system_output(cls, x1, x2):
        """ Calculate pH of the system given the states of the system.

            Value pH is defined as the negative logarithm (to base 10) of the hydrogen ion concentration in mol.L^{-1}.
            pH = -log_10{c[H+]}

        Args:
            x1 (float): Acetic acid concentrations
            x2 (float): Sodium hydroxide concentration

        Returns:
            float: Value obtained from a polynomial roots, resulting in pH of the solution in the vessel.

        Notes:
            c[H+] is the concentration of hydrogen ions in mol L^{-1} (mol/L or M)
        """
        # CONVERT both x1 & x2 from mol/mL --> mol/L.
        x1 = np.squeeze(x1) / 10**(-3)
        x2 = np.squeeze(x2) / 10**(-3)

        def ph_polynom_coeff(x1, x2, ka, kw):
            return 1, (x2+ka), (x2*ka - x1*ka - kw), -kw*ka
        a, b, c, d = ph_polynom_coeff(x1, x2, cls.ka, cls.kw)
        # Polynomial coefficients need to be reversed.
        p = Poly([d, c, b, a])
        p = p.roots()
        pH_root = 1
        for i in range(len(p)):  # Only one will be positive
            if p[i] > 0:
                pH_root = p[i]
                if pH_root.imag == 0:
                    pH_root = pH_root.real
                break
        # pH = -math.log10(pHRoot)
        pH = -log10(pH_root)
        if pH > 14:
            pH = 14
        elif pH < 0:
            pH = 0
        return pH

    @classmethod
    def initial_conditions(cls, us, x1s=False, x2s=False):
        """Primarily generates a steady state values which can be used as an initial conditions indeed.

        Args:
            us: Steady state flow of sodium hydroxide
            x1s (bool): Steady state of acetic acid concentration
            x2s (bool): Steady state of sodium hydroxide concentration

        Returns: Either tuple of initial conditions (xs, us, ys) or a vector of sequences [us, ys]

        """
        F1, F2, c1, c2, V = cls.F1, us, cls.c1, cls.c2, cls.V

        if x1s is True and x2s is True:
            x1s = (F1*c1) / (F1+F2)
            x2s = (F2*c2) / (F1+F2)
        elif x1s is True:
            x1s = (F1*c1) / (F1+F2)
            x2s = 0
        elif x2s is True:
            x1s = 0
            x2s = (F2*c2) / (F1+F2)
        else:
            x1s = 0
            x2s = 0
        pH = cls.system_output(x1s, x2s)
        return np.array([x1s, x2s]), pH

    @classmethod
    def analytic(cls, t, x0, u):
        """ Calculates the analytic solution

           Two modes: 1. external input; 2. no external input

           Args:
               t (array_like): Independent variable, time vector
               x0 (array_like): Dependent variable, initial conditions of the output value
               u : (array_like) External variable, control input vector F2

           Returns:
               array: A vector composed of analytical solutions to the system of differential equations.
       """
        if np.shape(u)[0] == 1:
            u = np.squeeze(u)
        else:
            u = u

        F1 = cls.F1
        c1 = cls.c1
        c2 = cls.c2
        V = cls.V

        t_init = 0

        x_solve = np.zeros(shape=(cls.nx, len(t)))

        x_init = x0
        prev_x = x0
        u_prev = u[0]
        for i in range(0, len(t)):
            if u[i] != u_prev:
                # Create a new initial conditions
                x_init[0] = prev_x[0]
                x_init[1] = prev_x[1]
                t_init = t[i]

            z = (F1 + u[i]) / V
            v1 = F1 * c1 / V
            v2 = u[i] * c2 / V

            C1 = (x_init[0] - v1 * (1 / z)) * 1 / exp(-z * t_init)
            C2 = (x_init[1] - v2 * (1 / z)) * 1 / exp(-z * t_init)

            y1 = exp(-z * t[i]) * C1 + v1 * (1 / z)
            y2 = exp(-z * t[i]) * C2 + v2 * (1 / z)

            # Watch out for slicing and overwriting memory!!!
            x_solve[:, i] = [y1, y2]

            # Save the last outputs
            prev_x = [y1, y2]
            u_prev = u[i]
        return x_solve
