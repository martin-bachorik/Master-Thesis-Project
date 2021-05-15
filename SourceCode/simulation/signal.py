from random import uniform
from random import gauss
from scipy import signal
import numpy as np

__all__ = ['RandStep', 'Step', 'Sawtooth']


class Step:
    def __init__(self, step_time, step=None):
        """Settings for a step sequence

        Args:
            step (list): List containing all desired step vales
            step_time: Time to perform step change
        """
        self.step_vector = step
        self.step_time = step_time
        self.ref_timer = None

    def out(self, t: any, dim=(None, None)) -> any:
        """Generate a step signal sequence

        Args:
            dim: Dimension tuple in form (samples, params)
            t: Time vector

        Returns:
            array_like: Signal sequence corresponding to the time vector.

        """
        u = np.zeros(shape=dim)
        j = 0
        for i in range(len(t)):
            if t[i] % self.step_time == 0 and t[i] != 0 and j + 1 != len(self.step_vector):
                j += 1
            u[i, :] = self.step_vector[j]
        return u


class RandStep:
    def __init__(self,  step_time, step_interval=None, n_step=None, ss=None):
        """ Settings for a random step sequence

        Args:
            step_interval (list): Probability interval <a, b>
            step_time: Time to perform step change
            n_step (int): Number of steps

        """
        self.ss = ss
        self.n_step = n_step
        self.interval = step_interval
        self.step_time = step_time

    def out(self, t: any, dim=(None, None)) -> any:
        """Generate a random sequence

        Args:
            dim: Dimension tuple in form (samples, params)
            t: Time vector

        Returns:
            array_like: Signal sequence corresponding to the time vector.
        """
        lB = self.interval[0]  # Lower Boundary
        uB = self.interval[1]  # Upper Boundary
        # Initialize random step vector each sampling period using comprehensive list.
        step_vector = [round(uniform(lB, uB), 1) for _ in range(self.n_step)]
        u = np.zeros(shape=dim)  # Initialize step control input array u.
        j = 0

        for i in range(len(t)):  # Excluding the last point
            if t[i] % self.step_time == 0 and t[i] != 0 and j+1 != len(step_vector) and i != len(t)-1:  # No last step
                j += 1

            if self.ss is not None and j == 0:
                u[i, :] = self.ss
            else:
                u[i, :] = step_vector[j]
        return u


class GaussStep:
    def __init__(self,  step_time, mu=None, sigma=None, n_step=None, ss=None):
        """ Settings for a Gauss step sequence

        Args:
            mu (float)
            sigma (float)
            step_time: Time to perform step change
            n_step (int): Number of steps

        Notes:
            Preferred signal for closed-loop control training data set.

        """
        self.ss = ss
        self.n_step = n_step
        self.mu = mu
        self.sigma = sigma
        self.step_time = step_time

    def out(self, t: any, dim=(None, None)) -> any:
        """Generate a Gauss sequence

        Args:
            dim: Dimension tuple in form (samples, params)
            t: Time vector

        Returns:
            array_like: Signal sequence corresponding to the time vector.
        """

        step_vector = np.abs([round(gauss(self.mu, self.sigma), 1) for _ in range(self.n_step)])
        u = np.zeros(shape=dim)
        j = 0

        for i in range(len(t)):  # Excluding the last point
            if t[i] % self.step_time == 0 and t[i] != 0 and j+1 != len(step_vector) and i != len(t)-1:  # No last step
                j += 1

            if self.ss is not None and j == 0:
                u[i, :] = self.ss
            else:
                u[i, :] = step_vector[j]
        return u


class Sawtooth:
    def __init__(self, n_steps, step_time, delta_t, amp=None):
        self.delta_t = delta_t
        self.n_steps = n_steps

        self.step_k = int(step_time/delta_t)
        self.step_seq = np.linspace(0, self.step_k, self.step_k)

        self.amp = amp

    def out(self, t: any, dim=(None, None)):
        u = np.zeros(shape=dim)

        for i in range(self.n_steps):
            u[i*self.step_k:(i*self.step_k+self.step_k), 0] = self.amp*signal.sawtooth(2 * np.pi * 1 * self.step_seq, width=0.5) + self.amp  # one step sequence


        return u


class SawRandStep:
    def __init__(self,  step_time, saw_time, step_interval=None, n_step=None, ss=None):
        """ Settings for a random step sequence

        Args:
            step_interval (list): Probability interval <a, b>
            step_time: Time to perform step change
            n_step (int): Number of steps

        """
        self.ss = ss
        self.n_step = n_step
        self.interval = step_interval
        self.step_time = step_time
        self.saw_time = saw_time

    def out(self, t: any, dim=(None, None)) -> any:
        """Generate a random sequence

        Args:
            dim: Dimension tuple in form (samples, params)
            t: Time vector

        Returns:
            array_like: Signal sequence corresponding to the time vector.
        """
        lB = self.interval[0]  # Lower Boundary
        uB = self.interval[1]  # Upper Boundary
        # Initialize random step vector each sampling period using comprehensive list.
        step_vector = [round(uniform(lB, uB), 1) for _ in range(self.n_step)]
        step_vector[0] = self.ss  # keep the steady state value as first
        u = np.zeros(shape=dim)  # Initialize step control input array u.
        j = 0

        ramp_Step = self.saw_time
        count = 1
        for i in range(len(t)):  # Excluding the last point
            if t[i] % self.step_time == 0 and t[i] != 0 and j+1 != len(step_vector) and i != len(t)-1:  # No last step
                j += 1
                count = 1

            if self.ss is not None and j == 0:
                u[i, :] = self.ss
            else:
                if count != ramp_Step:

                    u[i, :] = (step_vector[j] - step_vector[j-1]) * (count / ramp_Step) + step_vector[j-1]
                    count += 1
                else:
                    u[i, :] = step_vector[j]
        return u


class SawGaussStep:
    def __init__(self, step_time, saw_time, delta_t, mu=None, sigma=None, n_step=None, ss=None):
        """ Settings for a random step sequence

        Args:
            step_time: Time to perform step change
            n_step (int): Number of steps

        """
        self.ss = ss
        self.n_step = n_step

        self.mu = mu
        self.sigma = sigma

        self.step_time = step_time
        self.saw_time = saw_time / delta_t

    def out(self, t: any, dim=(None, None)) -> any:
        """Generate a random sequence

        Args:
            dim: Dimension tuple in form (samples, params)
            t: Time vector

        Returns:
            array_like: Signal sequence corresponding to the time vector.
        """

        # Initialize random step vector each sampling period using comprehensive list.
        step_vector = np.abs([round(gauss(self.mu, self.sigma), 1) for _ in range(self.n_step)])
        step_vector[0] = self.ss  # keep the steady state value as first
        u = np.zeros(shape=dim)  # Initialize step control input array u.
        j = 0

        ramp_Step = self.saw_time
        count = 1
        for i in range(len(t)):  # Excluding the last point
            if t[i] % self.step_time == 0 and t[i] != 0 and j + 1 != len(step_vector) and i != len(
                    t) - 1:  # No last step
                j += 1
                count = 1

            if self.ss is not None and j == 0:
                u[i, :] = self.ss
            else:
                if count != ramp_Step:

                    u[i, :] = (step_vector[j] - step_vector[j - 1]) * (count / ramp_Step) + step_vector[j - 1]
                    count += 1
                else:
                    u[i, :] = step_vector[j]
        return u