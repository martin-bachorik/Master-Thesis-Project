class P:
    def __init__(self, Zr=None):
        """ Standard Proportional-Integral-Derivative controller

        Args:
            Zr: Controller gain

        returns:
            Control input to the process: u(k) = Zre(k)
        """
        self.tag_spec = False
        self.Ts = None

        self.Zr = Zr

    def __call__(self, e):
        return self.Zr * e


class PI(P):
    def __init__(self, Zr=None, Ti=None):
        super(PI, self).__init__(Zr)
        """ Standard Proportional-Integral controller.
        
        Integration(Summation) part uses forward integration(summation). 

        Args:
            Zr: Controller gain
            Ti: Time constant of the integral part

        returns:
            Control input to the process: u(k) =Zr{e(k) + Ts/Ti[sum^k_i{e(i)}]}
        """

        self.Ti = Ti
        self.sum_e = 0  # I part summation

    def __call__(self, e):
        self.sum_e = self.sum_e + e
        return self.Zr * (e + (self.Ts / self.Ti) * self.sum_e)


class PIOver:
    def __init__(self, Zr=None, Ti=None, beta=None):
        """ Standard Proportional-Integral controller with weighted reference.

        Integration(Summation) part uses forward integration(summation).

        Args:
            Zr: Controller gain
            Ti: Time constant of the integral part

        returns:
            Control input to the process: u(k) =Zr{e(k) + Ts/Ti[sum^k_i{e(i)}]}
        """
        self.Zr = Zr
        self.Ti = Ti

        self.Ts = None
        self.sum_e = 0
        self.beta = beta

        self.tag_spec = True

    def __call__(self, e, we, ye):
        self.sum_e = self.sum_e + e
        return self.Zr * ((self.beta*we-ye) + (self.Ts / self.Ti) * self.sum_e)


class PID(PI):
    def __init__(self, Zr=None, Ti=None, Td=None):
        super(PID, self).__init__(Zr, Ti)
        """ Standard Proportional-Integral-Derivative controller

        Args:
            Zr: Controller gain
            Ti: Time constant of the integral part
            Td: Time constant of the derivative part

        returns:
            Control input to the process: u(k) =Zr{e(k) + Ts/Ti[sum^k_i{e(i)}] + Td/Ts[e(k) - e(k-1)]}
        """

        self.Td = Td
        self.prev_e = 0

    def __call__(self, e):
        self.sum_e = self.sum_e + e
        self.dev_e = e - self.prev_e  # error difference
        u = self.Zr * (e + (self.Ts / self.Ti) * self.sum_e + self.Td / self.Ts * self.dev_e)  # forward integration
        self.prev_e = e  # update previous error
        return u


class PIDBump:
    def __init__(self, Zr=None, Ti=None, Td=None):
        self.Zr = Zr
        self.Ti = Ti
        self.Td = Td

        self.Ts = None
        self.tag_spec = True

        self.sum_e = 0
        self.prev_e = 0
        self.prev_y = 0

    def __call__(self, e, we, ye):
        self.sum_e = self.sum_e + e
        self.dev_e = - (ye-self.prev_y)  # error difference
        u = self.Zr * ((0.7*we-ye) + (self.Ts / self.Ti) * self.sum_e + self.Td / self.Ts * self.dev_e)  # forward integration
        self.prev_e = e  # update previous error
        self.prev_y = ye
        return u


class PIRec(PI):
    def __init__(self, Zr=None, Ti=None):
        super(PIRec, self).__init__(Zr, Ti)
        """Proportional-Integral controller with recurrent incrementation.

        Integration(Summation) part uses forward integration(summation). 

        Args:
            Zr: Controller gain
            Ti: Time constant of the integral part

        returns:
            Control input to the process: u(k) =delta_u(k) - u(k-1)
        """
        self.prev_e = 0
        self.prev_sum_e = 0

    def __call__(self, e):
        u = self.Zr * (e - self.prev_e + (self.Ts / self.Ti) * e) + self.Zr * (
                self.prev_e + (self.Ts / self.Ti) * self.prev_sum_e)
        # u = self.Zr * e - self.prev_e + (self.Ts / self.Ti) * e + self.Zr * self.prev_e + (self.Ts / self.Ti) * self.prev_sum_e
        self.prev_sum_e = self.prev_sum_e + self.prev_e
        self.prev_e = e
        return u


class PIDRec(PID):
    def __init__(self, Zr=None, Ti=None, Td=None):
        super(PIDRec, self).__init__(Zr, Ti, Td)
        """Proportional-Integral-Derivative controller with recurrent incrementation.
        
        Integration(Summation) part uses forward integration(summation). 

        Args:
            Zr: Controller gain
            Ti: Time constant of the integral part
            Td: Time constant of the derivative part

        returns:
            Control input to the process: u(k) =delta_u(k) - u(k-1)
        """
        self.prev_prev_e = 0
        self.prev_sum_e = 0

    def __call__(self, e):
        self.sum_e = self.sum_e + e
        u = self.Zr * (e - self.prev_e + (self.Ts / self.Ti) * e + self.Td / self.Ts * (
                e - 2 * self.prev_e + self.prev_prev_e)) + \
            self.Zr * (self.prev_e + (self.Ts / self.Ti) * self.prev_sum_e + self.Td / self.Ts * (
                self.prev_e - self.prev_prev_e))
        self.prev_prev_e = self.prev_e
        self.prev_sum_e = self.prev_sum_e + self.prev_e
        self.prev_e = e  # update previous error
        return u


class PIGainSchedule:
    def __init__(self):
        self.Zr = 0
        self.Ti = 80

        self.u = 0

        self.tag_spec = True

        self.Ts = None
        self.prev_e = 0

        self.sum_e = 0  # I part summation
        self.prev_sum_e = 0

    def __call__(self, e, y):
        self.sum_e = self.sum_e + e
        if 0 <= y < 5:
            self.Zr = 0.8
        elif 5 <= y < 7:
            self.Zr = 0.15
        elif 7 <= y <= 11:
            self.Zr = 0.13
        elif 11 <= y <= 14:
            self.Zr = 0.18
        return self.Zr * (e + (self.Ts / self.Ti) * self.sum_e)


class PIRecGainSchedule:
    def __init__(self):
        self.Zr = 0
        self.Ti = 80

        self.u = 0

        self.tag_spec = True

        self.Ts = None
        self.prev_e = 0

        self.sum_e = 0  # I part summation
        self.prev_sum_e = 0

    def __call__(self, e, y):
        self.sum_e = self.sum_e + e
        if 0 <= y < 5:
            self.Zr = 0.1
        elif 5 <= y < 8:
            self.Zr = 0.1
        elif 8 <= y <= 11:
            self.Zr = 0.1
        elif 11 <= y <= 14:
            self.Zr = 0.4

        self.u = self.Zr * (e - self.prev_e + (self.Ts / self.Ti) * e) + self.Zr * (
                self.prev_e + (self.Ts / self.Ti) * self.prev_sum_e)

        self.prev_sum_e = self.prev_sum_e + self.prev_e
        self.prev_e = e
        return self.u

# Version 2
# class PID:
#     def __init__(self, Zr=None, Ti=None, Td=None):
#         """ Standard Proportional-Integral-Derivative controller
#
#         Args:
#             Zr: Controller gain
#             Ti: Time constant of the integral part
#             Td: Time constant of the derivative part
#
#         returns:
#             Control input to the process
#         """
#         self.Ts = None
#
#         self.Zr = Zr
#
#         self.Ti = Ti
#         self.sum_e = 0  # I part summation
#
#         self.Td = Td
#         self.prev_e = 0  # D part previous error
#
#     def __call__(self, e):
#         if self.Zr is not None and (self.Ti and self.Td) is None:  # P
#             return self.Zr * e
#         elif (self.Zr and self.Ti) is not None and self.Td is None:  # PI
#             self.sum_e = self.sum_e + e
#             return self.Zr * e + (self.Ts / self.Ti) * self.sum_e
#         elif (self.Zr and self.Ti and self.Td) is not None:  # PID
#             self.sum_e = self.sum_e + e
#             self.dev_e = e - self.prev_e  # error difference
#             u = self.Zr * e + (self.Ts / self.Ti) * self.sum_e + self.Td / self.Ts * self.dev_e
#             self.prev_e = e  # update previous error
#             return u

