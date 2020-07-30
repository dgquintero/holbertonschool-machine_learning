#!/usr/bin/env python3
"""class Poisson that represents a poisson distribution"""


class Poisson():
    """ class Poisson """
    e = 2.7182818285

    def __init__(self, data=None, lambtha=1.):
        """ class constructor to call """

        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)
        else:
            if type(data) is not list:
                raise TypeError("data must be a list")
            elif len(data) < 2:
                raise ValueError("data must contain multiple values")
            else:
                self.lambtha = (sum(data) / len(data))

    def pmf(self, k):
        """ Poisson probability pmf """
        factorial_k = 1
        k = int(k)
        if k < 0:
            return 0
        for i in range(1, k + 1):
            factorial_k *= i
        pmf_p = (Poisson.e ** (-self.lambtha)) * (self.lambtha ** k)\
            / factorial_k

        return pmf_p

    def cdf(self, k):
        """ Poisson probability cdf """
        k = int(k)
        if k < 0:
            return 0
        cdf_p = 0
        for i in range(k + 1):
            cdf_p += self.pmf(i)
        return cdf_p
