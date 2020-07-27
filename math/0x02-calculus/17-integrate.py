#!/usr/bin/env python3
from numpy.polynomial import polynomial as P
"""Calculates the integral of a polynomial"""


def poly_integral(poly, C=0):
    """Calculates the integral of a polynomial"""
    P.polyint(poly, C)
    #return poli_int
