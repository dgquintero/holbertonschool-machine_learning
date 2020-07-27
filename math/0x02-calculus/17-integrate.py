#!/usr/bin/env python3
"""Calculates the integral of a polynomial"""


def poly_derivative(poly):
    """Calculates the integral of a polynomial"""
    if type(poly) is not list or len(poly) == 0:
        return None
    elif len(poly) == 1:
        return [0]
    else:
        deriv_poly = [poly[i] * i for i in range(1, len(poly))]
        if deriv_poly == 0:
            return ([0])
        return deriv_poly
