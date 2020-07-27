#!/usr/bin/env python3
"""Calculates the integral of a polynomial"""


def poly_integral(poly, C=0):
    """Calculates the integral of a polynomial"""
    # return (P.polyint(poly, C))
    # return poli_int
    if type(poly) is not list or len(poly) == 0:
        return None
    elif len(poly) == 1:
        return [0]
    elif poly == [0]:
        return [C]
    else:
        integral_poly = [C]
        [integral_poly.append(poly[i] / (i + 1)) for i in range(1, len(poly))]
        # if integral_poly == 0:
        #    return ([0])
        return integral_poly
