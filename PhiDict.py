import numpy as np
PHI_functions = {
    0: {
        "PHI": lambda ξ:np.array([
             1
        ]),
        "dPHI": lambda ξ:np.array([
             0
        ]),
        "CRT": lambda a, b: np.array([[
             1
        ]]),
    },
    1: {
        "PHI": lambda ξ: np.array([
             -ξ/2 + 1/2,
             +ξ/2 + 1/2,
        ]),
        "dPHI": lambda ξ:np.array([
             -1/2,
             +1/2,
        ]),
        "CRT": lambda a, b: np.array([
            [2 - b,   - a],  
            [  - b, 2 - a]
        ])/(2 - a - b)
    },
    2: {
        "PHI": lambda ξ: np.array([
             ξ**2/2 - ξ/2,
            -ξ**2   +   1,
             ξ**2/2 + ξ/2,
        ]),
        "dPHI": lambda ξ:np.array([
               ξ - 1/2,
            -2*ξ,
               ξ + 1/2,
        ]),
        "CRT": lambda a, b: np.array([
             [(-a*b + 2.0*a + b**2 - 4.0*b + 4.0),
              (-a*b + a + b**2 - b),
              (-a*b + b**2 + 2.0*b)],
             [(4.0*a*b - 8.0*a),
              (4.0*a*b - 4.0*a - 4.0*b + 4.0),
              (4.0*a*b - 8.0*b)],
             [(a**2 - a*b + 2.0*a),
              (a**2 - a*b - a + b),
              (a**2 - a*b - 4.0*a + 2.0*b + 4.0)]
        ])/(2 - a - b)**2
    },
    #4: {
    #    "PHI": lambda ξ: np.array([
    #        -0.5625*ξ**3 + 0.5625*ξ**2 + 0.0625*ξ - 0.0625,
    #         1.6875*ξ**3 - 0.5625*ξ**2 - 1.6875*ξ + 0.5625,
    #        -1.6875*ξ**3 - 0.5625*ξ**2 + 1.6875*ξ + 0.5625,
    #         0.5625*ξ**3 + 0.5625*ξ**2 - 0.0625*ξ - 0.0625
    #    ]),
    #    "CRT": lambda a, b: np.array([
    #         [(-a*b + 2.0*a + b**2 - 4.0*b + 4.0),
    #          (-a*b + a + b**2 - b),
    #          (-a*b + b**2 + 2.0*b)],
    #         [(4.0*a*b - 8.0*a),
    #          (4.0*a*b - 4.0*a - 4.0*b + 4.0),
    #          (4.0*a*b - 8.0*b)],
    #         [(a**2 - a*b + 2.0*a),
    #          (a**2 - a*b - a + b),
    #          (a**2 - a*b - 4.0*a + 2.0*b + 4.0)]
    #    ])/(2 - a - b)**2
    #},
    #5: {
    #    "PHI": [
    #        lambda ξ:  0.666666666666667*ξ**4 - 0.666666666666667*ξ**3 - 0.166666666666667*ξ**2 + 0.166666666666667*ξ,
    #        lambda ξ: -2.66666666666667*ξ**4 + 1.33333333333333*ξ**3 + 2.66666666666667*ξ**2 - 1.33333333333333*ξ,
    #        lambda ξ:  4.0*ξ**4 - 5.0*ξ**2 + 1.0,
    #        lambda ξ: -2.66666666666667*ξ**4 - 1.33333333333333*ξ**3 + 2.66666666666667*ξ**2 + 1.33333333333333*ξ,
    #        lambda ξ:  0.666666666666667*ξ**4 + 0.666666666666667*ξ**3 - 0.166666666666667*ξ**2 - 0.166666666666667*ξ
    #    ],
    #}
}