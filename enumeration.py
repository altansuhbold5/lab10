import itertools

VARS = ['B', 'E', 'A', 'J', 'M']

P_B1 = 0.001
P_E1 = 0.002
P_A1 = {(0, 0): 0.001, (0, 1): 0.29, (1, 0): 0.94, (1, 1): 0.95}
P_J1 = {0: 0.05, 1: 0.90}
P_M1 = {0: 0.01, 1: 0.70}


def bern(p1, value):
    return p1 if value == 1 else 1 - p1


def joint(b, e, a, j, m):
    return (bern(P_B1, b) * bern(P_E1, e) * bern(P_A1[(b, e)], a)
            * bern(P_J1[a], j) * bern(P_M1[a], m))


def query(var, evidence=None):
    evidence = evidence or {}
    totals = {0: 0.0, 1: 0.0}
    for values in itertools.product([0, 1], repeat=len(VARS)):
        world = dict(zip(VARS, values))
        if any(world[k] != v for k, v in evidence.items()):
            continue
        totals[world[var]] += joint(world['B'], world['E'], world['A'],
                                    world['J'], world['M'])
    z = totals[0] + totals[1]
    return {0: totals[0] / z, 1: totals[1] / z}


if __name__ == '__main__':
    cases = [
        ('Prior  P(J=1)', 'J', {}),
        ('Prior  P(M=1)', 'M', {}),
        ('Prior  P(A=1)', 'A', {}),
        ('Diag   P(B=1 | J=1)', 'B', {'J': 1}),
        ('Diag   P(B=1 | J=1, M=1)', 'B', {'J': 1, 'M': 1}),
        ('Diag   P(E=1 | J=1, M=1)', 'E', {'J': 1, 'M': 1}),
        ('Diag   P(A=1 | J=1, M=1)', 'A', {'J': 1, 'M': 1}),
        ('Diag   P(B=1 | J=1, M=0)', 'B', {'J': 1, 'M': 0}),
        ('Pred   P(J=1 | B=1)', 'J', {'B': 1}),
        ('Pred   P(A=1 | B=1)', 'A', {'B': 1}),
        ('EA     P(B=1 | A=1)', 'B', {'A': 1}),
        ('EA     P(B=1 | A=1, E=1)', 'B', {'A': 1, 'E': 1}),
        ('EA     P(B=1 | A=1, E=0)', 'B', {'A': 1, 'E': 0}),
    ]
    for label, var, ev in cases:
        print(f'{label:30s} = {query(var, ev)[1]:.6f}')