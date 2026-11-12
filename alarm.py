try:
    from pgmpy.models import DiscreteBayesianNetwork as BN
except ImportError:
    from pgmpy.models import BayesianNetwork as BN
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

model = BN([('Burglary', 'Alarm'), ('Earthquake', 'Alarm'),
            ('Alarm', 'JohnCalls'), ('Alarm', 'MaryCalls')])

cpd_b = TabularCPD('Burglary', 2, [[0.999], [0.001]])
cpd_e = TabularCPD('Earthquake', 2, [[0.998], [0.002]])
cpd_a = TabularCPD('Alarm', 2,
                   [[0.999, 0.71, 0.06, 0.05],
                    [0.001, 0.29, 0.94, 0.95]],
                   evidence=['Burglary', 'Earthquake'], evidence_card=[2, 2])
cpd_j = TabularCPD('JohnCalls', 2,
                   [[0.95, 0.10],
                    [0.05, 0.90]],
                   evidence=['Alarm'], evidence_card=[2])
cpd_m = TabularCPD('MaryCalls', 2,
                   [[0.99, 0.30],
                    [0.01, 0.70]],
                   evidence=['Alarm'], evidence_card=[2])
model.add_cpds(cpd_b, cpd_e, cpd_a, cpd_j, cpd_m)

print('check_model():', model.check_model())
print('nodes:', model.nodes())
print('edges:', model.edges())

infer = VariableElimination(model)


def p1(var, evidence=None):
    return infer.query([var], evidence=evidence or {}).values[1]


print('\nPrior')
print('P(JohnCalls=1)  =', round(p1('JohnCalls'), 5))
print('P(MaryCalls=1)  =', round(p1('MaryCalls'), 5))
print('P(Alarm=1)      =', round(p1('Alarm'), 5))

print('\nPosterior')
print('P(B|J=1)        =', round(p1('Burglary', {'JohnCalls': 1}), 4))
print('P(B|J=1,M=1)    =', round(p1('Burglary', {'JohnCalls': 1, 'MaryCalls': 1}), 4))
print('P(E|J=1,M=1)    =', round(p1('Earthquake', {'JohnCalls': 1, 'MaryCalls': 1}), 4))
print('P(A|J=1,M=1)    =', round(p1('Alarm', {'JohnCalls': 1, 'MaryCalls': 1}), 4))
print('P(B|J=1,M=0)    =', round(p1('Burglary', {'JohnCalls': 1, 'MaryCalls': 0}), 4))
print('P(J=1|B=1)      =', round(p1('JohnCalls', {'Burglary': 1}), 4))

print('\nExplaining away')
print('P(B|A=1)        =', round(p1('Burglary', {'Alarm': 1}), 4))
print('P(B|A=1,E=1)    =', round(p1('Burglary', {'Alarm': 1, 'Earthquake': 1}), 4))
print('P(B|A=1,E=0)    =', round(p1('Burglary', {'Alarm': 1, 'Earthquake': 0}), 4))

print('\nActive trails')
print('B нотолгоогүй      :', model.active_trail_nodes('Burglary'))
print('B | Alarm=1        :', model.active_trail_nodes('Burglary', observed=['Alarm']))
print('J нотолгоогүй      :', model.active_trail_nodes('JohnCalls'))
print('J | Alarm=1        :', model.active_trail_nodes('JohnCalls', observed=['Alarm']))
print('Local independencies(JohnCalls):', model.local_independencies('JohnCalls'))