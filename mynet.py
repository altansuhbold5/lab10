try:
    from pgmpy.models import DiscreteBayesianNetwork as BN
except ImportError:
    from pgmpy.models import BayesianNetwork as BN
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

model = BN([('Sleep', 'Exam'), ('Study', 'Exam'), ('Exam', 'Pass')])

cpd_s = TabularCPD('Sleep', 2, [[0.4], [0.6]])
cpd_t = TabularCPD('Study', 2, [[0.5], [0.5]])
cpd_e = TabularCPD('Exam', 2,
                   [[0.9, 0.6, 0.5, 0.1],
                    [0.1, 0.4, 0.5, 0.9]],
                   evidence=['Sleep', 'Study'], evidence_card=[2, 2])
cpd_p = TabularCPD('Pass', 2,
                   [[0.8, 0.05],
                    [0.2, 0.95]],
                   evidence=['Exam'], evidence_card=[2])
model.add_cpds(cpd_s, cpd_t, cpd_e, cpd_p)

print('check_model():', model.check_model())
infer = VariableElimination(model)


def p1(var, evidence=None):
    return infer.query([var], evidence=evidence or {}).values[1]


print('P(Exam=1)                    =', round(p1('Exam'), 4))
print('P(Study=1 | Exam=1)          =', round(p1('Study', {'Exam': 1}), 4))
print('P(Study=1 | Exam=1, Sleep=1) =', round(p1('Study', {'Exam': 1, 'Sleep': 1}), 4))
print('P(Study=1 | Exam=1, Sleep=0) =', round(p1('Study', {'Exam': 1, 'Sleep': 0}), 4))
print('P(Pass=1)                    =', round(p1('Pass'), 4))

print('\nActive trails')
print('Sleep нотолгоогүй : ', model.active_trail_nodes('Sleep'))
print('Sleep | Exam=1   : ', model.active_trail_nodes('Sleep', observed=['Exam']))