import pyomo.environ as pyo
from pyomo.contrib.appsi.solvers import Highs

# construir modelo
m = pyo.ConcreteModel()
# ... tus variables / restricciones / objetivo ...

solver = Highs()  # o pyo.SolverFactory("appsi_highs")
res = solver.solve(m)

print(res.termination_condition)
print(pyo.value(m.obj))
