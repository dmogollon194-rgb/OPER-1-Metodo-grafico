import streamlit as st
import pyomo.environ as pyo


def construir_modelo():
    """
    Modelo lineal muy sencillo:

    Minimizar:   3x + 5y
    s.a.
        2x +  y >= 8
         x + 2y >= 8
        x, y >= 0
    """
    m = pyo.ConcreteModel()

    # Variables
    m.x = pyo.Var(domain=pyo.NonNegativeReals)
    m.y = pyo.Var(domain=pyo.NonNegativeReals)

    # Función objetivo
    m.obj = pyo.Objective(expr=3 * m.x + 5 * m.y, sense=pyo.minimize)

    # Restricciones
    m.c1 = pyo.Constraint(expr=2 * m.x + m.y >= 8)
    m.c2 = pyo.Constraint(expr=m.x + 2 * m.y >= 8)

    return m


def resolver_modelo():
    # Construir modelo
    m = construir_modelo()

    # Usar solver HiGHS vía appsi (no requiere ejecutable externo)
    solver = pyo.SolverFactory("appsi_highs")
    resultado = solver.solve(m)

    return m, resultado


# ===================== INTERFAZ STREAMLIT =====================

st.title("Ejemplo Pyomo + HiGHS en Streamlit")

st.markdown(
    """
Modelo de prueba (muy pequeño) resuelto con Pyomo y el solver **HiGHS**.

Se resuelve el problema:

\\[
\\min \\ 3x + 5y \\\\
\\text{s.a.} \\\\
2x + y \\ge 8 \\\\
x + 2y \\ge 8 \\\\
x, y \\ge 0
\\]
"""
)

if st.button("Resolver modelo"):
    try:
        modelo, resultado = resolver_modelo()

        st.subheader("Estado del solver")
        st.write(str(resultado.solver.termination_condition))

        st.subheader("Solución óptima")
        st.write(f"x* = {pyo.value(modelo.x):.4f}")
        st.write(f"y* = {pyo.value(modelo.y):.4f}")
        st.write(f"Valor óptimo de la función objetivo = {pyo.value(modelo.obj):.4f}")

        st.subheader("Restricciones activas")
        st.write(f"2x + y = {2 * pyo.value(modelo.x) + pyo.value(modelo.y):.4f}")
        st.write(f"x + 2y = {pyo.value(modelo.x) + 2 * pyo.value(modelo.y):.4f}")

    except Exception as e:
        st.error(f"Error al resolver el modelo: {e}")
    st.sidebar("by M.Sc. Dilan Mogollón")
