import streamlit as st
import pyomo.environ as pyo
import numpy as np
import matplotlib.pyplot as plt


# =================== MARCA DE AGUA ===================
watermark = """
<style>
.watermark {
    position: fixed;
    top: 15px;
    right: -200px;
    opacity: 0.35;
    font-size: 18px;
    font-weight: bold;
    color: #888888;
    z-index: 1000;
}
</style>

<div class="watermark">by M.Sc. Dilan Mogollón</div>
"""

st.markdown(watermark, unsafe_allow_html=True)
# =====================================================


st.title("Método gráfico (2 variables) con Pyomo")

st.markdown(
    """
Esta app resuelve un problema de programación lineal con dos variables:

\\[
\\min/\\max \\ Z = c_1 x + c_2 y
\\]

sujeto a restricciones lineales en \\(x, y\\), y muestra la región factible.
"""
)

# =================== SIDEBAR: DATOS DEL MODELO ===================

st.sidebar.header("Datos del modelo")

# Tipo de problema
tipo_problema = st.sidebar.selectbox("Tipo de problema", ["Minimizar", "Maximizar"])

# Coeficientes de la función objetivo
c1 = st.sidebar.number_input("Coeficiente de x en Z", value=3.0)
c2 = st.sidebar.number_input("Coeficiente de y en Z", value=5.0)

# Número de restricciones
n_restr = st.sidebar.number_input(
    "Número de restricciones",
    min_value=1,
    max_value=6,
    value=2,
    step=1,
)

# Lectura de restricciones
restricciones = []
for k in range(n_restr):
    st.sidebar.subheader(f"Restricción {k+1}")
    a = st.sidebar.number_input(
        f"a{k+1} (coeficiente de x)",
        value=1.0,
        key=f"a_{k}",
    )
    b = st.sidebar.number_input(
        f"b{k+1} (coeficiente de y)",
        value=1.0,
        key=f"b_{k}",
    )
    sentido = st.sidebar.selectbox(
        f"Tipo {k+1}",
        ["<=", ">=", "="],
        key=f"sentido_{k}",
    )
    rhs = st.sidebar.number_input(
        f"Lado derecho {k+1}",
        value=8.0,
        key=f"rhs_{k}",
    )
    restricciones.append((a, b, sentido, rhs))

st.sidebar.markdown("---")
st.sidebar.markdown("Variables con restricción por defecto:  \n**x ≥ 0, y ≥ 0**")


# =================== FUNCIÓN PARA CONSTRUIR Y RESOLVER ===================

def construir_y_resolver_modelo(c1, c2, restricciones, tipo_problema):
    m = pyo.ConcreteModel()

    # Variables
    m.x = pyo.Var(domain=pyo.NonNegativeReals)
    m.y = pyo.Var(domain=pyo.NonNegativeReals)

    # Función objetivo
    if tipo_problema == "Minimizar":
        m.obj = pyo.Objective(expr=c1 * m.x + c2 * m.y, sense=pyo.minimize)
    else:
        m.obj = pyo.Objective(expr=c1 * m.x + c2 * m.y, sense=pyo.maximize)

    # Restricciones
    m.cons = pyo.ConstraintList()
    for (a, b, sentido, rhs) in restricciones:
        if sentido == "<=":
            m.cons.add(a * m.x + b * m.y <= rhs)
        elif sentido == ">=":
            m.cons.add(a * m.x + b * m.y >= rhs)
        else:  # "="
            m.cons.add(a * m.x + b * m.y == rhs)

    # Solver HiGHS (no requiere ejecutable externo)
    solver = pyo.SolverFactory("appsi_highs")
    resultado = solver.solve(m)

    return m, resultado


# =================== BOTÓN PRINCIPAL ===================

if st.button("Resolver y graficar"):
    try:
        modelo, resultado = construir_y_resolver_modelo(
            c1, c2, restricciones, tipo_problema
        )

        st.subheader("Estado del solver")
        st.write(str(resultado.solver.termination_condition))

        x_opt = pyo.value(modelo.x)
        y_opt = pyo.value(modelo.y)
        z_opt = pyo.value(modelo.obj)

        st.subheader("Solución óptima")
        st.write(f"x* = {x_opt:.4f}")
        st.write(f"y* = {y_opt:.4f}")
        st.write(f"Z* = {z_opt:.4f}")

        # =================== GRÁFICA DE LA REGIÓN FACTIBLE ===================

        # Escala aproximada según RHS
        max_rhs = max([abs(r[3]) for r in restricciones] + [1.0])
        lim = max_rhs * 1.2

        X = np.linspace(0, lim, 400)
        Y = np.linspace(0, lim, 400)
        XX, YY = np.meshgrid(X, Y)

        # Matriz booleana de factibilidad
        factible = np.ones_like(XX, dtype=bool)

        for (a, b, sentido, rhs) in restricciones:
            if sentido == "<=":
                factible &= (a * XX + b * YY <= rhs + 1e-9)
            elif sentido == ">=":
                factible &= (a * XX + b * YY >= rhs - 1e-9)
            else:  # "="
                factible &= np.isclose(a * XX + b * YY, rhs, atol=1e-3)

        fig, ax = plt.subplots()

        # Región factible
        ax.contourf(
            XX,
            YY,
            factible,
            levels=[0, 0.5, 1],
            alpha=0.3,
        )

        # Líneas de las restricciones
        for (a, b, sentido, rhs) in restricciones:
            if abs(b) > 1e-8:
                y_line = (rhs - a * X) / b
                ax.plot(X, y_line, label=f"{a}x + {b}y {sentido} {rhs}")
            else:
                # Recta vertical: x = rhs/a
                if abs(a) > 1e-8:
                    x_line = rhs / a
                    ax.axvline(x_line, label=f"{a}x {sentido} {rhs}")

        # Punto óptimo
        ax.scatter([x_opt], [y_opt], s=50)
        ax.annotate("Óptimo", (x_opt, y_opt), textcoords="offset points",
                    xytext=(5, 5))

        # Recta de la FO que pasa por el óptimo
        if abs(c2) > 1e-8:
            z_opt = c1 * x_opt + c2 * y_opt
            y_obj = (z_opt - c1 * X) / c2
            ax.plot(X, y_obj, linestyle="--", label="FO en Z*")

        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title("Región factible y solución óptima")
        ax.legend(loc="upper right", fontsize=7)

        st.pyplot(fig)

    except Exception as e:
        st.error(f"Error al resolver o graficar el modelo: {e}")
