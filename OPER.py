import streamlit as st
import pyomo.environ as pyo
import numpy as np
import matplotlib.pyplot as plt
# =================== MARCA DE AGUA ===================
watermark = """
<style>
.watermark {
    position: fixed;
    top: 150px;
    right: 25px;
    opacity: 0.95;                 /* casi visible completa */
    font-size: 22px;               /* más grande */
    font-weight: 900;              /* extra bold */
    color: #ff4b4b;                /* rojo brillante Streamlit-like */
    text-shadow: 1px 1px 2px #000; /* sombra para contraste */
    z-index: 2000;                 /* por encima de todo */
}
</style>
<div class="watermark">by M.Sc. Dilan Mogollón</div>
"""
st.markdown(watermark, unsafe_allow_html=True)
# =====================================================
st.title("Método gráfico")
st.write("Esta app resuelve un problema de programación lineal con dos variables:")
st.latex(r"""
\min/\max \; Z = c_1 x + c_2 y
""")
st.write("Sujeto a restricciones lineales en \(x\) y \(y\), y muestra la región factible.")
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

    solver = pyo.SolverFactory("appsi_highs")
    resultado = solver.solve(m)
    return m, resultado


# =================== TABS: MODELO / GRÁFICA ===================
tab_modelo, tab_grafica = st.tabs(["Modelo", "Gráfica"])

# -------- TAB MODELO --------
with tab_modelo:
    st.header("Definición del modelo")

    # Tipo de problema
    tipo_problema = st.selectbox("Tipo de problema", ["Minimizar", "Maximizar"])

    # Función objetivo
    st.subheader("Función objetivo")

    col_fo_inputs, col_fo_latex = st.columns([2, 3])

    with col_fo_inputs:
        c1 = st.number_input("Coeficiente de x en Z", value=3.0)
        c2 = st.number_input("Coeficiente de y en Z", value=5.0)

    with col_fo_latex:
        sentido_tex = r"\min" if tipo_problema == "Minimizar" else r"\max"
        sign_c2 = "+" if c2 >= 0 else "-"
        abs_c2 = abs(c2)
        st.latex(
            rf"{sentido_tex}\ Z = {c1}x {sign_c2} {abs_c2}y"
        )

    st.markdown("---")

    # Número de restricciones
    st.subheader("Restricciones")
    n_restr = st.number_input(
        "Número de restricciones",
        min_value=1,
        max_value=6,
        value=2,
        step=1,
    )

    restricciones = []
    for k in range(n_restr):
        st.markdown(f"**Restricción {k+1}**")
        col_inputs, col_latex = st.columns([2, 3])

        with col_inputs:
            a = st.number_input(
                f"a{k+1} (coeficiente de x)",
                value=1.0,
                key=f"a_{k}",
            )
            b = st.number_input(
                f"b{k+1} (coeficiente de y)",
                value=1.0,
                key=f"b_{k}",
            )
            sentido = st.selectbox(
                f"Tipo {k+1}",
                ["<=", ">=", "="],
                key=f"sentido_{k}",
            )
            rhs = st.number_input(
                f"Lado derecho {k+1}",
                value=8.0,
                key=f"rhs_{k}",
            )

        with col_latex:
            sign_b = "+" if b >= 0 else "-"
            abs_b = abs(b)
            # Ej: 2x + 3y <= 8
            st.latex(
                rf"{a}x {sign_b} {abs_b}y \; {sentido} \; {rhs}"
            )

        restricciones.append((a, b, sentido, rhs))

    st.markdown("---")
    st.markdown("Variables con restricción por defecto:  \n**x ≥ 0, y ≥ 0**")

    # Botón para resolver
    if st.button("Resolver modelo"):
        try:
            modelo, resultado = construir_y_resolver_modelo(
                c1, c2, restricciones, tipo_problema
            )

            x_opt = pyo.value(modelo.x)
            y_opt = pyo.value(modelo.y)
            z_opt = pyo.value(modelo.obj)

            st.subheader("Resultado del solver")
            st.write(str(resultado.solver.termination_condition))

            st.subheader("Solución óptima")
            st.write(f"x* = {x_opt:.4f}")
            st.write(f"y* = {y_opt:.4f}")
            st.write(f"Z* = {z_opt:.4f}")

            # Guardar en session_state para usar en la pestaña de gráfica
            st.session_state["modelo_resuelto"] = True
            st.session_state["x_opt"] = x_opt
            st.session_state["y_opt"] = y_opt
            st.session_state["z_opt"] = z_opt
            st.session_state["restricciones"] = restricciones
            st.session_state["c1"] = c1
            st.session_state["c2"] = c2

        except Exception as e:
            st.error(f"Error al resolver el modelo: {e}")


# -------- TAB GRÁFICA --------
with tab_grafica:
    st.header("Gráfica de la región factible")

    if "modelo_resuelto" not in st.session_state or not st.session_state["modelo_resuelto"]:
        st.info("Primero define el modelo y pulsa **Resolver modelo** en la pestaña *Modelo*.")
    else:
        restricciones = st.session_state["restricciones"]
        x_opt = st.session_state["x_opt"]
        y_opt = st.session_state["y_opt"]
        c1 = st.session_state["c1"]
        c2 = st.session_state["c2"]

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
                if abs(a) > 1e-8:
                    x_line = rhs / a
                    ax.axvline(x_line, label=f"{a}x {sentido} {rhs}")

        # Punto óptimo
        ax.scatter([x_opt], [y_opt], s=50)
        ax.annotate(
            "Óptimo", (x_opt, y_opt),
            textcoords="offset points",
            xytext=(5, 5),
        )

        # Recta de la FO que pasa por el óptimo (solo si c2 ≠ 0)
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

