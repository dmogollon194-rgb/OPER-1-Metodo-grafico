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
    opacity: 0.95;
    font-size: 22px;
    font-weight: 900;
    color: #ff4b4b;
    text-shadow: 1px 1px 2px #000;
    z-index: 2000;
}
</style>
<div class="watermark">by M.Sc. Dilan Mogollón</div>
"""
st.markdown(watermark, unsafe_allow_html=True)


# =================== AUX: DOMINIO SEGÚN NATURALEZA ===================
def obtener_dominio(tipo):
    if tipo == "Real ≥ 0":
        return pyo.NonNegativeReals
    elif tipo == "Entera ≥ 0":
        return pyo.NonNegativeIntegers
    elif tipo == "Binaria":
        return pyo.Binary
# =================== FUNCIÓN PARA CONSTRUIR Y RESOLVER ===================
def construir_y_resolver_modelo(c1, c2, restricciones, tipo_problema, tipo_x, tipo_y):
    m = pyo.ConcreteModel()

    # Variables según naturaleza elegida
    dom_x = obtener_dominio(tipo_x)
    dom_y = obtener_dominio(tipo_y)

    m.x = pyo.Var(domain=dom_x)
    m.y = pyo.Var(domain=dom_y)

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
    # Título grande y centrado
    st.markdown(
        "<h2 style='text-align:center; margin-top:0;'>Definición del modelo</h2>",
        unsafe_allow_html=True
    )
    st.markdown("---")

    # Tipo de problema
    st.subheader("Tipo de problema", )
    tipo_problema = st.selectbox("Seleccione: ",["Minimizar", "Maximizar"])
    # Naturaleza de las variables
    st.subheader("Naturaleza de las variables")
    col_nat_x, col_nat_y = st.columns(2)

    with col_nat_x:
        tipo_x = st.selectbox(
            "Naturaleza de X",
            ["Real ≥ 0", "Entera ≥ 0", "Binaria"],
            key="tipo_x"
        )
    with col_nat_y:
        tipo_y = st.selectbox(
            "Naturaleza de Y",
            ["Real ≥ 0", "Entera ≥ 0", "Binaria"],
            key="tipo_y"
        )
    # Función objetivo
    st.subheader("Función objetivo")

    col_fo_inputs, col_fo_latex = st.columns([2, 3])

    with col_fo_inputs:
        c1 = st.number_input("Coeficiente de X", value=3.0)
        c2 = st.number_input("Coeficiente de Y", value=5.0)

    with col_fo_latex:
        sentido_tex = r"\min" if tipo_problema == "Minimizar" else r"\max"
        sign_c2 = "+" if c2 >= 0 else "-"
        abs_c2 = abs(c2)
        st.latex(rf"{sentido_tex}\ Z = {c1}x {sign_c2} {abs_c2}y")

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
        # Título de la restricción y línea separadora
        st.markdown("---")
        st.markdown(
            f"<h4 style='margin-bottom:0;'>Restricción {k+1}</h4>",
            unsafe_allow_html=True
        )

        # Una sola fila con todos los inputs
        col_a, col_b, col_sent, col_rhs = st.columns([1, 1, 1, 1])

        with col_a:
            a = st.number_input(
                f"a{k+1} (coef. x)",
                value=1.0,
                key=f"a_{k}",
            )
        with col_b:
            b = st.number_input(
                f"b{k+1} (coef. y)",
                value=1.0,
                key=f"b_{k}",
            )
        with col_sent:
            sentido = st.selectbox(
                f"Tipo {k+1}",
                ["<=", ">=", "="],
                key=f"sentido_{k}",
            )
        with col_rhs:
            rhs = st.number_input(
                f"Lado derecho {k+1}",
                value=8.0,
                key=f"rhs_{k}",
            )

        # Ecuación en LaTeX debajo de la fila de inputs
        sign_b = "+" if b >= 0 else "-"
        abs_b = abs(b)
        st.latex(rf"{a}x {sign_b} {abs_b}y \; {sentido} \; {rhs}")

        restricciones.append((a, b, sentido, rhs))

    st.markdown("---")
    st.markdown(
        "Por defecto se grafica en el primer cuadrante; si eliges variables libres "
        "la gráfica puede mostrar parte de la región fuera de x,y ≥ 0."
    )

    # Botón para resolver
    if st.button("Resolver modelo"):
        try:
            modelo, resultado = construir_y_resolver_modelo(
                c1, c2, restricciones, tipo_problema, tipo_x, tipo_y
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

            # Guardar en session_state para la pestaña Gráfica
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
    st.markdown(
        "<h2 style='text-align:center; margin-top:0;'>Gráfica de la región factible</h2>",
        unsafe_allow_html=True
    )
    st.markdown("---")

    if "modelo_resuelto" not in st.session_state or not st.session_state["modelo_resuelto"]:
        st.info("Primero define el modelo y pulsa **Resolver modelo** en la pestaña *Modelo*.")
    else:
        import plotly.graph_objects as go
        import plotly.express as px

        restricciones = st.session_state["restricciones"]
        x_opt = st.session_state["x_opt"]
        y_opt = st.session_state["y_opt"]
        c1 = st.session_state["c1"]
        c2 = st.session_state["c2"]

        # Determinar límites aproximados
        max_rhs = max([abs(r[3]) for r in restricciones] + [10])
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
            else:
                factible &= np.isclose(a * XX + b * YY, rhs, atol=1e-3)

        # Crear figura interactiva
        fig = go.Figure()

        # Región factible (semi-transparente)
        fig.add_trace(go.Contour(
            x=X,
            y=Y,
            z=factible.astype(int),
            showscale=False,
            colorscale=[[0, 'rgba(0,0,0,0)'], [1, 'rgba(0,150,255,0.3)']],
            opacity=0.4,
            name="Región factible"
        ))

        # Líneas de restricciones
        for (a, b, sentido, rhs) in restricciones:
            if abs(b) > 1e-8:
                y_line = (rhs - a * X) / b
                fig.add_trace(go.Scatter(
                    x=X,
                    y=y_line,
                    mode="lines",
                    name=f"{a}x + {b}y {sentido} {rhs}"
                ))
            else:
                # Recta vertical
                if abs(a) > 1e-8:
                    x_line = rhs / a
                    fig.add_trace(go.Scatter(
                        x=[x_line, x_line],
                        y=[0, lim],
                        mode="lines",
                        name=f"{a}x {sentido} {rhs}"
                    ))

        # Punto óptimo
        fig.add_trace(go.Scatter(
            x=[x_opt],
            y=[y_opt],
            mode="markers+text",
            text=["Óptimo"],
            textposition="top right",
            marker=dict(size=10, color="red"),
            name="Solución óptima"
        ))

        # Recta de la FO pasando por el óptimo
        if abs(c2) > 1e-8:
            z_opt = c1 * x_opt + c2 * y_opt
            y_obj = (z_opt - c1 * X) / c2
            fig.add_trace(go.Scatter(
                x=X,
                y=y_obj,
                mode="lines",
                line=dict(dash="dash", color="red"),
                name="FO en Z*"
            ))

        # Ajustar zoom a la región visible
        xs = XX[factible]
        ys = YY[factible]
        if len(xs) > 0:
            fig.update_xaxes(range=[max(0, xs.min() - 1), xs.max() + 1])
            fig.update_yaxes(range=[max(0, ys.min() - 1), ys.max() + 1])
        else:
            fig.update_xaxes(range=[0, lim])
            fig.update_yaxes(range=[0, lim])

        fig.update_layout(
            width=800,
            height=600,
            title="Región factible y solución óptima (interactivo)",
            xaxis_title="x",
            yaxis_title="y",
            legend=dict(x=0.7, y=1.0)
        )

        st.plotly_chart(fig, use_container_width=True)
