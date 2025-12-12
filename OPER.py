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

# =================== CSS ESTÉTICO GLOBAL ===================
css = """
<style>
section[data-testid="stSidebar"] {
    width: 320px !important;
    font-size: 18px !important;
    padding: 20px !important;
}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    font-size: 28px !important;
    font-weight: 900 !important;
}
section[data-testid="stSidebar"] label {
    font-size: 18px !important;
}
.main-container {
    max-width: 1100px;
    margin-left: auto;
    margin-right: auto;
}
h2 {
    font-size: 36px !important;
    text-align: center !important;
    font-weight: 900 !important;
}
h3, h4 {
    font-size: 26px !important;
    font-weight: 700 !important;
}
div[data-baseweb="select"] > div {
    font-size: 18px !important;
}
input[type="number"] {
    font-size: 18px !important;
}
hr {
    border: 0;
    border-top: 1px solid #444;
    margin: 25px 0;
}
.block-container {
    padding-top: 1rem !important;
    padding-bottom: 3rem !important;
}
</style>
"""
st.markdown(css, unsafe_allow_html=True)

# Contenedor principal estilizado
st.markdown('<div class="main-container">', unsafe_allow_html=True)

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

    # Suffix para precios sombra
    m.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)

    m.x = pyo.Var(domain=obtener_dominio(tipo_x))
    m.y = pyo.Var(domain=obtener_dominio(tipo_y))

    if tipo_problema == "Minimizar":
        m.obj = pyo.Objective(expr=c1 * m.x + c2 * m.y, sense=pyo.minimize)
    else:
        m.obj = pyo.Objective(expr=c1 * m.x + c2 * m.y, sense=pyo.maximize)

    m.cons = pyo.ConstraintList()
    for (a, b, sentido, rhs) in restricciones:
        if sentido == "<=":
            m.cons.add(a * m.x + b * m.y <= rhs)
        elif sentido == ">=":
            m.cons.add(a * m.x + b * m.y >= rhs)
        else:
            m.cons.add(a * m.x + b * m.y == rhs)

    solver = pyo.SolverFactory("appsi_highs")
    resultado = solver.solve(m, load_solutions=True)

    return m, resultado

# =================== SIDEBAR: NAVEGACIÓN ===================
st.sidebar.title("Navegación")
vista = st.sidebar.radio("Vista", ["Modelo", "Gráfica"])

# =================== VISTA MODELO ===================
if vista == "Modelo":

    st.markdown("<h2>Definición del modelo</h2>", unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)

    # Tipo de problema
    st.subheader("Tipo de problema")
    tipo_problema = st.selectbox(
        "Seleccione:",
        ["Minimizar", "Maximizar"],
        key="tipo_problema"
    )

    # Naturaleza variables
    st.subheader("Naturaleza de las variables")
    col1, col2 = st.columns(2)

    with col1:
        tipo_x = st.selectbox(
            "Naturaleza de X",
            ["Real ≥ 0", "Entera ≥ 0", "Binaria"],
            key="tipo_x"
        )

    with col2:
        tipo_y = st.selectbox(
            "Naturaleza de Y",
            ["Real ≥ 0", "Entera ≥ 0", "Binaria"],
            key="tipo_y"
        )

    # Función objetivo
    st.subheader("Función objetivo")
    col_fo_inputs, col_fo_latex = st.columns([2, 3])

    with col_fo_inputs:
        c1 = st.number_input("Coeficiente de X", value=3.0, key="c1")
        c2 = st.number_input("Coeficiente de Y", value=5.0, key="c2")

    sentido_tex = r"\min" if tipo_problema == "Minimizar" else r"\max"
    st.latex(rf"{sentido_tex}\ Z = {c1}x + {c2}y")

    st.markdown("<hr>", unsafe_allow_html=True)

    # Restricciones
    st.subheader("Restricciones")
    n_restr = st.number_input(
        "Número de restricciones",
        min_value=1,
        max_value=6,
        value=2,
        key="n_restr"
    )

    restricciones = []
    for k in range(n_restr):

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown(f"<h4>Restricción {k+1}</h4>", unsafe_allow_html=True)

        col_a, col_b, col_sent, col_rhs = st.columns(4)

        with col_a:
            a = st.number_input(
                f"Coeficiente de X en R{k+1}",
                value=1.0,
                key=f"a{k}"
            )

        with col_b:
            b = st.number_input(
                f"Coeficiente de Y en R{k+1}",
                value=1.0,
                key=f"b{k}"
            )

        with col_sent:
            sentido = st.selectbox(
                f"Sentido en R{k+1}",
                ["<=", ">=", "="],
                key=f"sent{k}"
            )

        with col_rhs:
            rhs = st.number_input(
                f"LD en R{k+1}",
                value=8.0,
                key=f"rhs{k}"
            )

        st.latex(rf"{a}x + {b}y\ {sentido}\ {rhs}")
        restricciones.append((a, b, sentido, rhs))

    # =================== BOTÓN RESOLVER (solo calcula y guarda) ===================
    boton_resolver = st.button("Resolver modelo")

    if boton_resolver:
        try:
            modelo, resultado = construir_y_resolver_modelo(
                c1, c2, restricciones, tipo_problema, tipo_x, tipo_y
            )

            x_opt = pyo.value(modelo.x)
            y_opt = pyo.value(modelo.y)
            z_opt = pyo.value(modelo.obj)
            status = str(resultado.solver.termination_condition)

            # Precios sombra
            duales = []
            dualDisponible = (tipo_x == "Real ≥ 0" and tipo_y == "Real ≥ 0")

            if dualDisponible:
                for i, cons in enumerate(modelo.cons.values(), start=1):
                    dual_val = modelo.dual.get(cons, 0)
                    duales.append([f"Restricción {i}", dual_val])
            else:
                for i in range(len(restricciones)):
                    duales.append(
                        [f"Restricción {i+1}",
                         "No disponible (modelo entero/binario)"]
                    )

            # Guardar en session_state
            st.session_state["modelo_resuelto"] = True
            st.session_state["solver_status"] = status
            st.session_state["x_opt"] = x_opt
            st.session_state["y_opt"] = y_opt
            st.session_state["z_opt"] = z_opt
            st.session_state["restricciones"] = restricciones
            st.session_state["c1"] = c1
            st.session_state["c2"] = c2
            st.session_state["duales"] = duales

        except Exception as e:
            st.error(f"Error: {e}")

    # =================== MOSTRAR RESULTADOS SI YA HAY SOLUCIÓN ===================
    if st.session_state.get("modelo_resuelto", False):
        st.subheader("Resultado del solver")
        st.write(st.session_state.get("solver_status", ""))

        st.subheader("Solución óptima")
        st.write(f"x* = {st.session_state['x_opt']:.4f}")
        st.write(f"y* = {st.session_state['y_opt']:.4f}")
        st.write(f"Z* = {st.session_state['z_opt']:.4f}")

        st.subheader("Precios sombra (valores duales)")
        st.table(st.session_state.get("duales", []))

# =================== VISTA GRÁFICA ===================
if vista == "Gráfica":

    st.markdown("<h2>Gráfica de la región factible</h2>", unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)

    if not st.session_state.get("modelo_resuelto", False):
        st.info("Primero define y resuelve el modelo en la vista 'Modelo'.")
    else:
        import plotly.graph_objects as go

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
        for a, b, s, rhs in restricciones:
            if s == "<=":
                factible &= (a * XX + b * YY <= rhs + 1e-9)
            elif s == ">=":
                factible &= (a * XX + b * YY >= rhs - 1e-9)
            else:
                factible &= np.isclose(a * XX + b * YY, rhs, atol=1e-3)

        fig = go.Figure()

        # Región factible
        fig.add_trace(go.Contour(
            x=X,
            y=Y,
            z=factible.astype(int),
            showscale=False,
            colorscale=[[0, 'rgba(0,0,0,0)'], [1, 'rgba(0,150,255,0.3)']],
            opacity=0.4,
            name="Región factible"
        ))

        # Líneas de las restricciones
        for (a, b, s, rhs) in restricciones:
            if abs(b) > 1e-8:
                y_line = (rhs - a * X) / b
                fig.add_trace(go.Scatter(
                    x=X,
                    y=y_line,
                    mode="lines",
                    name=f"{a}x + {b}y {s} {rhs}"
                ))
            else:
                if abs(a) > 1e-8:
                    x_line = rhs / a
                    fig.add_trace(go.Scatter(
                        x=[x_line, x_line],
                        y=[0, lim],
                        mode="lines",
                        name=f"{a}x {s} {rhs}"
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

        # Zoom automático sobre región factible
        xs = XX[factible]
        ys = YY[factible]
        if xs.size > 0:
            x_min = max(0, xs.min() - 1)
            x_max = xs.max() + 1
            y_min = max(0, ys.min() - 1)
            y_max = ys.max() + 1
            fig.update_xaxes(range=[x_min, x_max])
            fig.update_yaxes(range=[y_min, y_max])
        else:
            fig.update_xaxes(range=[0, lim])
            fig.update_yaxes(range=[0, lim])

        fig.update_layout(
            width=800,
            height=600,
            title="Región factible y solución óptima",
            xaxis_title="x",
            yaxis_title="y",
            legend=dict(x=0.7, y=1.0)
        )

        st.plotly_chart(fig, use_container_width=True)

# Cerrar contenedor principal
st.markdown("</div>", unsafe_allow_html=True)
