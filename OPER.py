import streamlit as st
import pyomo.environ as pyo
import numpy as np

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
    font-size: 26px !important;
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

# =================== ENUMERAR VÉRTICES (para sensibilidad FO) ===================
def enumerar_vertices(restricciones, tipo_x, tipo_y, tol=1e-7):
    # Incluye no negatividad si aplica
    todas = list(restricciones)
    if tipo_x in ["Real ≥ 0", "Entera ≥ 0"]:
        todas.append((1.0, 0.0, ">=", 0.0))  # x >= 0
    if tipo_y in ["Real ≥ 0", "Entera ≥ 0"]:
        todas.append((0.0, 1.0, ">=", 0.0))  # y >= 0

    def es_factible(x, y):
        if tipo_x in ["Real ≥ 0", "Entera ≥ 0"] and x < -tol:
            return False
        if tipo_y in ["Real ≥ 0", "Entera ≥ 0"] and y < -tol:
            return False
        for (a, b, sentido, rhs) in restricciones:
            val = a * x + b * y
            if sentido == "<=" and val > rhs + tol:
                return False
            elif sentido == ">=" and val < rhs - tol:
                return False
            elif sentido == "=" and abs(val - rhs) > tol:
                return False
        return True

    vertices = []
    n = len(todas)
    for i in range(n):
        a1, b1, s1, rhs1 = todas[i]
        for j in range(i+1, n):
            a2, b2, s2, rhs2 = todas[j]
            det = a1*b2 - a2*b1
            if abs(det) < tol:
                continue
            x = (rhs1*b2 - rhs2*b1) / det
            y = (a1*rhs2 - a2*rhs1) / det
            if es_factible(x, y):
                vertices.append((x, y))

    uniq = {}
    for x, y in vertices:
        key = (round(x, 6), round(y, 6))
        uniq[key] = (x, y)
    return list(uniq.values())

# =================== RANGOS DE COEFICIENTES FO ===================
def rangos_coeficientes(vertices, x_opt, y_opt, c1, c2, tipo_problema):
    if len(vertices) == 0:
        return None

    otros = []
    for vx, vy in vertices:
        if abs(vx - x_opt) > 1e-6 or abs(vy - y_opt) > 1e-6:
            otros.append((vx, vy))

    if len(otros) == 0:
        return {"c1": (-np.inf, np.inf), "c2": (-np.inf, np.inf)}

    c1_min, c1_max = -np.inf, np.inf
    c2_min, c2_max = -np.inf, np.inf
    minimiza = (tipo_problema == "Minimizar")

    for (vx, vy) in otros:
        dx = x_opt - vx
        dy = y_opt - vy

        if minimiza:
            # c1*dx + c2*dy <= 0
            if abs(dx) > 1e-9:
                bound = -c2*dy/dx
                if dx > 0:
                    c1_max = min(c1_max, bound)
                else:
                    c1_min = max(c1_min, bound)
            if abs(dy) > 1e-9:
                bound = -c1*dx/dy
                if dy > 0:
                    c2_max = min(c2_max, bound)
                else:
                    c2_min = max(c2_min, bound)
        else:
            # c1*dx + c2*dy >= 0
            if abs(dx) > 1e-9:
                bound = -c2*dy/dx
                if dx > 0:
                    c1_min = max(c1_min, bound)
                else:
                    c1_max = min(c1_max, bound)
            if abs(dy) > 1e-9:
                bound = -c1*dx/dy
                if dy > 0:
                    c2_min = max(c2_min, bound)
                else:
                    c2_max = min(c2_max, bound)

    return {"c1": (c1_min, c1_max), "c2": (c2_min, c2_max)}

# =================== CONSTRUIR Y RESOLVER MODELO ===================
def construir_y_resolver_modelo(c1, c2, restricciones, tipo_problema, tipo_x, tipo_y):
    m = pyo.ConcreteModel()
    m.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)

    m.x = pyo.Var(domain=obtener_dominio(tipo_x))
    m.y = pyo.Var(domain=obtener_dominio(tipo_y))

    if tipo_problema == "Minimizar":
        m.obj = pyo.Objective(expr=c1*m.x + c2*m.y, sense=pyo.minimize)
    else:
        m.obj = pyo.Objective(expr=c1*m.x + c2*m.y, sense=pyo.maximize)

    m.cons = pyo.ConstraintList()
    for (a, b, sentido, rhs) in restricciones:
        if sentido == "<=":
            m.cons.add(a*m.x + b*m.y <= rhs)
        elif sentido == ">=":
            m.cons.add(a*m.x + b*m.y >= rhs)
        else:
            m.cons.add(a*m.x + b*m.y == rhs)

    solver = pyo.SolverFactory("appsi_highs")
    resultado = solver.solve(m, load_solutions=True)
    return m, resultado

# =================== SIDEBAR ===================
st.sidebar.title("Configuración")

tipo_problema = st.sidebar.selectbox(
    "Tipo de problema",
    ["Minimizar", "Maximizar"],
    key="tipo_problema_widget"
)

n_restr = st.sidebar.number_input(
    "Número de restricciones",
    min_value=1,
    max_value=6,
    value=2,
    key="n_restr_widget"
)

st.sidebar.markdown("---")
st.sidebar.subheader("Solución óptima")

if st.session_state.get("modelo_resuelto", False):
    st.sidebar.write(f"Estado solver: {st.session_state.get('solver_status', '')}")
    st.sidebar.write(f"x* = {st.session_state['x_opt']:.4f}")
    st.sidebar.write(f"y* = {st.session_state['y_opt']:.4f}")
    st.sidebar.write(f"Z* = {st.session_state['z_opt']:.4f}")
else:
    st.sidebar.info("Resuelve el modelo para ver la solución aquí.")

# =================== CUERPO PRINCIPAL ===================
st.markdown("<h2>Método gráfico (2 variables) con Pyomo</h2>", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

# Naturaleza de las variables
st.subheader("Naturaleza de las variables")
col1, col2 = st.columns(2)
with col1:
    tipo_x = st.selectbox(
        "Naturaleza de X",
        ["Real ≥ 0", "Entera ≥ 0", "Binaria"],
        key="tipo_x_widget"
    )
with col2:
    tipo_y = st.selectbox(
        "Naturaleza de Y",
        ["Real ≥ 0", "Entera ≥ 0", "Binaria"],
        key="tipo_y_widget"
    )

# Función objetivo
st.subheader("Función objetivo")
col_fo_inputs, col_fo_latex = st.columns([2, 3])
with col_fo_inputs:
    c1 = st.number_input("Coeficiente de X", value=3.0, key="c1_widget")
    c2 = st.number_input("Coeficiente de Y", value=5.0, key="c2_widget")

sentido_tex = r"\min" if tipo_problema == "Minimizar" else r"\max"
st.latex(rf"{sentido_tex}\ Z = {c1}x + {c2}y")

st.markdown("<hr>", unsafe_allow_html=True)

# Restricciones
st.subheader("Restricciones")
restricciones = []
for k in range(int(n_restr)):
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(f"<h4>Restricción {k+1}</h4>", unsafe_allow_html=True)

    col_a, col_b, col_sent, col_rhs = st.columns(4)
    with col_a:
        a = st.number_input(
            f"Coeficiente de X en R{k+1}",
            value=1.0,
            key=f"a{k}_widget"
        )
    with col_b:
        b = st.number_input(
            f"Coeficiente de Y en R{k+1}",
            value=1.0,
            key=f"b{k}_widget"
        )
    with col_sent:
        sentido = st.selectbox(
            f"Sentido en R{k+1}",
            ["<=", ">=", "="],
            key=f"sent{k}_widget"
        )
    with col_rhs:
        rhs = st.number_input(
            f"LD en R{k+1}",
            value=8.0,
            key=f"rhs{k}_widget"
        )

    st.latex(rf"{a}x + {b}y\ {sentido}\ {rhs}")
    restricciones.append((a, b, sentido, rhs))

# =================== BOTÓN: RESOLVER Y GUARDAR ===================
if st.button("Resolver y graficar"):
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
        dual_continuo = (tipo_x == "Real ≥ 0" and tipo_y == "Real ≥ 0")
        if dual_continuo:
            for i, cons in enumerate(modelo.cons.values(), start=1):
                dual_val = modelo.dual.get(cons, 0)
                duales.append([f"Restricción {i}", dual_val])
        else:
            for i in range(len(restricciones)):
                duales.append(
                    [f"Restricción {i+1}",
                     "No disponible (modelo entero/binario)"]
                )

        # Vértices y rangos de coeficientes (solo continuo)
        if dual_continuo:
            vertices = enumerar_vertices(restricciones, tipo_x, tipo_y)
            rangos = rangos_coeficientes(vertices, x_opt, y_opt, c1, c2, tipo_problema)
        else:
            vertices = []
            rangos = None

        # Guardar en session_state (nombres distintos a los keys de widgets)
        st.session_state["modelo_resuelto"] = True
        st.session_state["solver_status"] = status
        st.session_state["x_opt"] = x_opt
        st.session_state["y_opt"] = y_opt
        st.session_state["z_opt"] = z_opt
        st.session_state["restricciones"] = restricciones
        st.session_state["c1"] = c1
        st.session_state["c2"] = c2
        st.session_state["duales"] = duales
        st.session_state["vertices"] = vertices
        st.session_state["rangos_coef"] = rangos
        st.session_state["tipo_problema_val"] = tipo_problema
        st.session_state["tipo_x_val"] = tipo_x
        st.session_state["tipo_y_val"] = tipo_y

    except Exception as e:
        st.error(f"Error al resolver el modelo: {e}")

# =================== MOSTRAR GRÁFICA Y ANÁLISIS ===================
if st.session_state.get("modelo_resuelto", False):
    import plotly.graph_objects as go

    restricciones = st.session_state["restricciones"]
    x_opt = st.session_state["x_opt"]
    y_opt = st.session_state["y_opt"]
    c1_res = st.session_state["c1"]
    c2_res = st.session_state["c2"]

    # -------- Gráfica --------
    st.markdown("<hr>", unsafe_allow_html=True)
    st.subheader("Gráfica de la región factible")

    max_rhs = max([abs(r[3]) for r in restricciones] + [10])
    lim = max_rhs * 1.2

    X = np.linspace(0, lim, 400)
    Y = np.linspace(0, lim, 400)
    XX, YY = np.meshgrid(X, Y)

    factible = np.ones_like(XX, dtype=bool)
    for a, b, s, rhs in restricciones:
        if s == "<=":
            factible &= (a*XX + b*YY <= rhs + 1e-9)
        elif s == ">=":
            factible &= (a*XX + b*YY >= rhs - 1e-9)
        else:
            factible &= np.isclose(a*XX + b*YY, rhs, atol=1e-3)

    fig = go.Figure()
    fig.add_trace(go.Contour(
        x=X, y=Y, z=factible.astype(int),
        showscale=False,
        colorscale=[[0, 'rgba(0,0,0,0)'], [1, 'rgba(0,150,255,0.3)']],
        opacity=0.4,
        name="Región factible"
    ))

    for (a, b, s, rhs) in restricciones:
        if abs(b) > 1e-8:
            y_line = (rhs - a*X) / b
            fig.add_trace(go.Scatter(
                x=X, y=y_line, mode="lines",
                name=f"{a}x + {b}y {s} {rhs}"
            ))
        else:
            if abs(a) > 1e-8:
                x_line = rhs / a
                fig.add_trace(go.Scatter(
                    x=[x_line, x_line], y=[0, lim],
                    mode="lines", name=f"{a}x {s} {rhs}"
                ))

    fig.add_trace(go.Scatter(
        x=[x_opt], y=[y_opt],
        mode="markers+text",
        text=["Óptimo"], textposition="top right",
        marker=dict(size=10, color="red"),
        name="Solución óptima"
    ))

    if abs(c2_res) > 1e-8:
        z_opt = c1_res*x_opt + c2_res*y_opt
        y_obj = (z_opt - c1_res*X) / c2_res
        fig.add_trace(go.Scatter(
            x=X, y=y_obj, mode="lines",
            line=dict(dash="dash", color="red"),
            name="FO en Z*"
        ))

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
        width=800, height=600,
        title="Región factible y solución óptima",
        xaxis_title="x", yaxis_title="y",
        legend=dict(x=0.7, y=1.0)
    )
    st.plotly_chart(fig, use_container_width=True)

    # -------- Precios sombra --------
    st.markdown("<hr>", unsafe_allow_html=True)
    st.subheader("Precios sombra (valores duales)")
    st.table(st.session_state.get("duales", []))

    # -------- Rangos de coeficientes --------
    st.markdown("<hr>", unsafe_allow_html=True)
    st.subheader("Análisis de sensibilidad de los coeficientes de la FO")

    tipo_x_val = st.session_state["tipo_x_val"]
    tipo_y_val = st.session_state["tipo_y_val"]
    rangos = st.session_state.get("rangos_coef", None)

    if not (tipo_x_val == "Real ≥ 0" and tipo_y_val == "Real ≥ 0"):
        st.info(
            "El análisis de rangos de los coeficientes de la función objetivo "
            "solo se realiza para modelos continuos (variables reales)."
        )
    elif rangos is None:
        st.info("No se pudieron calcular los rangos de los coeficientes.")
    else:
        def formato_intervalo(lo, hi):
            def fmt(v):
                if np.isneginf(v):
                    return "-∞"
                if np.isposinf(v):
                    return "+∞"
                return f"{v:.4f}"
            return fmt(lo), fmt(hi)

        c1_lo, c1_hi = formato_intervalo(*rangos["c1"])
        c2_lo, c2_hi = formato_intervalo(*rangos["c2"])

        data_sens = [
            ["c1 (coef. de x)", f"{c1_res:.4f}", c1_lo, c1_hi],
            ["c2 (coef. de y)", f"{c2_res:.4f}", c2_lo, c2_hi],
        ]
        st.table(data_sens)
        st.caption(
            "Intervalos en los que puede variar cada coeficiente de la función "
            "objetivo, manteniendo fijo el otro, sin cambiar el punto óptimo (x*, y*)."
        )

# Cerrar contenedor principal
st.markdown("</div>", unsafe_allow_html=True)
