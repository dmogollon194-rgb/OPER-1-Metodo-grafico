import streamlit as st
import pyomo.environ as pyo
import numpy as np
import plotly.graph_objects as go

# =================== WATERMARK ===================
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

# =================== GLOBAL UI STYLES ===================
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

# Styled main container
st.markdown('<div class="main-container">', unsafe_allow_html=True)

# =================== INITIALIZE SESSION STATE ===================
if "open_expander" not in st.session_state:
    st.session_state["open_expander"] = None

# =================== HELPER: VARIABLE DOMAIN ===================
def get_domain(tipo):
    if tipo == "Nonnegative Real":
        return pyo.NonNegativeReals
    elif tipo == "Nonnegative Integer":
        return pyo.NonNegativeIntegers
    elif tipo == "Binary":
        return pyo.Binary

# =================== HELPER: EQUATION objective functionRMATTING ===================
def fmt_num(v):
    if abs(v - round(v)) < 1e-9:
        return str(int(round(v)))
    return f"{v:.2f}"

def equation_text(a, b, operator, rhs):
    return f"{fmt_num(a)}x + {fmt_num(b)}y {operator} {fmt_num(rhs)}"

# =================== POINT FEASIBILITY ===================
def is_feasible_point(x, y, constraints, x_type, y_type, tol=1e-7):
    if x_type in ["Nonnegative Real", "Nonnegative Integer"] and x < -tol:
        return False
    if y_type in ["Nonnegative Real", "Nonnegative Integer"] and y < -tol:
        return False

    if x_type == "Binary" and not (-tol <= x <= 1 + tol):
        return False
    if y_type == "Binary" and not (-tol <= y <= 1 + tol):
        return False

    for (a, b, operator, rhs) in constraints:
        val = a * x + b * y
        if operator == "<=" and val > rhs + tol:
            return False
        elif operator == ">=" and val < rhs - tol:
            return False
        elif operator == "=" and abs(val - rhs) > tol:
            return False

    return True

# =================== ENUMERATE VERTICES ===================
def enumerate_vertices(constraints, x_type, y_type, tol=1e-7):
    all_constraints = list(constraints)

    if x_type in ["Nonnegative Real", "Nonnegative Integer"]:
        all_constraints.append((1.0, 0.0, ">=", 0.0))
    if y_type in ["Nonnegative Real", "Nonnegative Integer"]:
        all_constraints.append((0.0, 1.0, ">=", 0.0))

    if x_type == "Binary":
        all_constraints.append((1.0, 0.0, ">=", 0.0))
        all_constraints.append((1.0, 0.0, "<=", 1.0))
    if y_type == "Binary":
        all_constraints.append((0.0, 1.0, ">=", 0.0))
        all_constraints.append((0.0, 1.0, "<=", 1.0))

    vertices = []
    n = len(all_constraints)

    for i in range(n):
        a1, b1, _, rhs1 = all_constraints[i]
        for j in range(i + 1, n):
            a2, b2, _, rhs2 = all_constraints[j]
            det = a1 * b2 - a2 * b1

            if abs(det) < tol:
                continue

            x = (rhs1 * b2 - rhs2 * b1) / det
            y = (a1 * rhs2 - a2 * rhs1) / det

            if is_feasible_point(x, y, constraints, x_type, y_type, tol):
                vertices.append((x, y))

    uniq = {}
    for x, y in vertices:
        key = (round(x, 6), round(y, 6))
        uniq[key] = (x, y)

    return list(uniq.values())

# =================== SORT VERTICES ===================
def sort_vertices(vertices):
    if len(vertices) <= 2:
        return vertices

    xs = np.array([v[0] for v in vertices])
    ys = np.array([v[1] for v in vertices])

    cx = np.mean(xs)
    cy = np.mean(ys)

    angles = np.arctan2(ys - cy, xs - cx)
    order = np.argsort(angles)

    return [vertices[i] for i in order]

# =================== OBJECTIVE COEFFICIENT RANGES ===================
def coefficient_ranges(vertices, x_opt, y_opt, c1, c2, problem_type):
    if len(vertices) == 0:
        return None

    other_vertices = []
    for vx, vy in vertices:
        if abs(vx - x_opt) > 1e-6 or abs(vy - y_opt) > 1e-6:
            other_vertices.append((vx, vy))

    if len(other_vertices) == 0:
        return {"c1": (-np.inf, np.inf), "c2": (-np.inf, np.inf)}

    c1_min, c1_max = -np.inf, np.inf
    c2_min, c2_max = -np.inf, np.inf
    is_minimization = (problem_type == "Minimize")

    for (vx, vy) in other_vertices:
        dx = x_opt - vx
        dy = y_opt - vy

        if is_minimization:
            if abs(dx) > 1e-9:
                bound = -c2 * dy / dx
                if dx > 0:
                    c1_max = min(c1_max, bound)
                else:
                    c1_min = max(c1_min, bound)

            if abs(dy) > 1e-9:
                bound = -c1 * dx / dy
                if dy > 0:
                    c2_max = min(c2_max, bound)
                else:
                    c2_min = max(c2_min, bound)
        else:
            if abs(dx) > 1e-9:
                bound = -c2 * dy / dx
                if dx > 0:
                    c1_min = max(c1_min, bound)
                else:
                    c1_max = min(c1_max, bound)

            if abs(dy) > 1e-9:
                bound = -c1 * dx / dy
                if dy > 0:
                    c2_min = max(c2_min, bound)
                else:
                    c2_max = min(c2_max, bound)

    return {"c1": (c1_min, c1_max), "c2": (c2_min, c2_max)}

# =================== RHS SENSITIVITY RANGES ===================
def _canonical_rhs_rows(constraints):
    """Convert constraints to a canonical representation for 2D LP sensitivity."""
    rows = []

    for i, (a, b, operator, rhs) in enumerate(constraints):
        if operator == "<=":
            rows.append({
                "n": np.array([float(a), float(b)]),
                "r": float(rhs),
                "kind": "ineq",
                "source": i,
                "rhs_factor": 1.0,
            })
        elif operator == ">=":
            # a*x+b*y >= rhs  <=>  -a*x-b*y <= -rhs
            rows.append({
                "n": np.array([-float(a), -float(b)]),
                "r": -float(rhs),
                "kind": "ineq",
                "source": i,
                "rhs_factor": -1.0,
            })
        else:
            rows.append({
                "n": np.array([float(a), float(b)]),
                "r": float(rhs),
                "kind": "eq",
                "source": i,
                "rhs_factor": 1.0,
            })

    # Non-negativity bounds x >= 0 and y >= 0
    rows.append({
        "n": np.array([-1.0, 0.0]),
        "r": 0.0,
        "kind": "ineq",
        "source": None,
        "rhs_factor": 0.0,
        "name": "x >= 0",
    })
    rows.append({
        "n": np.array([0.0, -1.0]),
        "r": 0.0,
        "kind": "ineq",
        "source": None,
        "rhs_factor": 0.0,
        "name": "y >= 0",
    })

    return rows


def _dual_feasible_bases(rows, x_opt, y_opt, c1, c2, problem_type, tol=1e-8):
    """Find all 2-row optimal bases compatible with the current optimum."""
    point = np.array([float(x_opt), float(y_opt)])
    active = []

    for idx, row in enumerate(rows):
        activity = float(np.dot(row["n"], point))
        if row["kind"] == "eq" or abs(activity - row["r"]) <= 1e-7:
            active.append(idx)

    target = np.array([float(c1), float(c2)])
    if problem_type == "Minimize":
        target = -target

    candidates = []

    for p in range(len(active)):
        for q in range(p + 1, len(active)):
            i = active[p]
            j = active[q]
            B = np.vstack([rows[i]["n"], rows[j]["n"]])
            det = float(np.linalg.det(B))

            if abs(det) <= tol:
                continue

            try:
                multipliers = np.linalg.solve(B.T, target)
            except np.linalg.LinAlgError:
                continue

            ok = True
            for lam, row_idx in zip(multipliers, (i, j)):
                if rows[row_idx]["kind"] == "ineq" and lam < -1e-7:
                    ok = False
                    break

            if ok:
                candidates.append((i, j, abs(det)))

    # Special case: zero objective. Any nonsingular active pair is valid.
    if not candidates and np.linalg.norm(target) <= tol:
        for p in range(len(active)):
            for q in range(p + 1, len(active)):
                i = active[p]
                j = active[q]
                B = np.vstack([rows[i]["n"], rows[j]["n"]])
                det = float(np.linalg.det(B))
                if abs(det) > tol:
                    candidates.append((i, j, abs(det)))

    return candidates


def _rhs_delta_interval_for_basis(rows, constraints, basis, target_i, x_opt, y_opt, tol=1e-9):
    """Allowable RHS delta interval for one constraint under one optimal basis."""
    point = np.array([float(x_opt), float(y_opt)])
    b0, b1, _ = basis
    basis_rows = (b0, b1)
    B = np.vstack([rows[b0]["n"], rows[b1]["n"]])

    target_basis_pos = None
    for pos, row_idx in enumerate(basis_rows):
        if rows[row_idx]["source"] == target_i:
            target_basis_pos = pos
            break

    a, b, operator, rhs = constraints[target_i]

    # If the target row is nonbasic, the optimum does not move until
    # the row reaches the current solution.
    if target_basis_pos is None:
        lhs = float(a * x_opt + b * y_opt)
        if operator == "<=":
            return lhs - float(rhs), np.inf
        if operator == ">=":
            return -np.inf, lhs - float(rhs)
        return lhs - float(rhs), lhs - float(rhs)

    factor = rows[basis_rows[target_basis_pos]]["rhs_factor"]
    e = np.zeros(2)
    e[target_basis_pos] = factor

    try:
        direction = np.linalg.solve(B, e)
    except np.linalg.LinAlgError:
        return None

    delta_lo = -np.inf
    delta_hi = np.inf

    for row_idx, row in enumerate(rows):
        # The changing basic row is satisfied identically by construction.
        if row["source"] == target_i:
            continue

        q = float(np.dot(row["n"], direction))
        activity = float(np.dot(row["n"], point))

        if row["kind"] == "ineq":
            slack = float(row["r"] - activity)
            if slack < 0 and abs(slack) <= 1e-7:
                slack = 0.0

            if q > tol:
                delta_hi = min(delta_hi, slack / q)
            elif q < -tol:
                delta_lo = max(delta_lo, slack / q)
            elif slack < -1e-7:
                return None
        else:
            # An unchanged equality must continue to hold.
            if abs(q) > tol:
                delta_lo = max(delta_lo, 0.0)
                delta_hi = min(delta_hi, 0.0)

    if abs(delta_lo) < 1e-10:
        delta_lo = 0.0
    if abs(delta_hi) < 1e-10:
        delta_hi = 0.0

    if delta_lo > delta_hi + 1e-8:
        return None

    return delta_lo, delta_hi


def rhs_sensitivity_ranges(constraints, x_opt, y_opt, c1, c2, problem_type):
    """
    Compute one-at-a-time RHS ranges for a 2-variable continuous LP.

    The interval reported is the range of each RHS for which an optimal
    basis compatible with the current solution remains primal and dual feasible.
    """
    rows = _canonical_rhs_rows(constraints)
    bases = _dual_feasible_bases(rows, x_opt, y_opt, c1, c2, problem_type)

    if not bases:
        return None

    results = []

    for i, (a, b, operator, rhs) in enumerate(constraints):
        candidate_intervals = []

        # Prefer bases containing the target constraint, because this gives
        # the usual textbook RHS sensitivity range for a binding row.
        containing = [
            basis for basis in bases
            if rows[basis[0]]["source"] == i or rows[basis[1]]["source"] == i
        ]
        bases_to_test = containing if containing else bases

        for basis in bases_to_test:
            interval = _rhs_delta_interval_for_basis(
                rows, constraints, basis, i, x_opt, y_opt
            )
            if interval is None:
                continue

            lo, hi = interval
            if lo <= 1e-8 and hi >= -1e-8:
                # Score wider intervals higher; infinite sides dominate.
                left = np.inf if np.isneginf(lo) else max(0.0, -lo)
                right = np.inf if np.isposinf(hi) else max(0.0, hi)
                score = (
                    int(np.isinf(left)) + int(np.isinf(right)),
                    (0.0 if np.isinf(left) else left) + (0.0 if np.isinf(right) else right),
                    basis[2],
                )
                candidate_intervals.append((score, lo, hi))

        if not candidate_intervals:
            results.append(None)
            continue

        candidate_intervals.sort(key=lambda t: t[0], reverse=True)
        _, delta_lo, delta_hi = candidate_intervals[0]

        rhs_value = float(rhs)
        rhs_min = -np.inf if np.isneginf(delta_lo) else rhs_value + delta_lo
        rhs_max = np.inf if np.isposinf(delta_hi) else rhs_value + delta_hi
        allowable_decrease = np.inf if np.isneginf(delta_lo) else max(0.0, -delta_lo)
        allowable_increase = np.inf if np.isposinf(delta_hi) else max(0.0, delta_hi)

        results.append({
            "rhs_current": rhs_value,
            "rhs_min": rhs_min,
            "rhs_max": rhs_max,
            "allowable_decrease": allowable_decrease,
            "allowable_increase": allowable_increase,
        })

    return results

# =================== BUILD AND SOLVE MODEL ===================
def build_and_solve_model(c1, c2, constraints, problem_type, x_type, y_type):
    m = pyo.ConcreteModel()

    continuous_dual = (x_type == "Nonnegative Real" and y_type == "Nonnegative Real")
    if continuous_dual:
        m.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)

    m.x = pyo.Var(domain=get_domain(x_type))
    m.y = pyo.Var(domain=get_domain(y_type))

    if problem_type == "Minimize":
        m.obj = pyo.Objective(expr=c1 * m.x + c2 * m.y, sense=pyo.minimize)
    else:
        m.obj = pyo.Objective(expr=c1 * m.x + c2 * m.y, sense=pyo.maximize)

    m.cons = pyo.ConstraintList()
    for (a, b, operator, rhs) in constraints:
        if operator == "<=":
            m.cons.add(a * m.x + b * m.y <= rhs)
        elif operator == ">=":
            m.cons.add(a * m.x + b * m.y >= rhs)
        else:
            m.cons.add(a * m.x + b * m.y == rhs)

    solver = pyo.SolverFactory("appsi_highs")
    result = solver.solve(m, load_solutions=True)

    return m, result

# =================== CALLBACKS TO KEEP EXPANDER OPEN ===================
def keep_expander_open(k):
    st.session_state["open_expander"] = k

# =================== SIDEBAR ===================
st.sidebar.title("Configuration")

problem_type = st.sidebar.selectbox(
    "Problem type",
    ["Minimize", "Maximize"],
    key="problem_type_widget"
)

n_restr = st.sidebar.number_input(
    "Number of constraints",
    min_value=1,
    max_value=20,
    value=2,
    step=1,
    key="n_constraints_widget"
)

st.sidebar.markdown("---")
st.sidebar.subheader("Optimal solution")

if st.session_state.get("model_solved", False):
    st.sidebar.write(f"Solver status: {st.session_state.get('solver_status', '')}")
    st.sidebar.write(f"x* = {st.session_state['x_opt']:.4f}")
    st.sidebar.write(f"y* = {st.session_state['y_opt']:.4f}")
    st.sidebar.write(f"Z* = {st.session_state['z_opt']:.4f}")
else:
    st.sidebar.info("Solve the model to view the solution here.")

# =================== MAIN CONTENT ===================
st.markdown("<h2>Graphical Method</h2>", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

# Variable domains
st.subheader("Variable Domains")
col1, col2 = st.columns(2)

with col1:
    x_type = st.selectbox(
        "Domain of X",
        ["Nonnegative Real", "Nonnegative Integer", "Binary"],
        key="x_type_widget"
    )

with col2:
    y_type = st.selectbox(
        "Domain of Y",
        ["Nonnegative Real", "Nonnegative Integer", "Binary"],
        key="y_type_widget"
    )

# Objective function
st.subheader("Objective Function")
col_fo_inputs, col_fo_latex = st.columns([2, 3])

with col_fo_inputs:
    c1 = st.number_input("Coefficient of X", value=3.0, key="c1_widget")
    c2 = st.number_input("Coefficient of Y", value=5.0, key="c2_widget")

sense_tex = r"\min" if problem_type == "Minimize" else r"\max"
st.latex(rf"{sense_tex}\ Z = {c1}x + {c2}y")

st.markdown("<hr>", unsafe_allow_html=True)

# =================== CONSTRAINTS ===================
st.subheader("Constraints")
constraints = []

for k in range(int(n_restr)):
    a_preview = st.session_state.get(f"a{k}_widget", 1.0)
    b_preview = st.session_state.get(f"b{k}_widget", 1.0)
    operator_preview = st.session_state.get(f"sent{k}_widget", "<=")
    rhs_preview = st.session_state.get(f"rhs{k}_widget", 8.0)

    title = f"Constraint {k+1}: {equation_text(a_preview, b_preview, operator_preview, rhs_preview)}"

    expanded_now = (st.session_state.get("open_expander") == k)

    with st.expander(title, expanded=expanded_now):
        col_a, col_b, col_sent, col_rhs = st.columns(4)

        with col_a:
            a = st.number_input(
                f"X coefficient in C{k+1}",
                value=float(a_preview),
                key=f"a{k}_widget",
                on_change=keep_expander_open,
                args=(k,)
            )

        with col_b:
            b = st.number_input(
                f"Y coefficient in C{k+1}",
                value=float(b_preview),
                key=f"b{k}_widget",
                on_change=keep_expander_open,
                args=(k,)
            )

        with col_sent:
            options = ["<=", ">=", "="]
            idx = options.index(operator_preview) if operator_preview in options else 0
            operator = st.selectbox(
                f"Operator in C{k+1}",
                options,
                index=idx,
                key=f"sent{k}_widget",
                on_change=keep_expander_open,
                args=(k,)
            )

        with col_rhs:
            rhs = st.number_input(
                f"RHS in C{k+1}",
                value=float(rhs_preview),
                key=f"rhs{k}_widget",
                on_change=keep_expander_open,
                args=(k,)
            )

        st.latex(rf"{a}x + {b}y\ {operator}\ {rhs}")

    a_current = st.session_state.get(f"a{k}_widget", 1.0)
    b_current = st.session_state.get(f"b{k}_widget", 1.0)
    operator_current = st.session_state.get(f"sent{k}_widget", "<=")
    rhs_current = st.session_state.get(f"rhs{k}_widget", 8.0)

    constraints.append((a_current, b_current, operator_current, rhs_current))

# =================== BUTTON: SOLVE AND STORE ===================
if st.button("Solve and plot"):
    try:
        model, result = build_and_solve_model(
            c1, c2, constraints, problem_type, x_type, y_type
        )

        x_opt = pyo.value(model.x)
        y_opt = pyo.value(model.y)
        z_opt = pyo.value(model.obj)
        status = str(result.solver.termination_condition)

        duals = []
        continuous_dual = (x_type == "Nonnegative Real" and y_type == "Nonnegative Real")

        if continuous_dual:
            for i, cons in enumerate(model.cons.values(), start=1):
                a_i, b_i, operator_i, rhs_i = constraints[i - 1]
                equation_i = equation_text(a_i, b_i, operator_i, rhs_i)
                dual_value = model.dual.get(cons, 0)
                duals.append([equation_i, dual_value])
        else:
            for i in range(len(constraints)):
                a_i, b_i, operator_i, rhs_i = constraints[i]
                equation_i = equation_text(a_i, b_i, operator_i, rhs_i)
                duals.append([equation_i, "Not available (integer/binary model)"])

        vertices = enumerate_vertices(constraints, x_type, y_type)

        if continuous_dual:
            ranges = coefficient_ranges(vertices, x_opt, y_opt, c1, c2, problem_type)
            rhs_ranges = rhs_sensitivity_ranges(
                constraints, x_opt, y_opt, c1, c2, problem_type
            )
        else:
            ranges = None
            rhs_ranges = None

        st.session_state["model_solved"] = True
        st.session_state["solver_status"] = status
        st.session_state["x_opt"] = x_opt
        st.session_state["y_opt"] = y_opt
        st.session_state["z_opt"] = z_opt
        st.session_state["constraints"] = constraints
        st.session_state["c1"] = c1
        st.session_state["c2"] = c2
        st.session_state["duals"] = duals
        st.session_state["vertices"] = vertices
        st.session_state["coefficient_ranges_result"] = ranges
        st.session_state["rhs_ranges_result"] = rhs_ranges
        st.session_state["problem_type_value"] = problem_type
        st.session_state["x_type_value"] = x_type
        st.session_state["y_type_value"] = y_type

    except Exception as e:
        st.error(f"Error solving the model: {e}")

# =================== DISPLAY GRAPH AND ANALYSIS ===================
if st.session_state.get("model_solved", False):
    constraints = st.session_state["constraints"]
    x_opt = st.session_state["x_opt"]
    y_opt = st.session_state["y_opt"]
    c1_result = st.session_state["c1"]
    c2_result = st.session_state["c2"]
    vertices = st.session_state.get("vertices", [])

    st.markdown("<hr>", unsafe_allow_html=True)
    st.subheader("Feasible Region Plot")

    x_candidates = [x_opt] + [v[0] for v in vertices] if vertices else [x_opt]
    y_candidates = [y_opt] + [v[1] for v in vertices] if vertices else [y_opt]

    max_x = max(x_candidates + [10])
    max_y = max(y_candidates + [10])
    lim = 1.25 * max(max_x, max_y)

    X = np.linspace(0, lim, 400)

    fig = go.Figure()

    if len(vertices) >= 3:
        sorted_vertices = sort_vertices(vertices)
        x_poly = [v[0] for v in sorted_vertices] + [sorted_vertices[0][0]]
        y_poly = [v[1] for v in sorted_vertices] + [sorted_vertices[0][1]]

        fig.add_trace(go.Scatter(
            x=x_poly,
            y=y_poly,
            mode="lines",
            fill="toself",
            fillcolor="rgba(0,150,255,0.25)",
            line=dict(color="rgba(0,150,255,0.9)", width=2),
            name="Feasible region"
        ))

    for (a, b, s, rhs) in constraints:
        if abs(b) > 1e-8:
            y_line = (rhs - a * X) / b
            fig.add_trace(go.Scatter(
                x=X,
                y=y_line,
                mode="lines",
                name=equation_text(a, b, s, rhs)
            ))
        else:
            if abs(a) > 1e-8:
                x_line = rhs / a
                fig.add_trace(go.Scatter(
                    x=[x_line, x_line],
                    y=[0, lim],
                    mode="lines",
                    name=equation_text(a, 0, s, rhs)
                ))

    fig.add_trace(go.Scatter(
        x=[x_opt],
        y=[y_opt],
        mode="markers+text",
        text=["Optimum"],
        textposition="top right",
        marker=dict(size=10, color="red"),
        name="Optimal solution"
    ))

    if abs(c2_result) > 1e-8:
        z_opt_line = c1_result * x_opt + c2_result * y_opt
        y_obj = (z_opt_line - c1_result * X) / c2_result
        fig.add_trace(go.Scatter(
            x=X,
            y=y_obj,
            mode="lines",
            line=dict(dash="dash", color="red"),
            name="Objective function at Z*"
        ))
    elif abs(c1_result) > 1e-8:
        x_line = (c1_result * x_opt + c2_result * y_opt) / c1_result
        fig.add_trace(go.Scatter(
            x=[x_line, x_line],
            y=[0, lim],
            mode="lines",
            line=dict(dash="dash", color="red"),
            name="Objective function at Z*"
        ))

    if len(vertices) >= 1:
        xs = [v[0] for v in vertices] + [x_opt]
        ys = [v[1] for v in vertices] + [y_opt]

        x_min = min(xs) - 5
        x_max = max(xs) + 5
        y_min = min(ys) - 5
        y_max = max(ys) + 5

        fig.update_xaxes(range=[x_min, x_max])
        fig.update_yaxes(range=[y_min, y_max])
    else:
        fig.update_xaxes(range=[0, lim])
        fig.update_yaxes(range=[0, lim])

    fig.update_layout(
        width=850,
        height=600,
        title="Feasible Region and Optimal Solution",
        xaxis_title="x",
        yaxis_title="y",
        legend=dict(x=0.68, y=1.0)
    )

    st.plotly_chart(fig, use_container_width=True)

    # -------- Shadow prices --------
    st.markdown("<hr>", unsafe_allow_html=True)
    st.subheader("Shadow Prices")

    duals = st.session_state.get("duals", [])

    if isinstance(duals, list) and len(duals) > 0:
        dual_table = {
            "Constraint": [row[0] for row in duals],
            "Shadow price (Dual)": [row[1] for row in duals],
        }
        st.table(dual_table)
    else:
        st.info("No shadow prices are available.")

    # -------- RHS ranges --------
    st.markdown("<hr>", unsafe_allow_html=True)
    st.subheader("Constraint RHS Sensitivity")

    x_type_value = st.session_state["x_type_value"]
    y_type_value = st.session_state["y_type_value"]
    rhs_ranges = st.session_state.get("rhs_ranges_result", None)

    def format_sensitivity_value(v):
        if v is None:
            return "N/A"
        if np.isneginf(v):
            return "-∞"
        if np.isposinf(v):
            return "+∞"
        return f"{v:.4f}"

    if not (x_type_value == "Nonnegative Real" and y_type_value == "Nonnegative Real"):
        st.info("RHS sensitivity analysis is available only for continuous LP models.")
    elif rhs_ranges is None:
        st.info("The RHS sensitivity ranges could not be calculated for this solution.")
    else:
        rhs_table = {
            "Constraint": [],
            "Current RHS": [],
            "Minimum RHS": [],
            "Maximum RHS": [],
            "Allowable decrease": [],
            "Allowable increase": [],
            "Shadow price": [],
        }

        for i, (a_i, b_i, operator_i, rhs_i) in enumerate(constraints):
            equation_i = equation_text(a_i, b_i, operator_i, rhs_i)
            rr = rhs_ranges[i] if i < len(rhs_ranges) else None

            rhs_table["Constraint"].append(equation_i)
            rhs_table["Current RHS"].append(f"{rhs_i:.4f}")

            if rr is None:
                rhs_table["Minimum RHS"].append("N/A")
                rhs_table["Maximum RHS"].append("N/A")
                rhs_table["Allowable decrease"].append("N/A")
                rhs_table["Allowable increase"].append("N/A")
            else:
                rhs_table["Minimum RHS"].append(format_sensitivity_value(rr["rhs_min"]))
                rhs_table["Maximum RHS"].append(format_sensitivity_value(rr["rhs_max"]))
                rhs_table["Allowable decrease"].append(
                    format_sensitivity_value(rr["allowable_decrease"])
                )
                rhs_table["Allowable increase"].append(
                    format_sensitivity_value(rr["allowable_increase"])
                )

            dual_value = duals[i][1] if i < len(duals) else "N/A"
            if isinstance(dual_value, (int, float, np.integer, np.floating)):
                dual_value = f"{float(dual_value):.4f}"
            rhs_table["Shadow price"].append(dual_value)

        st.table(rhs_table)
        st.caption(
            "Minimum/maximum RHS values are one-at-a-time sensitivity limits: "
            "inside this interval, an optimal basis compatible with the current solution remains valid."
        )

    # -------- Coefficient ranges --------
    st.markdown("<hr>", unsafe_allow_html=True) 
    st.subheader("Objective Function Coefficients")

    x_type_value = st.session_state["x_type_value"]
    y_type_value = st.session_state["y_type_value"]
    ranges = st.session_state.get("coefficient_ranges_result", None)

    if not (x_type_value == "Nonnegative Real" and y_type_value == "Nonnegative Real"):
        st.info(
            "Analysis not available for integer or binary models."
        )
    elif ranges is None:
        st.info("The coefficient ranges could not be calculated.")
    else:
        def format_interval(lo, hi):
            def fmt(v):
                if np.isneginf(v):
                    return "-∞"
                if np.isposinf(v):
                    return "+∞"
                return f"{v:.4f}"
            return fmt(lo), fmt(hi)

        c1_lo, c1_hi = format_interval(*ranges["c1"])
        c2_lo, c2_hi = format_interval(*ranges["c2"])

        sensitivity_table = {
            "Coefficient": ["c1 (coefficient of x)", "c2 (coefficient of y)"],
            "Current value": [f"{c1_result:.4f}", f"{c2_result:.4f}"],
            "Lower bound": [c1_lo, c2_lo],
            "Upper bound": [c1_hi, c2_hi],
        }
        st.table(sensitivity_table)

# Close main container
st.markdown("</div>", unsafe_allow_html=True)
