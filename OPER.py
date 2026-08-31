import streamlit as st
import pyomo.environ as pyo
import numpy as np
import plotly.graph_objects as go


# =================== PAGE CONFIGURATION ===================
st.set_page_config(
    page_title="Graphical Method",
    layout="wide",
)


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
def get_domain(variable_type):
    if variable_type == "Nonnegative Real":
        return pyo.NonNegativeReals
    if variable_type == "Nonnegative Integer":
        return pyo.NonNegativeIntegers
    if variable_type == "Binary":
        return pyo.Binary

    raise ValueError(f"Unknown variable domain: {variable_type}")


# =================== HELPER: EQUATION FORMATTING ===================
def fmt_num(value):
    value = float(value)
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.2f}"


def equation_text(a, b, operator, rhs):
    return f"{fmt_num(a)}x + {fmt_num(b)}y {operator} {fmt_num(rhs)}"


# =================== POINT FEASIBILITY ===================
def is_feasible_point(x, y, constraints, x_type, y_type, tol=1e-7):
    # X domain
    if x_type == "Nonnegative Real":
        if x < -tol:
            return False
    elif x_type == "Nonnegative Integer":
        if x < -tol or abs(x - round(x)) > tol:
            return False
    elif x_type == "Binary":
        if min(abs(x), abs(x - 1.0)) > tol:
            return False

    # Y domain
    if y_type == "Nonnegative Real":
        if y < -tol:
            return False
    elif y_type == "Nonnegative Integer":
        if y < -tol or abs(y - round(y)) > tol:
            return False
    elif y_type == "Binary":
        if min(abs(y), abs(y - 1.0)) > tol:
            return False

    # User constraints
    for a, b, operator, rhs in constraints:
        lhs = a * x + b * y

        if operator == "<=" and lhs > rhs + tol:
            return False
        if operator == ">=" and lhs < rhs - tol:
            return False
        if operator == "=" and abs(lhs - rhs) > tol:
            return False

    return True


# =================== ENUMERATE CONTINUOUS VERTICES ===================
def enumerate_vertices(constraints, x_type, y_type, tol=1e-7):
    """
    Enumerate pairwise intersections used to display the graphical feasible region.

    For continuous models, these are the usual extreme-point candidates.
    For integer/binary models, the optimization itself is still solved by HiGHS;
    this function is used only for graphical support.
    """
    all_constraints = list(constraints)

    # Add domain boundary lines.
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
    n_rows = len(all_constraints)

    for i in range(n_rows):
        a1, b1, _, rhs1 = all_constraints[i]

        for j in range(i + 1, n_rows):
            a2, b2, _, rhs2 = all_constraints[j]
            determinant = a1 * b2 - a2 * b1

            if abs(determinant) < tol:
                continue

            x = (rhs1 * b2 - rhs2 * b1) / determinant
            y = (a1 * rhs2 - a2 * rhs1) / determinant

            # For the polygon, continuous feasibility is what matters.
            # Integer/binary domains are handled by the solver, not by polygon vertices.
            graphical_x_type = "Nonnegative Real" if x_type == "Nonnegative Integer" else x_type
            graphical_y_type = "Nonnegative Real" if y_type == "Nonnegative Integer" else y_type

            if is_feasible_point(
                x,
                y,
                constraints,
                graphical_x_type,
                graphical_y_type,
                tol,
            ):
                vertices.append((x, y))

    unique_vertices = {}

    for x, y in vertices:
        key = (round(float(x), 6), round(float(y), 6))
        unique_vertices[key] = (float(x), float(y))

    return list(unique_vertices.values())


# =================== SORT VERTICES ===================
def sort_vertices(vertices):
    if len(vertices) <= 2:
        return vertices

    xs = np.array([point[0] for point in vertices], dtype=float)
    ys = np.array([point[1] for point in vertices], dtype=float)

    center_x = np.mean(xs)
    center_y = np.mean(ys)

    angles = np.arctan2(ys - center_y, xs - center_x)
    order = np.argsort(angles)

    return [vertices[index] for index in order]


# =================== OBJECTIVE COEFFICIENT RANGES ===================
def coefficient_ranges(vertices, x_opt, y_opt, c1, c2, problem_type):
    """
    Compute one-at-a-time objective-coefficient ranges for a 2-variable
    continuous LP using the current optimal extreme point.
    """
    if len(vertices) == 0:
        return None

    other_vertices = []

    for vx, vy in vertices:
        if abs(vx - x_opt) > 1e-6 or abs(vy - y_opt) > 1e-6:
            other_vertices.append((vx, vy))

    if len(other_vertices) == 0:
        return {
            "c1": (-np.inf, np.inf),
            "c2": (-np.inf, np.inf),
        }

    c1_min = -np.inf
    c1_max = np.inf
    c2_min = -np.inf
    c2_max = np.inf

    is_minimization = problem_type == "Minimize"

    for vx, vy in other_vertices:
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

    return {
        "c1": (c1_min, c1_max),
        "c2": (c2_min, c2_max),
    }


# =================== CONTINUOUS RHS SENSITIVITY ===================
def _canonical_rhs_rows(constraints):
    """Create n*x <= r rows used for 2D LP RHS sensitivity."""
    rows = []

    for index, (a, b, operator, rhs) in enumerate(constraints):
        if operator == "<=":
            rows.append({
                "normal": np.array([float(a), float(b)]),
                "rhs": float(rhs),
                "kind": "ineq",
                "source": index,
                "rhs_factor": 1.0,
            })
        elif operator == ">=":
            rows.append({
                "normal": np.array([-float(a), -float(b)]),
                "rhs": -float(rhs),
                "kind": "ineq",
                "source": index,
                "rhs_factor": -1.0,
            })
        else:
            rows.append({
                "normal": np.array([float(a), float(b)]),
                "rhs": float(rhs),
                "kind": "eq",
                "source": index,
                "rhs_factor": 1.0,
            })

    # Nonnegativity boundaries: -x <= 0 and -y <= 0.
    rows.append({
        "normal": np.array([-1.0, 0.0]),
        "rhs": 0.0,
        "kind": "ineq",
        "source": None,
        "rhs_factor": 0.0,
    })
    rows.append({
        "normal": np.array([0.0, -1.0]),
        "rhs": 0.0,
        "kind": "ineq",
        "source": None,
        "rhs_factor": 0.0,
    })

    return rows


def _dual_feasible_bases(rows, x_opt, y_opt, c1, c2, problem_type, tol=1e-8):
    """Return nonsingular active two-row bases compatible with optimality."""
    point = np.array([float(x_opt), float(y_opt)])
    active_rows = []

    for row_index, row in enumerate(rows):
        activity = float(np.dot(row["normal"], point))
        if row["kind"] == "eq" or abs(activity - row["rhs"]) <= 1e-7:
            active_rows.append(row_index)

    objective = np.array([float(c1), float(c2)])
    target = -objective if problem_type == "Minimize" else objective
    candidates = []

    for p in range(len(active_rows)):
        for q in range(p + 1, len(active_rows)):
            i = active_rows[p]
            j = active_rows[q]
            basis_matrix = np.vstack([rows[i]["normal"], rows[j]["normal"]])
            determinant = float(np.linalg.det(basis_matrix))

            if abs(determinant) <= tol:
                continue

            try:
                multipliers = np.linalg.solve(basis_matrix.T, target)
            except np.linalg.LinAlgError:
                continue

            dual_feasible = True
            for multiplier, row_index in zip(multipliers, (i, j)):
                if rows[row_index]["kind"] == "ineq" and multiplier < -1e-7:
                    dual_feasible = False
                    break

            if dual_feasible:
                candidates.append((i, j, abs(determinant)))

    # If the objective is identically zero, any nonsingular active pair works.
    if not candidates and np.linalg.norm(target) <= tol:
        for p in range(len(active_rows)):
            for q in range(p + 1, len(active_rows)):
                i = active_rows[p]
                j = active_rows[q]
                basis_matrix = np.vstack([rows[i]["normal"], rows[j]["normal"]])
                determinant = float(np.linalg.det(basis_matrix))
                if abs(determinant) > tol:
                    candidates.append((i, j, abs(determinant)))

    return candidates


def _rhs_delta_interval_for_basis(rows, constraints, basis, target_index, x_opt, y_opt, tol=1e-9):
    """RHS delta interval that preserves primal feasibility for one optimal basis."""
    point = np.array([float(x_opt), float(y_opt)])
    row_0, row_1, _ = basis
    basis_rows = (row_0, row_1)
    basis_matrix = np.vstack([rows[row_0]["normal"], rows[row_1]["normal"]])

    target_basis_position = None
    for position, row_index in enumerate(basis_rows):
        if rows[row_index]["source"] == target_index:
            target_basis_position = position
            break

    a, b, operator, rhs = constraints[target_index]

    # A nonbasic inequality can move outward indefinitely; inward movement
    # is limited by the slack at the current optimum.
    if target_basis_position is None:
        lhs = float(a * x_opt + b * y_opt)
        if operator == "<=":
            return lhs - float(rhs), np.inf
        if operator == ">=":
            return -np.inf, lhs - float(rhs)
        return 0.0, 0.0

    rhs_factor = rows[basis_rows[target_basis_position]]["rhs_factor"]
    unit_change = np.zeros(2)
    unit_change[target_basis_position] = rhs_factor

    try:
        direction = np.linalg.solve(basis_matrix, unit_change)
    except np.linalg.LinAlgError:
        return None

    delta_lower = -np.inf
    delta_upper = np.inf

    for row in rows:
        if row["source"] == target_index:
            continue

        directional_activity = float(np.dot(row["normal"], direction))
        current_activity = float(np.dot(row["normal"], point))

        if row["kind"] == "ineq":
            slack = float(row["rhs"] - current_activity)
            if abs(slack) <= 1e-7:
                slack = 0.0

            if directional_activity > tol:
                delta_upper = min(delta_upper, slack / directional_activity)
            elif directional_activity < -tol:
                delta_lower = max(delta_lower, slack / directional_activity)
            elif slack < -1e-7:
                return None
        else:
            # An unchanged equality must remain satisfied.
            if abs(directional_activity) > tol:
                delta_lower = max(delta_lower, 0.0)
                delta_upper = min(delta_upper, 0.0)

    if abs(delta_lower) < 1e-10:
        delta_lower = 0.0
    if abs(delta_upper) < 1e-10:
        delta_upper = 0.0

    if delta_lower > delta_upper + 1e-8:
        return None

    return delta_lower, delta_upper


def continuous_rhs_sensitivity_ranges(constraints, x_opt, y_opt, c1, c2, problem_type):
    """
    One-at-a-time RHS sensitivity for the two-variable continuous LP.

    The reported interval preserves at least one optimal basis compatible
    with the current optimal solution.
    """
    rows = _canonical_rhs_rows(constraints)
    bases = _dual_feasible_bases(
        rows, x_opt, y_opt, c1, c2, problem_type
    )

    if not bases:
        return None

    results = []

    for target_index, (_, _, _, rhs) in enumerate(constraints):
        candidate_intervals = []

        containing_bases = [
            basis
            for basis in bases
            if rows[basis[0]]["source"] == target_index
            or rows[basis[1]]["source"] == target_index
        ]

        bases_to_test = containing_bases if containing_bases else bases

        for basis in bases_to_test:
            interval = _rhs_delta_interval_for_basis(
                rows,
                constraints,
                basis,
                target_index,
                x_opt,
                y_opt,
            )

            if interval is None:
                continue

            delta_lower, delta_upper = interval

            if delta_lower <= 1e-8 and delta_upper >= -1e-8:
                left_width = np.inf if np.isneginf(delta_lower) else max(0.0, -delta_lower)
                right_width = np.inf if np.isposinf(delta_upper) else max(0.0, delta_upper)
                score = (
                    int(np.isinf(left_width)) + int(np.isinf(right_width)),
                    (0.0 if np.isinf(left_width) else left_width)
                    + (0.0 if np.isinf(right_width) else right_width),
                    basis[2],
                )
                candidate_intervals.append((score, delta_lower, delta_upper))

        if not candidate_intervals:
            results.append(None)
            continue

        candidate_intervals.sort(key=lambda item: item[0], reverse=True)
        _, delta_lower, delta_upper = candidate_intervals[0]

        rhs_value = float(rhs)
        rhs_min = -np.inf if np.isneginf(delta_lower) else rhs_value + delta_lower
        rhs_max = np.inf if np.isposinf(delta_upper) else rhs_value + delta_upper

        results.append({
            "rhs_current": rhs_value,
            "rhs_min": rhs_min,
            "rhs_max": rhs_max,
            "allowable_decrease": (
                np.inf if np.isneginf(delta_lower) else max(0.0, -delta_lower)
            ),
            "allowable_increase": (
                np.inf if np.isposinf(delta_upper) else max(0.0, delta_upper)
            ),
            "method": "LP basis sensitivity",
        })

    return results


# =================== SOLVER HELPERS ===================
def create_highs_solver(time_limit=10.0):
    """Create the standard Pyomo HiGHS interface and verify availability."""
    solver = pyo.SolverFactory("highs")

    if solver is None or not solver.available(exception_flag=False):
        raise RuntimeError(
            "HiGHS is not available. Add 'highspy' to requirements.txt "
            "and redeploy the Streamlit application."
        )

    solver.options["time_limit"] = float(time_limit)

    return solver


# =================== DISCRETE RHS SENSITIVITY ===================
def _discrete_rhs_relaxation_breakpoint(
    constraints,
    target_index,
    x_opt,
    y_opt,
    c1,
    c2,
    problem_type,
    x_type,
    y_type,
    time_limit=3.0,
):
    """
    Find the relaxation breakpoint for one discrete/mixed RHS.

    A finite breakpoint is the first RHS level at which a strictly better
    solution can enter after relaxing the selected inequality.
    """
    model = pyo.ConcreteModel()
    model.x = pyo.Var(domain=get_domain(x_type))
    model.y = pyo.Var(domain=get_domain(y_type))

    model.cons = pyo.ConstraintList()

    for index, (a, b, operator, rhs) in enumerate(constraints):
        if index == target_index:
            continue

        expression = a * model.x + b * model.y

        if operator == "<=":
            model.cons.add(expression <= rhs)
        elif operator == ">=":
            model.cons.add(expression >= rhs)
        elif operator == "=":
            model.cons.add(expression == rhs)

    incumbent_value = float(c1 * x_opt + c2 * y_opt)
    epsilon = 1e-7 * max(1.0, abs(incumbent_value))
    epsilon = max(epsilon, 1e-9)

    objective_expression = c1 * model.x + c2 * model.y

    if problem_type == "Maximize":
        model.strictly_better = pyo.Constraint(
            expr=objective_expression >= incumbent_value + epsilon
        )
    else:
        model.strictly_better = pyo.Constraint(
            expr=objective_expression <= incumbent_value - epsilon
        )

    a, b, operator, _ = constraints[target_index]
    row_activity = a * model.x + b * model.y

    if operator == "<=":
        model.breakpoint_objective = pyo.Objective(
            expr=row_activity,
            sense=pyo.minimize,
        )
    elif operator == ">=":
        model.breakpoint_objective = pyo.Objective(
            expr=row_activity,
            sense=pyo.maximize,
        )
    else:
        return {
            "status": "equality",
            "value": None,
            "epsilon": epsilon,
        }

    solver = create_highs_solver(time_limit=time_limit)

    try:
        result = solver.solve(
            model,
            tee=False,
            load_solutions=False,
        )
    except Exception as error:
        return {
            "status": "error",
            "value": None,
            "message": str(error),
            "epsilon": epsilon,
        }

    termination = result.solver.termination_condition

    if termination == pyo.TerminationCondition.infeasible:
        return {
            "status": "no_better_solution",
            "value": None,
            "epsilon": epsilon,
        }

    if termination == pyo.TerminationCondition.maxTimeLimit:
        return {
            "status": "time_limit",
            "value": None,
            "epsilon": epsilon,
        }

    if termination != pyo.TerminationCondition.optimal:
        return {
            "status": str(termination),
            "value": None,
            "epsilon": epsilon,
        }

    model.solutions.load_from(result)

    return {
        "status": "optimal",
        "value": float(pyo.value(row_activity)),
        "epsilon": epsilon,
    }


def discrete_rhs_sensitivity_ranges(
    constraints,
    x_opt,
    y_opt,
    c1,
    c2,
    problem_type,
    x_type,
    y_type,
):
    """
    One-at-a-time RHS stability analysis for integer, binary, and mixed models.

    This is not classical LP sensitivity. It reports the interval over which
    the current discrete incumbent remains optimal, with finite relaxation
    breakpoints estimated through auxiliary reoptimization.
    """
    results = []

    for index, (a, b, operator, rhs) in enumerate(constraints):
        rhs = float(rhs)
        incumbent_activity = float(a * x_opt + b * y_opt)

        if operator == "=":
            results.append({
                "rhs_current": rhs,
                "rhs_min": rhs,
                "rhs_max": rhs,
                "allowable_decrease": 0.0,
                "allowable_increase": 0.0,
                "status": "exact equality",
                "method": "Discrete incumbent stability",
            })
            continue

        breakpoint = _discrete_rhs_relaxation_breakpoint(
            constraints,
            index,
            x_opt,
            y_opt,
            c1,
            c2,
            problem_type,
            x_type,
            y_type,
        )

        if operator == "<=":
            rhs_min = incumbent_activity

            if breakpoint["status"] == "no_better_solution":
                rhs_max = np.inf
            elif breakpoint["status"] == "optimal":
                rhs_max = max(rhs, float(breakpoint["value"]))
            else:
                rhs_max = None

        else:  # >=
            rhs_max = incumbent_activity

            if breakpoint["status"] == "no_better_solution":
                rhs_min = -np.inf
            elif breakpoint["status"] == "optimal":
                rhs_min = min(rhs, float(breakpoint["value"]))
            else:
                rhs_min = None

        allowable_decrease = None
        allowable_increase = None

        if rhs_min is not None:
            allowable_decrease = (
                np.inf if np.isneginf(rhs_min) else max(0.0, rhs - rhs_min)
            )

        if rhs_max is not None:
            allowable_increase = (
                np.inf if np.isposinf(rhs_max) else max(0.0, rhs_max - rhs)
            )

        results.append({
            "rhs_current": rhs,
            "rhs_min": rhs_min,
            "rhs_max": rhs_max,
            "allowable_decrease": allowable_decrease,
            "allowable_increase": allowable_increase,
            "status": breakpoint["status"],
            "method": "Discrete breakpoint reoptimization",
        })

    return results


# =================== BUILD AND SOLVE MODEL ===================
def build_and_solve_model(c1, c2, constraints, problem_type, x_type, y_type):
    model = pyo.ConcreteModel()

    continuous_model = (
        x_type == "Nonnegative Real"
        and y_type == "Nonnegative Real"
    )

    if continuous_model:
        model.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)

    model.x = pyo.Var(domain=get_domain(x_type))
    model.y = pyo.Var(domain=get_domain(y_type))

    objective_expression = c1 * model.x + c2 * model.y

    if problem_type == "Minimize":
        model.obj = pyo.Objective(
            expr=objective_expression,
            sense=pyo.minimize,
        )
    elif problem_type == "Maximize":
        model.obj = pyo.Objective(
            expr=objective_expression,
            sense=pyo.maximize,
        )
    else:
        raise ValueError(f"Unknown problem type: {problem_type}")

    model.cons = pyo.ConstraintList()

    for a, b, operator, rhs in constraints:
        expression = a * model.x + b * model.y

        if operator == "<=":
            model.cons.add(expression <= rhs)
        elif operator == ">=":
            model.cons.add(expression >= rhs)
        elif operator == "=":
            model.cons.add(expression == rhs)
        else:
            raise ValueError(f"Invalid constraint operator: {operator}")

    solver = create_highs_solver()

    # Do not load a solution until the termination condition is known.
    result = solver.solve(
        model,
        tee=False,
        load_solutions=False,
    )

    termination = result.solver.termination_condition

    if termination == pyo.TerminationCondition.infeasible:
        raise RuntimeError("The model is infeasible.")

    if termination in {
        pyo.TerminationCondition.unbounded,
        pyo.TerminationCondition.infeasibleOrUnbounded,
    }:
        raise RuntimeError("The model is unbounded or infeasible.")

    if termination == pyo.TerminationCondition.maxTimeLimit:
        raise RuntimeError(
            "The solver reached the 10-second time limit before proving optimality."
        )

    if termination != pyo.TerminationCondition.optimal:
        raise RuntimeError(
            f"The solver did not return an optimal solution. "
            f"Termination condition: {termination}"
        )

    # Load the verified optimal solution into the Pyomo model.
    model.solutions.load_from(result)

    return model, result


# =================== CALLBACKS ===================
def keep_expander_open(index):
    st.session_state["open_expander"] = index
    st.session_state["model_solved"] = False


def invalidate_solution():
    """Discard a stored result when any model input changes."""
    st.session_state["model_solved"] = False


# =================== SIDEBAR ===================
st.sidebar.title("Configuration")

problem_type = st.sidebar.selectbox(
    "Problem type",
    ["Minimize", "Maximize"],
    key="problem_type_widget",
    on_change=invalidate_solution,
)

n_constraints = st.sidebar.number_input(
    "Number of constraints",
    min_value=1,
    max_value=20,
    value=2,
    step=1,
    key="n_constraints_widget",
    on_change=invalidate_solution,
)

st.sidebar.markdown("---")
st.sidebar.subheader("Optimal solution")

if st.session_state.get("model_solved", False):
    st.sidebar.write(
        f"Solver status: {st.session_state.get('solver_status', '')}"
    )
    st.sidebar.write(f"x* = {st.session_state['x_opt']:.4f}")
    st.sidebar.write(f"y* = {st.session_state['y_opt']:.4f}")
    st.sidebar.write(f"Z* = {st.session_state['z_opt']:.4f}")
else:
    st.sidebar.info("Solve the model to view the solution here.")


# =================== MAIN CONTENT ===================
st.markdown("<h2>Graphical Method</h2>", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)


# =================== VARIABLE DOMAINS ===================
st.subheader("Variable Domains")
col1, col2 = st.columns(2)

with col1:
    x_type = st.selectbox(
        "Domain of X",
        ["Nonnegative Real", "Nonnegative Integer", "Binary"],
        key="x_type_widget",
        on_change=invalidate_solution,
    )

with col2:
    y_type = st.selectbox(
        "Domain of Y",
        ["Nonnegative Real", "Nonnegative Integer", "Binary"],
        key="y_type_widget",
        on_change=invalidate_solution,
    )


# =================== OBJECTIVE FUNCTION ===================
st.subheader("Objective Function")
col_objective_inputs, _ = st.columns([2, 3])

with col_objective_inputs:
    c1 = st.number_input(
        "Coefficient of X",
        value=3.0,
        key="c1_widget",
        on_change=invalidate_solution,
    )

    c2 = st.number_input(
        "Coefficient of Y",
        value=5.0,
        key="c2_widget",
        on_change=invalidate_solution,
    )

sense_tex = r"\min" if problem_type == "Minimize" else r"\max"
st.latex(rf"{sense_tex}\ Z = {c1}x + {c2}y")

st.markdown("<hr>", unsafe_allow_html=True)


# =================== CONSTRAINTS ===================
st.subheader("Constraints")
constraints = []

for k in range(int(n_constraints)):
    a_preview = st.session_state.get(f"a{k}_widget", 1.0)
    b_preview = st.session_state.get(f"b{k}_widget", 1.0)
    operator_preview = st.session_state.get(f"sent{k}_widget", "<=")
    rhs_preview = st.session_state.get(f"rhs{k}_widget", 8.0)

    title = (
        f"Constraint {k + 1}: "
        f"{equation_text(a_preview, b_preview, operator_preview, rhs_preview)}"
    )

    expanded_now = st.session_state.get("open_expander") == k

    with st.expander(title, expanded=expanded_now):
        col_a, col_b, col_operator, col_rhs = st.columns(4)

        with col_a:
            a = st.number_input(
                f"X coefficient in C{k + 1}",
                value=float(a_preview),
                key=f"a{k}_widget",
                on_change=keep_expander_open,
                args=(k,),
            )

        with col_b:
            b = st.number_input(
                f"Y coefficient in C{k + 1}",
                value=float(b_preview),
                key=f"b{k}_widget",
                on_change=keep_expander_open,
                args=(k,),
            )

        with col_operator:
            operator_options = ["<=", ">=", "="]
            operator_index = (
                operator_options.index(operator_preview)
                if operator_preview in operator_options
                else 0
            )

            operator = st.selectbox(
                f"Operator in C{k + 1}",
                operator_options,
                index=operator_index,
                key=f"sent{k}_widget",
                on_change=keep_expander_open,
                args=(k,),
            )

        with col_rhs:
            rhs = st.number_input(
                f"RHS in C{k + 1}",
                value=float(rhs_preview),
                key=f"rhs{k}_widget",
                on_change=keep_expander_open,
                args=(k,),
            )

        st.latex(rf"{a}x + {b}y\ {operator}\ {rhs}")

    a_current = st.session_state.get(f"a{k}_widget", 1.0)
    b_current = st.session_state.get(f"b{k}_widget", 1.0)
    operator_current = st.session_state.get(f"sent{k}_widget", "<=")
    rhs_current = st.session_state.get(f"rhs{k}_widget", 8.0)

    constraints.append(
        (
            float(a_current),
            float(b_current),
            operator_current,
            float(rhs_current),
        )
    )


# =================== SOLVE BUTTON ===================
if st.button("Solve and plot", type="primary"):
    st.session_state["model_solved"] = False

    with st.spinner("Solving model..."):
        try:
            model, result = build_and_solve_model(
                c1,
                c2,
                constraints,
                problem_type,
                x_type,
                y_type,
            )

            x_opt = float(pyo.value(model.x))
            y_opt = float(pyo.value(model.y))
            z_opt = float(pyo.value(model.obj))
            status = str(result.solver.termination_condition)

            continuous_model = (
                x_type == "Nonnegative Real"
                and y_type == "Nonnegative Real"
            )

            duals = []

            if continuous_model:
                for i, constraint_object in enumerate(
                    model.cons.values(),
                    start=1,
                ):
                    a_i, b_i, operator_i, rhs_i = constraints[i - 1]
                    equation_i = equation_text(
                        a_i,
                        b_i,
                        operator_i,
                        rhs_i,
                    )

                    dual_value = model.dual.get(constraint_object, None)

                    if dual_value is not None:
                        dual_value = float(dual_value)

                    duals.append([equation_i, dual_value])

            vertices = enumerate_vertices(
                constraints,
                x_type,
                y_type,
            )

            if continuous_model:
                ranges = coefficient_ranges(
                    vertices,
                    x_opt,
                    y_opt,
                    c1,
                    c2,
                    problem_type,
                )
                rhs_ranges = continuous_rhs_sensitivity_ranges(
                    constraints,
                    x_opt,
                    y_opt,
                    c1,
                    c2,
                    problem_type,
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
            st.session_state["c1"] = float(c1)
            st.session_state["c2"] = float(c2)
            st.session_state["duals"] = duals
            st.session_state["vertices"] = vertices
            st.session_state["coefficient_ranges_result"] = ranges
            st.session_state["rhs_sensitivity_result"] = rhs_ranges
            st.session_state["problem_type_value"] = problem_type
            st.session_state["x_type_value"] = x_type
            st.session_state["y_type_value"] = y_type

            st.success("Model solved successfully.")

        except Exception as error:
            st.session_state["model_solved"] = False
            st.error(f"Error solving the model: {error}")


# =================== DISPLAY GRAPH AND ANALYSIS ===================
if st.session_state.get("model_solved", False):
    solved_constraints = st.session_state["constraints"]
    x_opt = st.session_state["x_opt"]
    y_opt = st.session_state["y_opt"]
    c1_result = st.session_state["c1"]
    c2_result = st.session_state["c2"]
    vertices = st.session_state.get("vertices", [])

    st.markdown("<hr>", unsafe_allow_html=True)
    st.subheader("Feasible Region Plot")

    x_candidates = (
        [x_opt] + [point[0] for point in vertices]
        if vertices
        else [x_opt]
    )

    y_candidates = (
        [y_opt] + [point[1] for point in vertices]
        if vertices
        else [y_opt]
    )

    max_x = max(x_candidates + [10.0])
    max_y = max(y_candidates + [10.0])
    limit = 1.25 * max(max_x, max_y, 1.0)

    x_grid = np.linspace(0.0, limit, 400)

    figure = go.Figure()

    # Feasible polygon for the continuous relaxation.
    if len(vertices) >= 3:
        sorted_vertices = sort_vertices(vertices)

        x_polygon = [point[0] for point in sorted_vertices]
        y_polygon = [point[1] for point in sorted_vertices]

        x_polygon.append(sorted_vertices[0][0])
        y_polygon.append(sorted_vertices[0][1])

        figure.add_trace(
            go.Scatter(
                x=x_polygon,
                y=y_polygon,
                mode="lines",
                fill="toself",
                fillcolor="rgba(0,150,255,0.25)",
                line=dict(
                    color="rgba(0,150,255,0.9)",
                    width=2,
                ),
                name="Feasible region",
            )
        )

    # Constraint boundary lines.
    for a, b, operator, rhs in solved_constraints:
        if abs(b) > 1e-8:
            y_line = (rhs - a * x_grid) / b

            figure.add_trace(
                go.Scatter(
                    x=x_grid,
                    y=y_line,
                    mode="lines",
                    name=equation_text(a, b, operator, rhs),
                )
            )

        elif abs(a) > 1e-8:
            x_line = rhs / a

            figure.add_trace(
                go.Scatter(
                    x=[x_line, x_line],
                    y=[0.0, limit],
                    mode="lines",
                    name=equation_text(a, 0.0, operator, rhs),
                )
            )

    # Optimal solution.
    figure.add_trace(
        go.Scatter(
            x=[x_opt],
            y=[y_opt],
            mode="markers+text",
            text=["Optimum"],
            textposition="top right",
            marker=dict(
                size=10,
                color="red",
            ),
            name="Optimal solution",
        )
    )

    # Objective-function line through the optimum.
    if abs(c2_result) > 1e-8:
        optimal_objective_value = c1_result * x_opt + c2_result * y_opt
        y_objective = (
            optimal_objective_value - c1_result * x_grid
        ) / c2_result

        figure.add_trace(
            go.Scatter(
                x=x_grid,
                y=y_objective,
                mode="lines",
                line=dict(
                    dash="dash",
                    color="red",
                ),
                name="Objective function at Z*",
            )
        )

    elif abs(c1_result) > 1e-8:
        x_line = (
            c1_result * x_opt + c2_result * y_opt
        ) / c1_result

        figure.add_trace(
            go.Scatter(
                x=[x_line, x_line],
                y=[0.0, limit],
                mode="lines",
                line=dict(
                    dash="dash",
                    color="red",
                ),
                name="Objective function at Z*",
            )
        )

    # Axis limits.
    if len(vertices) >= 1:
        xs = [point[0] for point in vertices] + [x_opt]
        ys = [point[1] for point in vertices] + [y_opt]

        x_min = min(0.0, min(xs) - 1.0)
        x_max = max(xs) + 5.0
        y_min = min(0.0, min(ys) - 1.0)
        y_max = max(ys) + 5.0

        figure.update_xaxes(range=[x_min, x_max])
        figure.update_yaxes(range=[y_min, y_max])
    else:
        figure.update_xaxes(range=[0.0, limit])
        figure.update_yaxes(range=[0.0, limit])

    figure.update_layout(
        width=850,
        height=600,
        title="Feasible Region and Optimal Solution",
        xaxis_title="x",
        yaxis_title="y",
        legend=dict(x=0.68, y=1.0),
    )

    st.plotly_chart(
        figure,
        use_container_width=True,
    )


    # =================== SHADOW PRICES ===================
    st.markdown("<hr>", unsafe_allow_html=True)

    x_type_value = st.session_state["x_type_value"]
    y_type_value = st.session_state["y_type_value"]

    continuous_result = (
        x_type_value == "Nonnegative Real"
        and y_type_value == "Nonnegative Real"
    )

    if continuous_result:
        st.subheader("Shadow Prices")

        duals = st.session_state.get("duals", [])

        if isinstance(duals, list) and len(duals) > 0:
            dual_table = {
                "Constraint": [row[0] for row in duals],
                "Shadow price (Dual)": [
                    "N/A" if row[1] is None else f"{row[1]:.4f}"
                    for row in duals
                ],
            }

            st.table(dual_table)
        else:
            st.info("No shadow prices are available.")

    else:
        st.subheader("Shadow Prices")
        st.info(
            "Classical LP shadow prices are not available for integer or binary models."
        )


    # =================== RHS SENSITIVITY ===================
    st.markdown("<hr>", unsafe_allow_html=True)
    st.subheader("Constraint RHS Sensitivity")

    def format_sensitivity_value(value):
        if value is None:
            return "N/A"
        if np.isneginf(value):
            return "-∞"
        if np.isposinf(value):
            return "+∞"
        return f"{float(value):.4f}"

    rhs_ranges = st.session_state.get("rhs_sensitivity_result", None)

    if continuous_result:
        if rhs_ranges is None:
            st.info(
                "The RHS sensitivity ranges could not be calculated for this LP solution."
            )
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

            duals = st.session_state.get("duals", [])

            for index, (a, b, operator, rhs) in enumerate(solved_constraints):
                sensitivity = rhs_ranges[index] if index < len(rhs_ranges) else None
                rhs_table["Constraint"].append(
                    equation_text(a, b, operator, rhs)
                )
                rhs_table["Current RHS"].append(f"{float(rhs):.4f}")

                if sensitivity is None:
                    rhs_table["Minimum RHS"].append("N/A")
                    rhs_table["Maximum RHS"].append("N/A")
                    rhs_table["Allowable decrease"].append("N/A")
                    rhs_table["Allowable increase"].append("N/A")
                else:
                    rhs_table["Minimum RHS"].append(
                        format_sensitivity_value(sensitivity["rhs_min"])
                    )
                    rhs_table["Maximum RHS"].append(
                        format_sensitivity_value(sensitivity["rhs_max"])
                    )
                    rhs_table["Allowable decrease"].append(
                        format_sensitivity_value(sensitivity["allowable_decrease"])
                    )
                    rhs_table["Allowable increase"].append(
                        format_sensitivity_value(sensitivity["allowable_increase"])
                    )

                dual_value = duals[index][1] if index < len(duals) else None
                rhs_table["Shadow price"].append(
                    "N/A" if dual_value is None else f"{float(dual_value):.4f}"
                )

            st.table(rhs_table)
            st.caption(
                "Continuous LP ranges are one-at-a-time RHS sensitivity intervals. "
                "Within each interval, an optimal basis compatible with the current "
                "solution remains valid. Other RHS values are held fixed."
            )

    else:
        st.info(
            "For integer, binary, or mixed models, classical LP RHS sensitivity "
            "does not apply. The analysis below uses discrete reoptimization to "
            "find the stability interval of the current optimal solution."
        )

        if st.button(
            "Calculate discrete RHS sensitivity",
            key="calculate_discrete_rhs_sensitivity",
        ):
            with st.spinner("Calculating discrete RHS sensitivity..."):
                try:
                    rhs_ranges = discrete_rhs_sensitivity_ranges(
                        solved_constraints,
                        x_opt,
                        y_opt,
                        c1_result,
                        c2_result,
                        st.session_state["problem_type_value"],
                        x_type_value,
                        y_type_value,
                    )
                    st.session_state["rhs_sensitivity_result"] = rhs_ranges
                except Exception as error:
                    st.error(f"Error calculating RHS sensitivity: {error}")
                    rhs_ranges = None

        rhs_ranges = st.session_state.get("rhs_sensitivity_result", None)

        if rhs_ranges is not None:
            rhs_table = {
                "Constraint": [],
                "Current RHS": [],
                "Minimum RHS": [],
                "Maximum RHS": [],
                "Allowable decrease": [],
                "Allowable increase": [],
                "Status": [],
            }

            for index, (a, b, operator, rhs) in enumerate(solved_constraints):
                sensitivity = rhs_ranges[index] if index < len(rhs_ranges) else None
                rhs_table["Constraint"].append(
                    equation_text(a, b, operator, rhs)
                )
                rhs_table["Current RHS"].append(f"{float(rhs):.4f}")

                if sensitivity is None:
                    rhs_table["Minimum RHS"].append("N/A")
                    rhs_table["Maximum RHS"].append("N/A")
                    rhs_table["Allowable decrease"].append("N/A")
                    rhs_table["Allowable increase"].append("N/A")
                    rhs_table["Status"].append("N/A")
                else:
                    rhs_table["Minimum RHS"].append(
                        format_sensitivity_value(sensitivity["rhs_min"])
                    )
                    rhs_table["Maximum RHS"].append(
                        format_sensitivity_value(sensitivity["rhs_max"])
                    )
                    rhs_table["Allowable decrease"].append(
                        format_sensitivity_value(sensitivity["allowable_decrease"])
                    )
                    rhs_table["Allowable increase"].append(
                        format_sensitivity_value(sensitivity["allowable_increase"])
                    )
                    rhs_table["Status"].append(sensitivity.get("status", ""))

            st.table(rhs_table)
            st.caption(
                "Discrete/MIP ranges are not classical shadow-price intervals. "
                "Tightening is limited by feasibility of the current integer solution; "
                "relaxation is limited by the first breakpoint at which a strictly better "
                "solution can enter. A finite breakpoint should be interpreted as a change "
                "point; +∞ or -∞ means no better solution was found after removing that row."
            )


    # =================== COEFFICIENT RANGES ===================
    st.markdown("<hr>", unsafe_allow_html=True)
    st.subheader("Objective Function Coefficients")

    ranges = st.session_state.get(
        "coefficient_ranges_result",
        None,
    )

    if not continuous_result:
        st.info(
            "Objective-coefficient range analysis is available only for the continuous LP model."
        )

    elif ranges is None:
        st.info(
            "The coefficient ranges could not be calculated for this solution."
        )

    else:
        def format_interval(lower, upper):
            def format_bound(value):
                if np.isneginf(value):
                    return "-∞"
                if np.isposinf(value):
                    return "+∞"
                return f"{value:.4f}"

            return format_bound(lower), format_bound(upper)

        c1_lower, c1_upper = format_interval(*ranges["c1"])
        c2_lower, c2_upper = format_interval(*ranges["c2"])

        sensitivity_table = {
            "Coefficient": [
                "c1 (coefficient of x)",
                "c2 (coefficient of y)",
            ],
            "Current value": [
                f"{c1_result:.4f}",
                f"{c2_result:.4f}",
            ],
            "Lower bound": [
                c1_lower,
                c2_lower,
            ],
            "Upper bound": [
                c1_upper,
                c2_upper,
            ],
        }

        st.table(sensitivity_table)


# Close main container
st.markdown("</div>", unsafe_allow_html=True)
