# Graphical Method for Linear Programming

A Streamlit application for building, solving, and visualizing two-variable Linear Programming (LP) and Mixed-Integer Linear Programming (MILP) models using the graphical method.

The application combines:

- **Streamlit** for the graphical interface;
- **Pyomo** for mathematical-model construction;
- **HiGHS** as the optimization solver;
- **NumPy** for numerical calculations;
- **Plotly** for interactive visualization of the feasible region and optimal solution.

The application is designed specifically for models with **two decision variables**, \(x\) and \(y\).

---

# Main Features

The application allows the user to:

- choose a minimization or maximization problem;
- define the nature of variables \(x\) and \(y\);
- enter the coefficients of the objective function;
- define up to 20 constraints;
- solve the model using HiGHS;
- visualize the feasible region;
- identify the optimal solution;
- display the objective-function line at the optimum;
- calculate shadow prices for continuous LP models;
- estimate objective-function coefficient ranges for continuous LP models.

---

# Model Structure

The application solves models of the form:

\[
\min/\max \quad Z=c_1x+c_2y
\]

subject to:

\[
a_r x+b_r y
\begin{cases}
\leq\\
\geq\\
=
\end{cases}
d_r,
\qquad r=1,\ldots,m
\]

with variable domains selected independently for \(x\) and \(y\).

---

# Decision Variables

The application always works with two decision variables:

\[
x,\qquad y.
\]

Each variable can independently be defined as:

- **Nonnegative Real**
- **Nonnegative Integer**
- **Binary**

The corresponding mathematical domains are:

### Nonnegative Real

\[
x\geq0
\]

or:

\[
y\geq0.
\]

### Nonnegative Integer

\[
x\in\mathbb{Z}_{+}
\]

or:

\[
y\in\mathbb{Z}_{+}.
\]

### Binary

\[
x\in\{0,1\}
\]

or:

\[
y\in\{0,1\}.
\]

The two variables do not need to have the same domain.

For example:

```text
x: Nonnegative Integer
y: Binary
```

is allowed.

---

# Objective Function

The user enters the coefficients:

```text
Coefficient of X
Coefficient of Y
```

to construct:

\[
Z=c_1x+c_2y.
\]

The optimization sense can be selected as:

- **Minimize**
- **Maximize**

Example:

```text
Problem type: Maximize
Coefficient of X: 3
Coefficient of Y: 5
```

produces:

\[
\max Z=3x+5y.
\]

---

# Constraints

The application supports between:

```text
1 and 20 constraints
```

Each constraint is defined by four components:

- coefficient of \(x\);
- coefficient of \(y\);
- relational operator;
- right-hand side value.

Available relational operators are:

```text
<=
>=
=
```

For example:

```text
Coefficient of X: 2
Coefficient of Y: 3
Operator: <=
Right-hand side: 12
```

represents:

\[
2x+3y\leq12.
\]

Each constraint is shown inside a collapsible panel.

The active constraint panel remains open while its coefficients or operator are modified.

---

# Solver

The optimization model is constructed dynamically using **Pyomo** and solved with:

### HiGHS

The solver is called through:

```python
pyo.SolverFactory("appsi_highs")
```

No solver selection is required from the user.

---

# Solving the Model

After defining:

- problem type;
- variable domains;
- objective-function coefficients;
- constraints;

click:

```text
Solve and plot
```

The application sends the model to HiGHS and stores:

- solver termination condition;
- optimal value of \(x\);
- optimal value of \(y\);
- optimal objective value \(Z^*\).

The sidebar then displays:

\[
x^*,\qquad y^*,\qquad Z^*.
\]

---

# Feasibility Analysis

The application contains an internal feasibility routine that evaluates candidate points against:

- all model constraints;
- nonnegativity conditions;
- binary bounds when applicable.

A point \((x,y)\) is considered feasible only if all restrictions are satisfied within a numerical tolerance.

For example, for:

\[
2x+y\leq10
\]

a candidate point must satisfy:

\[
2x+y\leq10.
\]

---

# Vertex Enumeration

To construct the feasible region, the application computes intersections between pairs of boundary equations.

For two boundaries:

\[
a_1x+b_1y=d_1
\]

and:

\[
a_2x+b_2y=d_2,
\]

the application solves the corresponding \(2\times2\) linear system.

Only intersection points that satisfy all constraints are retained.

Duplicate vertices are removed numerically.

---

# Feasible Region

When at least three feasible vertices are detected, the application:

1. computes the geometric center of the vertices;
2. orders them by polar angle;
3. connects them to construct the feasible polygon.

The resulting region is displayed using Plotly.

The graph includes:

- feasible region;
- boundary line of every constraint;
- optimal solution;
- objective-function line at \(Z^*\).

---

# Graphical Output

The main visualization includes:

### Constraint Lines

Every constraint is displayed as a line.

For:

\[
ax+by=d,
\]

if \(b\neq0\):

\[
y=\frac{d-ax}{b}.
\]

If \(b=0\), the constraint is represented as a vertical line:

\[
x=\frac{d}{a}.
\]

### Feasible Region

When a bounded polygon can be constructed, the feasible region is filled and highlighted.

### Optimal Point

The optimal solution is displayed as a highlighted point:

\[
(x^*,y^*).
\]

### Objective-Function Line

The application also plots the objective-function line associated with the optimal value:

\[
c_1x+c_2y=Z^*.
\]

This helps visualize the geometric relationship between the objective function and the optimal vertex.

---

# Continuous, Integer, and Binary Models

The application can solve models with:

- both variables continuous;
- both variables integer;
- both variables binary;
- mixed combinations.

Examples:

```text
x: Nonnegative Real
y: Nonnegative Real
```

```text
x: Nonnegative Integer
y: Nonnegative Integer
```

```text
x: Binary
y: Binary
```

```text
x: Nonnegative Integer
y: Binary
```

Although the graphical representation is continuous, the optimizer respects the selected Pyomo variable domains.

Therefore, for integer or binary models, the optimal solution returned by HiGHS satisfies the discrete domain restrictions even though the plotted feasible polygon represents the continuous relaxation geometrically.

---

# Shadow Prices

Shadow prices are available only when both variables are:

```text
Nonnegative Real
```

In that case, the application imports the dual values from Pyomo using:

```python
pyo.Suffix(direction=pyo.Suffix.IMPORT)
```

For each constraint, the application displays:

- constraint equation;
- corresponding dual value.

The shadow price measures the marginal change in the optimal objective value associated with a marginal change in the right-hand side of a constraint, under the usual sensitivity-analysis assumptions.

---

# Important Limitation of Shadow Prices

Dual values are not reported for models containing integer or binary variables.

When either variable is discrete, the application displays:

```text
Not available for integer/binary model
```

This is intentional.

Classical LP shadow-price interpretation is based on the continuous linear-programming relaxation and does not directly apply to general MILP models.

---

# Objective-Function Coefficient Analysis

For continuous LP models, the application estimates ranges for:

\[
c_1
\]

and:

\[
c_2
\]

such that the current optimal vertex remains optimal.

The analysis compares the objective value at the current optimal solution with the objective value at the other feasible vertices.

The output table contains:

- coefficient;
- current value;
- lower bound;
- upper bound.

For example:

| Coefficient | Current value | Lower bound | Upper bound |
|---|---:|---:|---:|
| \(c_1\) | 3.0000 | 2.0000 | 5.0000 |
| \(c_2\) | 5.0000 | 3.0000 | \(+\infty\) |

Infinite intervals are displayed using:

```text
-∞
+∞
```

---

# Important Limitation of Coefficient Ranges

The coefficient-range analysis is enabled only when both variables are continuous:

```text
x: Nonnegative Real
y: Nonnegative Real
```

For integer or binary models, the application does not perform this classical sensitivity analysis.

---

# Example

Consider:

\[
\max Z=3x+5y
\]

subject to:

\[
2x+y\leq8
\]

\[
x+2y\leq8
\]

\[
x,y\geq0.
\]

In the application, enter:

```text
Problem type:
Maximize
```

Objective coefficients:

```text
X: 3
Y: 5
```

Constraint 1:

```text
X coefficient: 2
Y coefficient: 1
Operator: <=
RHS: 8
```

Constraint 2:

```text
X coefficient: 1
Y coefficient: 2
Operator: <=
RHS: 8
```

Variable domains:

```text
X: Nonnegative Real
Y: Nonnegative Real
```

After solving, the application:

- obtains the optimal solution from HiGHS;
- plots the feasible polygon;
- identifies the optimal point;
- displays the objective-function line;
- reports shadow prices;
- calculates coefficient ranges.

---

# Interface Structure

## Sidebar

The sidebar contains:

- problem type;
- number of constraints;
- optimal-solution summary after solving.

The optimal-solution summary includes:

```text
Solver status
x*
y*
Z*
```

---

## Main Area

The main interface is organized into the following sections:

1. **Variable Nature**
2. **Objective Function**
3. **Constraints**
4. **Solve and Plot**
5. **Feasible Region Graph**
6. **Shadow Prices**
7. **Objective-Function Coefficients**

---

# Session State

Streamlit session state is used to preserve:

- active constraint expander;
- optimal solution;
- objective-function coefficients;
- constraints;
- dual values;
- feasible vertices;
- coefficient ranges;
- variable domains;
- problem type.

This allows the interface to preserve the optimization results across Streamlit reruns.

---

# Installation

Python 3.10 or later is recommended.

Install dependencies with:

```bash
pip install -r requirements.txt
```

Recommended `requirements.txt`:

```text
streamlit
pyomo
numpy
plotly
highspy
```

---

# Running the Application

If the main file is named:

```text
Graphical_Method.py
```

run:

```bash
streamlit run Graphical_Method.py
```

Streamlit will normally start the application at an address similar to:

```text
http://localhost:8501
```

---

# Suggested Project Structure

```text
Graphical_Method/
├── Graphical_Method.py
├── README.md
└── requirements.txt
```

---

# Recommended Workflow

1. Start the application.
2. Select **Minimize** or **Maximize**.
3. Select the domain of \(x\).
4. Select the domain of \(y\).
5. Enter the objective-function coefficients.
6. Select the number of constraints.
7. Enter each constraint.
8. Click **Solve and plot**.
9. Review \(x^*\), \(y^*\), and \(Z^*\).
10. Inspect the feasible-region graph.
11. For continuous LP models, review shadow prices.
12. For continuous LP models, review objective-coefficient ranges.

---

# Requirements

The application requires:

### Streamlit

Used to build the interactive user interface.

### Pyomo

Used to define the optimization model.

### HiGHS / highspy

Used to solve LP and MILP models.

### NumPy

Used for:

- vertex calculations;
- numerical arrays;
- geometric ordering;
- sensitivity calculations.

### Plotly

Used to create the interactive feasible-region graph.

---

# Mathematical Scope

The current version supports optimization models with:

- exactly two decision variables;
- linear objective function;
- linear constraints;
- minimization or maximization;
- continuous variables;
- nonnegative integer variables;
- binary variables;
- mixed variable domains;
- equality and inequality constraints.

The application does not support:

- more than two decision variables;
- nonlinear objective functions;
- nonlinear constraints;
- classical graphical visualization in dimensions greater than two.

---

# Technical Notes

## Numerical Tolerance

Feasibility checks use a numerical tolerance to reduce floating-point errors.

## Parallel Constraints

When two boundary equations are parallel, their determinant is approximately zero and no intersection point is generated.

## Duplicate Vertices

Intersection points are rounded numerically before duplicate removal.

## Plot Limits

The application determines graphical limits from:

- feasible vertices;
- optimal solution;
- a minimum default visualization range.

Additional margins are applied around the detected points to improve readability.

---

# Current Scope

The current application includes:

- two-variable LP/MILP formulation;
- independent variable domains;
- objective-function definition;
- up to 20 linear constraints;
- HiGHS optimization;
- feasible-point validation;
- vertex enumeration;
- feasible-region visualization;
- optimal-point visualization;
- objective-function line at the optimum;
- shadow-price analysis for continuous models;
- objective-coefficient sensitivity analysis for continuous models.

---

# Author

**M.Sc. Dilan Mogollón**
