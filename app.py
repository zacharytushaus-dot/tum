
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from engine import Inputs, Tier, run_monte_carlo, IntervCost

# ---------- Percent helpers (UI shows %, engine gets 0–1) ----------
def _infer_digits(step_pct: float) -> int:
    # 5.0 -> 0, 0.1 -> 1, 0.01 -> 2, etc.
    s = str(step_pct)
    return len(s.split(".")[1].rstrip("0")) if "." in s else 0

def pct_slider(label, *, min_pct=0.0, max_pct=100.0, value_pct=0.0,
               step_pct=0.1, key=None, help=None, sidebar=True,
               period=None, digits=None, fmt=None):
    """
    digits: number of decimal places to display (overrides inference)
    fmt: full Streamlit format string, e.g. "%.0f%%" (overrides digits)
    """
    unit = f" (% {period})" if period else " (%)"
    if fmt is None:
        d = _infer_digits(step_pct) if digits is None else digits
        fmt = f"%.{d}f%%"

    args = dict(
        label=label + unit,
        min_value=float(min_pct),
        max_value=float(max_pct),
        value=float(value_pct),
        step=float(step_pct),
        format=fmt,
        key=key, help=help
    )
    v = (st.sidebar.slider(**args) if sidebar else st.slider(**args))
    return v / 100.0  # normalize to 0–1 for the engine

def pct_number(label, *, value_pct=0.0, step_pct=0.1, key=None, help=None,
               sidebar=True, period=None, min_pct=0.0, max_pct=100.0,
               digits=None, fmt=None):
    """
    number_input can't show % in the format, so we keep % in the label.
    """
    unit = f" (% {period})" if period else " (%)"
    if fmt is None:
        d = _infer_digits(step_pct) if digits is None else digits
        fmt = f"%.{d}f"

    args = dict(
        label=label + unit,
        min_value=float(min_pct),
        max_value=float(max_pct),
        value=float(value_pct),
        step=float(step_pct),
        format=fmt,
        key=key, help=help
    )
    v = (st.sidebar.number_input(**args) if sidebar else st.number_input(**args))
    return v / 100.0

st.set_page_config(page_title="Time Utility Model", layout="wide")

# Persist results across reruns so charts don't error before you run.
if "results" not in st.session_state:
    st.session_state.results = None

st.title("Time Utility Model")

# ------------------ Sidebar: profile & presets ------------------
st.sidebar.header("1) Your Profile & Presets")

age = st.sidebar.number_input("Your Age", min_value=18, max_value=95, value=21, step=1)
sex = st.sidebar.selectbox("Sex", ["Male", "Female"])

preset = st.sidebar.selectbox(
    "Preset",
    ["None", "Athlete", "Budget Health", "Max Longevity"],
    index=1
)

# Default risk multipliers (HR×adherence at 100%)
BASE_RISK_MULT = {
    "Consistent Sleep": 0.88,
    "Frequent Exercise": 0.68,
    "Frequent Sauna": 0.90, # ≥3–4 sessions/week; consider 0.75–0.80 for 4–7/wk
    "Mediterranean Diet": 0.77,
    "Meditation": 0.93,
    "Red-Light Therapy": 0.98,
    "Smoking Currently": 2.5, # harmful, editable in UI later if we want
    "Heavy Drinking": 1.25, # 3-4 drinks/day; use 1.35 for "very heavy"
}

# Preset → default toggle set
PRESET_TOGGLES = {
    "None":               {"Consistent Sleep": False, "Frequent Exercise": False, "Mediterranean Diet": False, "Meditation": False, "Red-Light Therapy": False, "Frequent Sauna": False, "Smoking Currently": False, "Heavy Drinking": False},
    "Athlete":            {"Consistent Sleep": True,  "Frequent Exercise": True,  "Mediterranean Diet": True,  "Meditation": True,  "Red-Light Therapy": False, "Frequent Sauna": False, "Smoking Currently": False, "Heavy Drinking": False},
    "Budget Health":      {"Consistent Sleep": True,  "Frequent Exercise": True,  "Mediterranean Diet": False, "Meditation": True,  "Red-Light Therapy": False, "Frequent Sauna": False, "Smoking Currently": False, "Heavy Drinking": False},
    "Max Longevity":      {"Consistent Sleep": True,  "Frequent Exercise": True,  "Mediterranean Diet": True,  "Meditation": True,  "Red-Light Therapy": True, "Frequent Sauna": True, "Smoking Currently": False, "Heavy Drinking": False},
}

st.sidebar.header("2) Your Health Habits")
toggles = {}
for name, default_on in PRESET_TOGGLES[preset].items():
    toggles[name] = st.sidebar.checkbox(name, value=default_on)

adherence = pct_slider("Adherence to your habits",
                       value_pct=75.0, step_pct=5.0, digits=0,)
st.sidebar.caption("How likely you are to commit to your habits")

# --- Weight status (mutually exclusive) ---
st.sidebar.subheader("Weight status")
weight_label = st.sidebar.selectbox(
    "Pick one",
    ["None / Healthy Range",
     "Overweight (BMI 27.5-29.9)",
     "High Body Weight (BMI 30-34.9)",
     "Very High Body Weight (BMI ≥35)"],
    index=0,
    help="If BMI misfits you (very muscular), use waist guidance: central obesity ≈ waist-to-height ≥0.60 or large waist (≥102 cm men, ≥88 cm women)."
)

WEIGHT_HR = {
    "None / Healthy Range":            1.00,
    "Overweight (BMI 27.5-29.9)":      1.20,  # Lancet IPD meta-analysis
    "High Body Weight (BMI 30-34.9)":  1.45,
    "Very High Body Weight (BMI ≥35)": 1.94,
}

# Final multipliers to feed engine (Excel rule)

# Canonical keys so UI labels, costs, and HR multipliers stay in sync
CANON = {
    "Consistent Sleep":         "sleep",
    "Frequent Exercise":        "exercise",
    "Frequent Sauna":           "sauna",
    "Mediterranean Diet":       "mediterraneandiet",
    "Meditation":               "meditation",
    "Red-Light Therapy":        "redlight",
    "Smoking Currently":        "smoker",
    "Heavy Drinking":           "heavyalcohol",
    "Weight status":            "weight",
}

intervention_on = {}
lifestyle_HRs = {}
for name, base in BASE_RISK_MULT.items():
    key = CANON[name]
    on = bool(toggles[name])
    intervention_on[key] = on

    if not on or base == 1.0:
        m = 1.0
    elif base < 1.0:
        # Protective habit → scale by adherence (80% adherence = 80% of the benefit)
        m = 1.0 - adherence * (1.0 - base)
    else:
        # Harmful exposure → do NOT scale by adherence
        m = float(base)

    lifestyle_HRs[key] = float(m)

# Add Weight status as its own factor (scaled by the same adherence rule)
w_on = (weight_label != "None / Healthy Range")
base_w = WEIGHT_HR[weight_label]
w_mult = base_w if w_on else 1.0  # weight is a state/exposure → no adherence scaling

lifestyle_HRs["weight"] = float(w_mult)
intervention_on["weight"] = bool(w_on)

# ------------------ Sidebar: habit costs (annual only) ------------------
with st.sidebar.expander("Health Habit Costs", expanded=False):
    st.caption("One field per habit: dollars per year (negative allowed if it saves you money). We apply healthcare inflation automatically each year.")

    # canonical keys used everywhere (from CANON above)
    ORDER = ["sleep", "exercise", "sauna", "mediterraneandiet", "meditation", "redlight", "smoker", "heavyalcohol", "weight"]
    LABEL_FOR = {v: k for k, v in CANON.items()}  # reverse map: key -> UI label

    # Defaults: previous one-time + recurring, converted to annual (keeps totals comparable)
    ANNUAL_DEFAULT = {
        "sleep":              50.0,         # ≈ $50/yr
        "exercise":           240.0,      # gym, etc.
        "sauna":              300.0,      # electricity/entry
        "mediterraneandiet":  5400.0,     # extra groceries vs baseline
        "meditation":         70.0,       # app subscription
        "redlight":           150.0,        # ≈ $157/yr
        "smoker":             800.0,      # cigarettes; make negative if quitting = savings
        "heavyalcohol":       1000.0,     # user-editable
        "weight":             0.0,         # no expense by default
    }

    annual_inputs = {}
    for key in ORDER:
        label = LABEL_FOR[key]
        if key == "weight":
            enabled = (weight_label != "None / Healthy Range")
        else:
            enabled = toggles[label]

        annual_inputs[key] = st.number_input(
            f"{label} — $ per year",
            value=float(ANNUAL_DEFAULT[key]),
            step=50.0,
            min_value=-1_000_000.0,  # allow negative for savings
            key=f"{key}_annual",
            disabled=not enabled,
        )

    # Feed the engine: horizon=1, one_time=0, recurring=annual
    intervention_costs = {
        k: IntervCost(horizon=1, one_time=0.0, recurring=float(annual_inputs[k]))
        for k in ORDER
    }

# ------------------ Sidebar: finance ------------------
st.sidebar.header("3) Your Finances")
start_capital = st.sidebar.number_input("Starting Capital ($)", min_value=0, value=10_000, step=1_000)
di0 = st.sidebar.number_input("Yearly Spending Budget ($)", min_value=0, value=10_000, step=1_000)
ret = pct_slider("Portfolio Return", min_pct=0.0, max_pct=15.0,
                 value_pct=5.0, step_pct=0.1, digits=1, period="per year")
income_growth = pct_slider("Spending Growth", min_pct=0.0, max_pct=10.0,
                           value_pct=3.0, step_pct=0.1, digits=1, period="per year")
hc_infl = pct_slider("Healthcare Inflation", min_pct=0.0, max_pct=10.0,
                     value_pct=3.0, step_pct=0.1, digits=1, period="per year")

# ------------------ Sidebar: breakthroughs ------------------
st.sidebar.header("4) Future Treatments, Ranked from Tiers 1-3")
def tier_inputs(label, p0, g_pp, cap, cost, years_gained):
    st.sidebar.subheader(label)
    kid = label.lower().replace(" ", "_")
    c = st.sidebar.number_input("Estimated Cost Today ($)", min_value=0, value=cost, step=1_000, key=f"{kid}_cost")
    y = st.sidebar.number_input("Years Added (after purchase)", min_value=0.0, value=years_gained, step=0.1, key=f"{kid}_years")
    p = pct_slider("Annual Chance of Occurring", min_pct=0.0, max_pct=50.0,
               value_pct=p0*100.0, step_pct=0.5, digits=1, key=f"{kid}_p0", period="per year")
    g = pct_slider("Increase in Chance Per Missed Year", min_pct=0.0, max_pct=5.0,
               value_pct=g_pp*100.0, step_pct=0.05, digits=2, key=f"{kid}_gpp", period="per missed year")
    cap_ = pct_slider("Max Annual Chance", min_pct=0.0, max_pct=100.0,
                  value_pct=cap*100.0, step_pct=0.5, digits=1, key=f"{kid}_cap", period="per year")
    return Tier(
        cost_today=c,
        years_gain=y,
        base_prob=p,
        growth_per_year=g,
        cap_prob=cap_,
    )

tier1 = tier_inputs("Tier 1", p0=0.03,  g_pp=0.0015, cap=0.10, cost=12_000,     years_gained=0.7)
tier2 = tier_inputs("Tier 2", p0=0.006, g_pp=0.0015, cap=0.10, cost=550_000,   years_gained=2.5)
tier3 = tier_inputs("Tier 3", p0=0.0012,g_pp=0.0015, cap=0.10, cost=2_400_000, years_gained=7.0)

# ------------------ Sidebar: longevity params ------------------
st.sidebar.header("5) Longevity Parameters")
lambda_plateau = st.sidebar.number_input("Late-age Risk Plateau λ", min_value=0.0, max_value=5.0, value=0.6, step=0.05)
drift_days = st.sidebar.number_input("Frontier Age Drift (days per year)", min_value=0.0, max_value=200.0, value=15.0, step=1.0)
le_improve = pct_slider("Life Expectancy Growth", min_pct=0.0, max_pct=2.0,
                        value_pct=0.2, step_pct=0.5, digits=1, period="per year")
max_age_today = st.sidebar.number_input("Frontier Age Today", min_value=100.0, max_value=130.0, value=119.0, step=0.5)

# ------------------ Sidebar: simulation ------------------
st.sidebar.header("6) Simulation")
draws = st.sidebar.slider("Simulation Runs", 1000, 50000, 5000, step=1000)
seed = st.sidebar.number_input("Random Seed (reproducible)", min_value=0, max_value=1_000_000, value=49, step=1)

if st.sidebar.button("Run Simulation", type="primary"):
    inputs = Inputs(
        start_age=int(age),
        sex=sex,
        draws=int(draws),
        investment_return=float(ret),
        start_capital=float(start_capital),
        # Excel-parity finance
        discretionary_income=float(di0),
        income_growth=float(income_growth),
        annual_contrib=0.0,                  # ignored when discretionary_income is provided
        contrib_growth=0.0,
        lambdaP=float(lambda_plateau),
        frontier_drift_days=float(drift_days),
        le_trend=float(le_improve),
        max_age_today=int(max_age_today),
        hc_inflation=float(hc_infl),
        lifestyle_HRs=lifestyle_HRs,
        adherence=float(adherence),
        tiers=[tier1, tier2, tier3],
        tier_repeatable=True,                 # repeatable breakthroughs
        intervention_costs=intervention_costs,
        intervention_on=intervention_on,
        grid_max_age=max(int(age) + 126, 170),
        seed=int(seed),
    )
    st.session_state.results = run_monte_carlo(inputs)

# ------------------ Render results (if any) ------------------
out = st.session_state.results

if out is None:
    st.info("Describe yourself on the left, then click **Run Simulation**")
    st.stop()  # ← do not execute the rest of the page until results exist
else:
    col1, col2, col3 = st.columns(3)
    median_life = float(np.median(out["projected_life"]))
    p5, p95 = np.percentile(out["projected_life"], [5,95])
    median_net = float(np.median(out["net_worth"]))

    with col1:
        st.metric(
            "Projected Life (median)",
            f"{median_life:.0f} years",
            help="The age when your biological age first equals or exceeds society's average life-expectancy for that calendar year"
        )
        # Replace old 'Personal gain' with cleaner 'Years remaining'
        years_remaining = int(round(median_life - age))
        st.metric("Years remaining (median)", f"{years_remaining:d} years")

# --- Derived “expected years” metrics ---
yrs_from_habits = float(np.sum(out["yrs_added_interventions"]))  # integrates the survival delta series

# Alive‑weighted expected years from future treatments (same calc you use later for the area chart)
le_series = out.get("threshold_series")
if le_series is None:
    le_series = out.get("le_threshold_series")
alive_mask = (out["bio_age"] < le_series[None, :]).astype(float)  # (D, T)
exp_tech_by_age = (out["tech_years_by_age"] * alive_mask).mean(axis=0)
yrs_from_tech = float(np.sum(exp_tech_by_age))

with col2:
    st.metric("Net Worth (median, $MM)", f"{median_net:,.2f}")
    st.metric(
        "Years from your habits (expected)",
        f"{yrs_from_habits:.1f} years",
        help="Expected years added from your current health habits, integrated over your lifetime."
    )

with col3:
    st.metric(
        "90% range (5th-95th percentile)",
        f"{p5:.0f}-{p95:.0f} years",
        help="Middle 90% of simulated lifespans; 5% of draws fall below the left value and 5% above the right"
    )
    st.metric(
        "Years from treatments (expected, if bought)",
        f"{yrs_from_tech:.1f} years",
        help="Alive-weighted expected years from Tier 1-3 purchases, gated by your budget and arrival probabilities."
    )
    
# ================== Lifespan + Wealth controls (row 1) ==================
ctrl_l, ctrl_r = st.columns(2)

with ctrl_l:
    hist_mode = st.radio(
        "How should we simulate lifespans?",
        ["Random chance each year", "Health threshold (no dice)"],
        index=1,  # default to deterministic
        help=("Random chance each year: we roll the dice on survival each simulated year based on your risk profile. "
              "Health threshold (no dice): we assume death the moment your projected health age meets the average "
              "life expectancy for that calendar year (no randomness).")
    )

with ctrl_r:
    view = st.radio("Wealth view", ["Scatter", "Heatmap"], index=0, horizontal=True)

# Tiny spacer so charts don’t collide with radios
st.write("")

# ================== Prep data for both charts ==================
# Left chart data
if hist_mode == "Random chance each year":
    life_data = out["projected_life_mc"]
    life_title = "Simulated Ages at Death"
    nbins = int(np.ptp(life_data)) + 1 if np.ptp(life_data) >= 1 else 10
    fig_life = px.histogram(
        life_data, nbins=nbins, title=life_title,
        labels={"value": "Age at death (years)", "count": "Simulations"}
    )
else:
    life_data = out.get("projected_life_frac", out["projected_life"])
    life_title = "Projected age at health threshold"
    fig_life = px.histogram(
        life_data, title=life_title,
        labels={"value": "Age at death (years)", "count": "Simulations"}
    )
    fig_life.update_traces(xbins=dict(size=0.5))

fig_life.update_xaxes(range=[int(age), None])
fig_life.update_layout(
    title=dict(text=life_title, x=0, xanchor="left", font=dict(size=16, color="#444")),
    margin=dict(t=40, r=0, l=0, b=0),
    height=420
)

# --- Right chart data (scatter/heatmap) ---
# Use stochastic lifespans for x when dice mode is selected; otherwise deterministic threshold life.
life_for_net = out["projected_life_mc"] if hist_mode == "Random chance each year" \
               else out.get("projected_life_frac", out["projected_life"])

if hist_mode == "Random chance each year":
    # align net worth to the MC death year
    idx = np.clip((life_for_net - age).astype(int), 0, out["balance_path"].shape[1] - 1)
    nw = out["balance_path"][np.arange(idx.size), idx] / 1e6  # scale to $MM
else:
    # engine already computed threshold-mode net worth at death
    nw = out["net_worth"]

# Count DISTINCT tech start events per draw (not years active).
ty = out["tech_years_by_age"]  # shape (D, T); 0 before any tech starts, >0 once something is active
# A start is when we go from 0 last year to >0 this year.
starts = (ty > 0) & np.concatenate([np.ones((ty.shape[0], 1), dtype=bool), ty[:, :-1] == 0], axis=1)
purchases = starts.sum(axis=1)

# Bucket for clearer legend
bucket = np.where(purchases == 0, "0",
          np.where(purchases == 1, "1",
          np.where(purchases <= 3, "2-3", "4+")))

df_nw = pd.DataFrame({"Life": life_for_net, "NetWorth": nw, "Purchases": bucket})

if view == "Scatter":
    fig_nw = px.scatter(
        df_nw, x="Life", y="NetWorth", color="Purchases",
        title="Simulated Wealth Outcomes by Lifespan",
        labels={"Life": "Age at death (years)", "NetWorth": "Net worth at death ($MM)"},
        opacity=0.55
    )
else:
    fig_nw = px.density_heatmap(
        df_nw, x="Life", y="NetWorth",
        nbinsx=40, nbinsy=40, color_continuous_scale="Blues",
        title="Simulated Wealth Outcomes by Lifespan",
        labels={"Life": "Age at death (years)", "NetWorth": "Net worth at death ($MM)"}
    )
fig_nw.update_xaxes(range=[int(age), None])
title_text = "Net worth vs projected life — density" if view == "Heatmap" else "Net worth vs projected life (per draw)"
fig_nw.update_layout(title=dict(text=title_text, x=0, xanchor="left", font=dict(size=16, color="#444")),
                     margin=dict(t=40, r=0, l=0, b=0), height=420)

# ================== Charts (row 2, perfectly aligned) ==================
col_l, col_r = st.columns(2)

with col_l:
    st.plotly_chart(fig_life, use_container_width=True)

with col_r:
    st.plotly_chart(fig_nw, use_container_width=True)

# Years-added chart: Excel-match mode by default with options
mode = st.selectbox(
    "Years-added chart",
    ["Your Health Habits", "Habits & Treatments (if alive)", "Future Treatments (if alive)"],
    index=0,
    help="Excel's dashboard chart shows interventions only. ...Alive-weighted' tech counts only when a draw is alive that year"
)

yrs_int = out["yrs_added_interventions"]                           # (T,)

# Alive-weighted expected tech by age
# Use the 1-D threshold series returned by the engine
le_series = out.get("threshold_series")  # shape (T,)
# Back-compat if you ever run an older engine that used a different name
if le_series is None:
    le_series = out.get("le_threshold_series")

alive_mask = (out["bio_age"] < le_series[None, :]).astype(float)  # (D, T)
exp_tech_by_age = (out["tech_years_by_age"] * alive_mask).mean(axis=0)

if mode.startswith("Your Health Habits"):
    y = yrs_int
elif mode.startswith("Habits & Treatments (if alive)"):
    y = yrs_int + exp_tech_by_age
else:  # Tech only
    y = exp_tech_by_age

fig_yrs = px.area(x=out["chrono_age"], y=y,
                  labels={"x": "Age", "y": "Expected Years Added"},
                  title=mode)
st.plotly_chart(fig_yrs, use_container_width=True)

# Optional diagnostic
with st.expander("Average Biological Age vs Life Expectancy"):
    mean_bio = out["bio_age"].mean(axis=0)        # (T,)
    le_series = out.get("threshold_series")       # (T,)
    if le_series is None:
        le_series = out.get("le_threshold_series")

    df = pd.DataFrame({
        "Age": out["chrono_age"],
        "Biological age (mean)": mean_bio,
        "LE threshold": le_series,                # already (T,) – no .mean(axis=0)
    })
    fig_diag = px.line(
        df, x="Age", y=["Biological age (mean)", "LE threshold"],
        title="Biological Age vs Average Life Expectancy"
    )
    st.plotly_chart(fig_diag, use_container_width=True)

# Finance diagnostics
with st.expander("Personal Finances, First 20 Years"):
    df_fin = pd.DataFrame({
        "Age": out["chrono_age"],
        "Discretionary Income": out["discretionary_income_by_year"],
        "Total Healthcare Spending": out["health_spend_by_year"],
        "Excess Cash for Investments": out["contrib_by_year"],
    })
    st.dataframe(df_fin.head(20), use_container_width=True, height=480)

# ------------------ Investments over time ------------------
st.subheader("Investments over time")

# --- Controls row ---
cA, cB, cC, cD = st.columns([2, 1, 1, 1])
with cA:
    compare = st.radio(
        "Include future treatments",
        ["Both (overlay)", "With treatments", "Without treatments"],
        index=0, horizontal=True,
        help="With = includes costs when they arrive and extra years gained. "
             "Without = identical inputs but no future treatments."
    )
with cB:
    avg = st.radio("Average", ["Mean", "Median"], index=0, horizontal=True)
with cC:
    band_opt = st.selectbox("Uncertainty band", ["None", "50%", "80%", "90%"], index=3)
with cD:
    alive_weighted = st.checkbox(
        "Alive-weighted", value=True,
        help="Average only across simulations still alive at each age."
    )

# --- Data we need from the engine ---
age_grid = out["chrono_age"]                           # (T,)
path_with = out.get("balance_path", None)              # (D, T) dollars
path_without = out.get("balance_no_tech_path", None)   # (D, T) dollars

# If the engine is old, avoid a crash and explain
if path_with is None or path_without is None:
    st.warning(
        "This deployment is using an older engine that doesn't export "
        "`balance_path`/`balance_no_tech_path`. Update `engine.py` (see Fix 1) "
        "and redeploy to enable the Investments chart."
    )
    st.stop()

# Convert to $MM for plotting
path_with = path_with / 1e6
path_without = path_without / 1e6

# Safe fetch of the threshold series
thr = out.get("threshold_series")
if thr is None:
    thr = out.get("le_threshold_series")               # (T,)

# Alive masks for with/without tech
alive_with = (out["bio_age"] < thr[None, :])           # (D, T)
cum_tech = np.cumsum(out["tech_years_by_age"], axis=1) # (D, T)
bio_no_tech = out["bio_age"] + cum_tech                # remove tech benefit → older bio age
alive_without = (bio_no_tech < thr[None, :])           # (D, T)

def summarize(path, alive_mask, use_mean: bool, band: str):
    """Return center line and optional percentile bands per age."""
    D, T = path.shape
    center = np.full(T, np.nan)
    lo = np.full(T, np.nan)
    hi = np.full(T, np.nan)
    for t in range(T):
        vals = path[alive_mask[:, t], t] if alive_weighted else path[:, t]
        if vals.size == 0:
            continue
        center[t] = (np.mean(vals) if use_mean else np.median(vals))
        if band != "None":
            if band == "50%": p = (25, 75)
            elif band == "80%": p = (10, 90)
            else: p = (5, 95)
            lo[t], hi[t] = np.percentile(vals, p)
    return center, lo, hi

m_with, lo_with, hi_with = summarize(path_with, alive_with, (avg == "Mean"), band_opt)
if path_without is None:
    m_without = lo_without = hi_without = None
else:
    m_without, lo_without, hi_without = summarize(path_without, alive_without, (avg == "Mean"), band_opt)

# --- Axis end: stop at the last age where anyone is alive in the plotted selection
def last_age_any(mask: np.ndarray) -> float:
    idx = np.where(mask.any(axis=0))[0]
    return float(age_grid[idx[-1]]) if idx.size else float(age_grid[-1])

end_with = last_age_any(alive_with)
end_without = last_age_any(alive_without)
x_end = (
    end_with if compare == "With treatments"
    else end_without if compare == "Without treatments"
    else max(end_with, end_without)
)
x_start = float(age_grid[0])

# --- Chart A: portfolio value by age ---
fig_bal = go.Figure()
if compare in ("Both (overlay)", "With treatments"):
    fig_bal.add_trace(go.Scatter(x=age_grid, y=m_with, mode="lines", name="With treatments"))
if compare in ("Both (overlay)", "Without treatments") and m_without is not None:
    fig_bal.add_trace(go.Scatter(x=age_grid, y=m_without, mode="lines", name="Without treatments"))

# Show band only when not overlay (keeps it readable)
if band_opt != "None" and compare != "Both (overlay)":
    lo, hi = (lo_with, hi_with) if compare == "With treatments" else (lo_without, hi_without)
    if lo is not None and hi is not None:
        fig_bal.add_trace(go.Scatter(
            x=np.concatenate([age_grid, age_grid[::-1]]),
            y=np.concatenate([hi, lo[::-1]]),
            fill="toself", line=dict(width=0), name=f"{band_opt} band",
            hoverinfo="skip", opacity=0.18
        ))

fig_bal.update_layout(
    title="Portfolio value by age",
    xaxis_title="Age (years)",
    yaxis_title="Balance ($MM)"
)
# ⬅️ Trim the x-axis to the real endpoint
fig_bal.update_xaxes(range=[x_start, x_end])

st.plotly_chart(fig_bal, use_container_width=True)

# --- Chart B: Annual cash flows (optional) ---
show_cf = st.checkbox(
    "Show annual cash flows", value=False,
    help="Cash into the portfolio vs expected treatment spend (excludes market returns)."
)
if show_cf:
    # Dollars (not millions) for cash flows
    contrib = out["contrib_by_year"]                         # $/yr
    tech_spend = out.get("tech_costs_by_age")               # $/yr per draw
    tech_spend_mean = np.zeros_like(contrib) if tech_spend is None else tech_spend.mean(axis=0)

    df_cf = pd.DataFrame({
        "Age": age_grid,
        "Contributions (DI − health)": contrib,
        "Expected treatment spend": -tech_spend_mean,        # negative = outflow
    })
    # Trim rows beyond the endpoint
    df_cf = df_cf[df_cf["Age"] <= x_end]

    fig_cf = px.bar(
        df_cf, x="Age",
        y=["Contributions (DI − health)", "Expected treatment spend"],
        barmode="relative",
        title="Cash flows into portfolio (excluding market returns)",
        labels={"value": "$ per year"}
    )
    # Format $ nicely and trim x-axis
    fig_cf.update_layout(yaxis_tickprefix="$", yaxis_tickformat=",.0f")
    fig_cf.update_xaxes(range=[x_start, x_end])

    st.plotly_chart(fig_cf, use_container_width=True)
