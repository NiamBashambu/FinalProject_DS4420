"""
Transfer Intel - DS4420 Final Project
Streamlit app entry point.

Two pages:
  1. Project Overview  - pipeline description, key findings, caterpillar plots
  2. Deal Evaluator    - posterior-based transfer fee assessment tool
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from pathlib import Path

# PAGE CONFIG 
st.set_page_config(
    page_title="Transfer Intel | DS4420",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# PATHS
BASE   = Path(__file__).parent.parent          # repo root
ASSETS = Path(__file__).parent / "assets"      # app/assets/

# CONSTANTS
POS_LABELS: dict[str, str] = {
    "AM": "Attacking Mid",
    "CB": "Centre Back",
    "CF": "Centre Forward",
    "CM": "Central Mid",
    "DM": "Defensive Mid",
    "GK": "Goalkeeper",
    "LB": "Left Back",
    "LM": "Left Mid",
    "LW": "Left Winger",
    "RB": "Right Back",
    "RM": "Right Mid",
    "RW": "Right Winger",
    "SS": "Second Striker",
}

CARD_BG      = "#111827"
TEXT_MUTED   = "#94a3b8"
ACCENT_BLUE  = "#3b82f6"
ACCENT_CYAN  = "#06b6d4"
SUCCESS      = "#10b981"
DANGER       = "#ef4444"
WARNING      = "#f59e0b"
NEUTRAL      = "#94a3b8"

# CUSTOM CSS
CSS = f"""
<style>
# Remove Streamlit chrome 
#MainMenu {{visibility: hidden;}}
footer    {{visibility: hidden;}}
header    {{visibility: hidden;}}

# Root
html, body, [class*="css"] {{
    font-family: 'Inter', 'Helvetica Neue', 'Segoe UI', Arial, sans-serif;
}}
.stApp {{
    background-color: #0a0e1a;
}}

# Main container  
.block-container {{
    padding-top: 2rem;
    padding-bottom: 2rem;
    padding-left: 2.5rem;
    padding-right: 2.5rem;
    max-width: 1600px;
}}

# Sidebar 
[data-testid="stSidebar"] {{
    background-color: #0d1117;
    border-right: 1px solid #1e293b;
}}
[data-testid="stSidebar"] .block-container {{
    padding-top: 2rem;
    padding-left: 1.25rem;
    padding-right: 1.25rem;
}}
[data-testid="stSidebar"] .stRadio > div {{
    gap: 0.2rem;
}}
[data-testid="stSidebar"] .stRadio label {{
    border-radius: 6px;
    padding: 0.45rem 0.7rem;
    transition: background 0.15s;
    color: {TEXT_MUTED} !important;
    font-size: 0.875rem;
    cursor: pointer;
}}
[data-testid="stSidebar"] .stRadio label:hover {{
    background: #1e293b;
    color: #f1f5f9 !important;
}}

# Headings 
h1, h2, h3 {{
    color: #f1f5f9 !important;
    font-weight: 600;
    letter-spacing: -0.02em;
}}
h1 {{ font-size: 1.75rem !important; }}
h2 {{
    font-size: 1.2rem !important;
    border-bottom: 1px solid #1e293b;
    padding-bottom: 0.4rem;
    margin-bottom: 1rem !important;
}}
h3 {{ font-size: 0.95rem !important; color: {TEXT_MUTED} !important; font-weight: 500; }}

# Metrics
[data-testid="stMetric"] {{
    background: {CARD_BG};
    border: 1px solid #1e293b;
    border-radius: 8px;
    padding: 0.9rem 1.1rem !important;
}}
[data-testid="stMetric"] label {{
    color: {TEXT_MUTED} !important;
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.07em;
}}
[data-testid="stMetricValue"] {{
    color: #f1f5f9 !important;
    font-size: 1.3rem !important;
    font-weight: 600;
}}

# Divider 
hr {{ border-color: #1e293b; margin: 1.2rem 0; }}

# Verdict / info boxes 
.verdict-box {{
    border-radius: 8px;
    padding: 0.85rem 1.1rem;
    margin: 0.4rem 0;
    font-size: 0.9rem;
    line-height: 1.65;
}}
.verdict-range   {{ background: #0a1929; border-left: 3px solid {ACCENT_BLUE};  color: #bfdbfe; }}
.verdict-bargain {{ background: #052e1c; border-left: 3px solid {SUCCESS};       color: #a7f3d0; }}
.verdict-overpay {{ background: #2a0a0a; border-left: 3px solid {DANGER};        color: #fca5a5; }}
.verdict-info    {{ background: {CARD_BG}; border-left: 3px solid {ACCENT_CYAN}; color: {TEXT_MUTED}; }}

# Overview cards  
.overview-card {{
    background: {CARD_BG};
    border: 1px solid #1e293b;
    border-radius: 10px;
    padding: 1.4rem 1.5rem;
    height: 100%;
}}
.overview-card .card-tag {{
    font-size: 0.65rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: {ACCENT_CYAN};
    margin-bottom: 0.6rem;
    font-weight: 600;
}}
.overview-card p {{
    color: #94a3b8;
    font-size: 0.85rem;
    line-height: 1.65;
    margin: 0;
}}

# Stat pills 
.stat-pill {{
    display: inline-block;
    background: #1e293b;
    border-radius: 20px;
    padding: 0.15rem 0.6rem;
    font-size: 0.75rem;
    color: {TEXT_MUTED};
    margin: 0.15rem 0.1rem 0 0;
}}

# Buttons
.stButton > button {{
    background: {ACCENT_BLUE} !important;
    color: #fff !important;
    border: none !important;
    border-radius: 6px !important;
    font-weight: 600 !important;
    letter-spacing: 0.02em;
    padding: 0.5rem 1.5rem !important;
    width: 100%;
    transition: background 0.15s;
}}
.stButton > button:hover {{
    background: #2563eb !important;
}}

# Section labels  
.section-label {{
    font-size: 0.68rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: {TEXT_MUTED};
    margin-bottom: 0.2rem;
}}

# Reduce selectbox / number input border noise 
[data-testid="stSelectbox"] > div > div,
[data-testid="stNumberInput"] input {{
    background: {CARD_BG} !important;
    border-color: #1e293b !important;
}}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# DATA LOADING
@st.cache_data
def load_posterior() -> pd.DataFrame:
    path = BASE / "data/processed/posterior_samples.csv"
    if path.exists():
        return pd.read_csv(path)
    return _synthetic_posterior()


@st.cache_data
def load_model2_input() -> pd.DataFrame:
    return pd.read_csv(BASE / "data/processed/model2_input.csv")


def _synthetic_posterior() -> pd.DataFrame:
    """Fallback: synthetic posterior matching the real column structure."""
    rng  = np.random.default_rng(42)
    df   = pd.read_csv(BASE / "data/processed/model2_input.csv")
    cors = sorted(df["league_pair"].unique())
    pos  = sorted(df["player_pos"].unique())
    ab   = ["U21", "21-25", "26-29", "30+"]
    n    = 8200

    data: dict[str, np.ndarray] = {"alpha": rng.normal(0.1, 0.3, n)}
    for c in cors:
        shift = 0.5 if c.split("_to_")[1] == "England" else 0.0
        data[f"gamma_{c}"] = rng.normal(shift, 0.3, n)
    for p in pos:
        data[f"delta_{p}"] = rng.normal(0.1, 0.2, n)
    for a in ab:
        shift = -0.2 if a == "U21" else 0.0
        data[f"phi_{a}"] = rng.normal(shift, 0.3, n)
    return pd.DataFrame(data)


# HELPERS
def age_band(age: int) -> str:
    if age < 21:   return "U21"
    if age <= 25:  return "21-25"
    if age <= 29:  return "26-29"
    return "30+"


def fmt_m(val: float) -> str:
    """Format a value in millions: €45.2M or €800K."""
    if val >= 1.0:
        return f"€{val:.1f}M"
    return f"€{val * 1000:.0f}K"


def corridor_label(raw: str) -> str:
    """'France_to_England'  →  'France → England'"""
    return raw.replace("_to_", " → ").replace("_", " ")


def selling_countries(post: pd.DataFrame) -> list[str]:
    seen: set[str] = set()
    for col in post.columns:
        if col.startswith("gamma_") and "_to_" in col:
            seen.add(col[len("gamma_"):].split("_to_")[0])
    return sorted(seen)


def buying_countries(post: pd.DataFrame) -> list[str]:
    seen: set[str] = set()
    for col in post.columns:
        if col.startswith("gamma_") and "_to_" in col:
            seen.add(col[len("gamma_"):].split("_to_")[1])
    return sorted(seen)


def fee_posterior(
    post: pd.DataFrame,
    corridor: str,
    position: str,
    age: int,
    mv_m: float,
) -> tuple[np.ndarray, bool]:
    """
    Returns (fee_samples_in_M, corridor_found).

    log_premium = alpha + gamma_corridor + delta_position + phi_ageband
    fee = mv * exp(log_premium)
    """
    gamma_col = f"gamma_{corridor}"
    delta_col = f"delta_{position}"
    phi_col   = f"phi_{age_band(age)}"

    log_prem       = post["alpha"].values.copy()
    corridor_found = gamma_col in post.columns
    if corridor_found:
        log_prem += post[gamma_col].values
    log_prem += post[delta_col].values + post[phi_col].values

    return mv_m * np.exp(log_prem), corridor_found


# PAGE 1 — PROJECT OVERVIEW
def page_overview(post: pd.DataFrame, inp: pd.DataFrame) -> None:
    st.markdown("## Transfer Market Inefficiencies")
    st.markdown(
        f"<p style='color:{TEXT_MUTED}; font-size:0.875rem; margin-top:-0.6rem; margin-bottom:1.5rem;'>"
        "DS4420 &nbsp;·&nbsp; Spring 2026 &nbsp;·&nbsp; Modeling fee premiums in European football transfers"
        "</p>",
        unsafe_allow_html=True,
    )

    # Three-column info cards
    c1, c2, c3 = st.columns(3, gap="medium")

    with c1:
        st.markdown(
            """<div class="overview-card">
            <div class="card-tag">The Problem</div>
            <p>Clubs don't just pay for a player's performance. They pay for scarcity, prestige,
            and the corridor the deal happens to cross. These premiums aren't random; they show up
            consistently on certain routes, positions, and age groups. We look at 1,400+ transfers
            across the Big 5 European leagues (2000-2023) to measure how big those patterns are.</p>
            </div>""",
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            """<div class="overview-card">
            <div class="card-tag">Model 1 &nbsp;·&nbsp; MLP Valuation</div>
            <p>A multi-layer perceptron trained on per-90 performance stats gives us a
            <em>baseline valuation</em> for each player, essentially what you'd expect to
            pay going purely off on-pitch output. That predicted value feeds into Model 2.</p>
            <p style="margin-top:0.85rem;">
            <span class="stat-pill">PyTorch</span>
            <span class="stat-pill">3-layer MLP</span>
            <span class="stat-pill">Per-90 features</span>
            <span class="stat-pill">Train / test split</span>
            </p>
            </div>""",
            unsafe_allow_html=True,
        )

    with c3:
        st.markdown(
            """<div class="overview-card">
            <div class="card-tag">Model 2 &nbsp;·&nbsp; Bayesian Hierarchical</div>
            <p>A hand-coded Gibbs sampler models the <em>log-ratio premium</em>
            (log actual fee - log predicted value) with partial pooling over three
            groups: transfer corridor (i.e. Spain to England), player position, and age band.
            The full posterior distribution gets passed through to every fee estimate.</p>
            <p style="margin-top:0.85rem;">
            <span class="stat-pill">R · Manual Gibbs</span>
            <span class="stat-pill">10 k iterations</span>
            <span class="stat-pill">Partial pooling</span>
            <span class="stat-pill">8,200 post-burnin draws</span>
            </p>
            </div>""",
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Key findings metrics
    st.markdown("### Key Findings")

    alpha_med      = post["alpha"].median()
    global_premium = np.exp(alpha_med)

    gamma_cols  = [c for c in post.columns if c.startswith("gamma_")]
    gamma_means = post[gamma_cols].mean().sort_values(ascending=False)
    top_corr    = corridor_label(gamma_means.index[0].replace("gamma_", ""))
    top_corr_x  = np.exp(gamma_means.iloc[0])
    low_corr    = corridor_label(gamma_means.index[-1].replace("gamma_", ""))
    low_corr_x  = np.exp(gamma_means.iloc[-1])

    delta_cols  = [c for c in post.columns if c.startswith("delta_")]
    delta_means = post[delta_cols].mean().sort_values(ascending=False)
    top_pos     = delta_means.index[0].replace("delta_", "")
    top_pos_x   = np.exp(delta_means.iloc[0])

    phi_cols   = [c for c in post.columns if c.startswith("phi_")]
    phi_means  = post[phi_cols].mean()

    m1, m2, m3, m4 = st.columns(4, gap="small")
    m1.metric(
        "Global Median Premium",
        f"{global_premium:.2f}×",
        help="exp(α) — how much more than MLP predicted value clubs typically pay, on average",
    )
    m2.metric(
        "Highest-Premium Corridor",
        top_corr,
        f"{top_corr_x:.2f}× multiplier",
    )
    m3.metric(
        "Lowest-Premium Corridor",
        low_corr,
        f"{low_corr_x:.2f}× multiplier",
    )
    m4.metric(
        "Most Expensive Position",
        POS_LABELS.get(top_pos, top_pos),
        f"{top_pos_x:.2f}× vs. model value",
    )

    st.markdown("---")

    # Caterpillar plots 
    st.markdown("### Posterior Effect Estimates")
    st.markdown(
        f"<p style='color:{TEXT_MUTED}; font-size:0.8rem; margin-top:-0.5rem; margin-bottom:1rem;'>"
        "Each dot is the posterior mean; the line is the 90% credible interval. "
        "Right of zero means clubs tend to pay a premium on top of the global baseline; left means a discount."
        "</p>",
        unsafe_allow_html=True,
    )

    pc1, pc2 = st.columns(2, gap="medium")

    corridor_img = ASSETS / "catepillar_corridor.png"
    position_img = ASSETS / "catepillar_position.png"

    with pc1:
        st.markdown(f"**Corridor Effects (γ)** : log-scale across {len(gamma_cols)} corridors")
        st.image(str(corridor_img), use_container_width=True)

    with pc2:
        st.markdown("**Position Effects** : relative to global mean")
        st.image(str(position_img), use_container_width=True)

    # Age band summary
    st.markdown("---")
    st.markdown("### Age Band Effects")
    st.markdown(
        f"<p style='color:{TEXT_MUTED}; font-size:0.8rem; margin-top:-0.5rem; margin-bottom:1rem;'>"
        "Posterior mean and 90% CI for each age group. Shows how much the expected fee shifts "
        "depending on where a player falls in their career relative to the global baseline.</p>",
        unsafe_allow_html=True,
    )
    age_band_img = ASSETS / "age_band.png"
    if age_band_img.exists():
        st.image(str(age_band_img), use_container_width=True)
    else:
        _age_band_chart(post)

# PAGE 2 — DEAL EVALUATOR
def page_deal_evaluator(post: pd.DataFrame, inp: pd.DataFrame) -> None:
    st.markdown("## Deal Evaluator")
    st.markdown(
        f"<p style='color:{TEXT_MUTED}; font-size:0.875rem; margin-top:-0.6rem; margin-bottom:1.5rem;'>"
        "Enter a player profile and see where a quoted fee lands relative to what clubs have historically paid "
        "for similar players on the same corridor."
        "</p>",
        unsafe_allow_html=True,
    )

    sell_opts = selling_countries(post)
    buy_opts  = buying_countries(post)
    pos_opts  = sorted(c.replace("delta_", "") for c in post.columns if c.startswith("delta_"))

    # Layout
    input_col, _, output_col = st.columns([1, 0.04, 2.3], gap="small")

    # Input panel
    with input_col:
        st.markdown("### Player Profile")

        mv_m = st.number_input(
            "Market Value (€M)",
            min_value=0.1, max_value=500.0,
            value=30.0, step=0.5, format="%.1f",
            help="Model 1 (MLP) predicted market value in millions of euros",
        )

        sa_col, ba_col = st.columns(2)
        with sa_col:
            def_sell = sell_opts.index("Spain") if "Spain" in sell_opts else 0
            sell_c   = st.selectbox("Selling Country", sell_opts, index=def_sell)
        with ba_col:
            def_buy = buy_opts.index("England") if "England" in buy_opts else 0
            buy_c   = st.selectbox("Buying Country",  buy_opts,  index=def_buy)

        pos_display = [f"{p}  —  {POS_LABELS.get(p, p)}" for p in pos_opts]
        def_pos     = pos_opts.index("CF") if "CF" in pos_opts else 0
        pos_sel     = st.selectbox("Position", pos_display, index=def_pos)
        position    = pos_sel.split("  —  ")[0].strip()

        player_age = st.number_input(
            "Player Age", min_value=16, max_value=36, value=24, step=1,
        )

        st.markdown("---")
        st.markdown("<p class='section-label'>Optional: Quoted Fee</p>", unsafe_allow_html=True)
        show_quote = st.checkbox("I have a quoted fee to assess")
        quoted_fee: float | None = None
        if show_quote:
            quoted_fee = st.number_input(
                "Quoted Fee (€M)",
                min_value=0.1, max_value=1000.0,
                value=float(mv_m), step=0.5, format="%.1f",
            )

        st.markdown("<br>", unsafe_allow_html=True)
        st.button("Evaluate Transfer", type="primary")

    # Output panel
    with output_col:
        corridor       = f"{sell_c}_to_{buy_c}"
        samples, found = fee_posterior(post, corridor, position, player_age, mv_m)
        lo5, lo25, med, hi75, hi95 = np.percentile(samples, [5, 25, 50, 75, 95])
        pos_lbl  = POS_LABELS.get(position, position)
        ab_label = age_band(player_age)
        premium  = med / mv_m

        # Warning if corridor not in model
        if not found:
            st.markdown(
                f"<div class='verdict-box verdict-info'>"
                f"⚠ The <strong>{corridor_label(corridor)}</strong> corridor has limited data "
                f"in the training set, so the fee estimate is based on the global intercept, "
                f"position, and age band only.</div>",
                unsafe_allow_html=True,
            )

        # Summary sentence
        summary = (
            f"A <strong>{pos_lbl}</strong> aged <strong>{player_age}</strong> ({ab_label}) "
            f"moving from <strong>{sell_c.replace('_', ' ')}</strong> to "
            f"<strong>{buy_c.replace('_', ' ')}</strong> with a market value of "
            f"<strong>{fmt_m(mv_m)}</strong> has typically sold for "
            f"<strong>{fmt_m(lo5)}</strong> to <strong>{fmt_m(hi95)}</strong> "
            f"<span style='color:{TEXT_MUTED};'>(90% credible interval).</span>"
        )
        st.markdown(
            f"<div class='verdict-box verdict-range'>{summary}</div>",
            unsafe_allow_html=True,
        )

        # Key metrics
        km1, km2, km3, km4 = st.columns(4, gap="small")
        km1.metric("Median Expected Fee", fmt_m(med))
        km2.metric("90% CI Low",          fmt_m(lo5))
        km3.metric("90% CI High",         fmt_m(hi95))
        km4.metric("Median Premium",      f"{premium:.2f}×", f"{(premium - 1) * 100:+.1f}% over MV")

        st.markdown("<br>", unsafe_allow_html=True)

        # Histogram
        fig = _histogram(samples, mv_m, lo5, hi95, med, quoted_fee)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        # Quoted fee verdict
        if quoted_fee is not None:
            if quoted_fee < lo5:
                v_class, v_word, v_color = "verdict-bargain", "BARGAIN", SUCCESS
                v_detail = (
                    f"The quoted fee of {fmt_m(quoted_fee)} is below the 5th percentile "
                    f"of the expected distribution. It's rare to pick up this profile this cheaply."
                )
            elif quoted_fee > hi95:
                v_class, v_word, v_color = "verdict-overpay", "OVERPAY", DANGER
                v_detail = (
                    f"The quoted fee of {fmt_m(quoted_fee)} is above the 95th percentile. "
                    f"Clubs almost never pay this much for this type of player on this corridor."
                )
            else:
                pct = (samples < quoted_fee).mean() * 100
                v_class, v_word, v_color = "verdict-range", "WITHIN RANGE", ACCENT_CYAN
                v_detail = (
                    f"The quoted fee of {fmt_m(quoted_fee)} sits at the "
                    f"{pct:.0f}th percentile of the expected distribution, "
                    f"which is within the normal range for this profile."
                )
            st.markdown(
                f"<div class='verdict-box {v_class}'>"
                f"<strong style='color:{v_color};'>{v_word}</strong>"
                f" &nbsp; {v_detail}</div>",
                unsafe_allow_html=True,
            )

        st.markdown("---")

        # Sourcing comparison panel
        st.markdown(f"### Sourcing Map: Top Corridors into {buy_c}")
        st.markdown(
            f"<p style='color:{TEXT_MUTED}; font-size:0.8rem; margin-top:-0.5rem; margin-bottom:0.75rem;'>"
            f"Fee ranges for a <strong style='color:#f1f5f9'>{pos_lbl}</strong>, "
            f"age <strong style='color:#f1f5f9'>{player_age}</strong>, "
            f"{fmt_m(mv_m)} market value, across the busiest selling markets into {buy_c}. "
            f"If the selected corridor looks expensive, this shows cheaper alternatives.</p>",
            unsafe_allow_html=True,
        )
        _sourcing_panel(post, inp, buy_c, position, player_age, mv_m, corridor)


def _histogram(
    samples:    np.ndarray,
    mv_m:       float,
    lo:         float,
    hi:         float,
    median_fee: float,
    quoted_fee: float | None,
) -> go.Figure:
    fig = go.Figure()

    # Main distribution
    fig.add_trace(go.Histogram(
        x=samples,
        nbinsx=70,
        histnorm="probability density",
        marker_color=ACCENT_BLUE,
        opacity=0.55,
        name="Posterior Fee Distribution",
        hovertemplate="€%{x:.1f}M &nbsp; density: %{y:.4f}<extra></extra>",
    ))

    # 90% CI shading
    fig.add_vrect(
        x0=lo, x1=hi,
        fillcolor=ACCENT_BLUE, opacity=0.08,
        layer="below", line_width=0,
    )

    # Market value reference
    fig.add_vline(
        x=mv_m, line_dash="dash",
        line_color=NEUTRAL, line_width=1.5,
        annotation_text="Market Value",
        annotation_position="top right",
        annotation_font=dict(color=NEUTRAL, size=9),
    )

    # Median
    fig.add_vline(
        x=median_fee, line_dash="solid",
        line_color=ACCENT_CYAN, line_width=2,
        annotation_text="Median",
        annotation_position="top left",
        annotation_font=dict(color=ACCENT_CYAN, size=9),
    )

    # 90% CI bound ticks
    for x, label in [(lo, "5th pct"), (hi, "95th pct")]:
        fig.add_vline(
            x=x, line_dash="dot",
            line_color="#334155", line_width=1,
            annotation_text=label,
            annotation_position="bottom right",
            annotation_font=dict(color="#475569", size=8),
        )

    # Quoted fee
    if quoted_fee is not None:
        qcolor = SUCCESS if quoted_fee < lo else (DANGER if quoted_fee > hi else ACCENT_CYAN)
        fig.add_vline(
            x=quoted_fee, line_dash="dot",
            line_color=qcolor, line_width=2.5,
            annotation_text="Quoted Fee",
            annotation_position="top right",
            annotation_font=dict(color=qcolor, size=9),
        )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=CARD_BG,
        plot_bgcolor=CARD_BG,
        height=310,
        margin=dict(l=20, r=20, t=30, b=40),
        xaxis=dict(
            title="Transfer Fee (€M)",
            gridcolor="#1e293b", zerolinecolor="#1e293b",
            tickprefix="€", ticksuffix="M",
            tickfont=dict(size=10, color=TEXT_MUTED),
        ),
        yaxis=dict(
            title="Posterior Density",
            gridcolor="#1e293b",
            tickfont=dict(size=10, color=TEXT_MUTED),
        ),
        showlegend=False,
        bargap=0.02,
        hoverlabel=dict(bgcolor="#1e293b", font_size=11),
    )
    return fig


def _sourcing_panel(
    post:        pd.DataFrame,
    inp:         pd.DataFrame,
    buy_c:       str,
    position:    str,
    player_age:  int,
    mv_m:        float,
    sel_corridor: str,
) -> None:
    # All corridors selling into buy_c that exist in posterior
    valid = {
        col.replace("gamma_", ""): col
        for col in post.columns
        if col.startswith("gamma_") and col.endswith(f"_to_{buy_c}")
    }
    if not valid:
        st.markdown(
            f"<div class='verdict-box verdict-info'>"
            f"No corridor data available for transfers into {buy_c}.</div>",
            unsafe_allow_html=True,
        )
        return

    counts = (
        inp[inp["league_pair"].isin(valid.keys())]
        .groupby("league_pair")
        .size()
        .sort_values(ascending=False)
    )

    top = counts.head(7).index.tolist()
    # Always include selected corridor if it's valid
    if sel_corridor in valid and sel_corridor not in top:
        top = [sel_corridor] + top[:6]

    if not top:
        st.markdown(
            f"<div class='verdict-box verdict-info'>"
            f"No transfer data available for corridors into {buy_c}.</div>",
            unsafe_allow_html=True,
        )
        return

    # Compute stats per corridor
    rows: list[dict] = []
    for corr in top:
        s, _ = fee_posterior(post, corr, position, player_age, mv_m)
        l5, l25, med, h75, h95 = np.percentile(s, [5, 25, 50, 75, 95])
        rows.append({
            "corr":  corr,
            "label": corridor_label(corr),
            "l5": l5, "l25": l25, "med": med, "h75": h75, "h95": h95,
            "n":  int(counts.get(corr, 0)),
            "selected": corr == sel_corridor,
        })

    rows.sort(key=lambda r: r["med"])   # cheapest first

    fig = go.Figure()

    for r in rows:
        lbl = r["label"]
        is_sel = r["selected"]
        c_main = WARNING if is_sel else ACCENT_BLUE

        # 90 % range — thin outer line
        fig.add_trace(go.Scatter(
            x=[r["l5"], r["h95"]], y=[lbl, lbl],
            mode="lines", line=dict(color=c_main, width=2),
            opacity=0.45, showlegend=False, hoverinfo="skip",
        ))
        # 50 % range — thick inner bar
        fig.add_trace(go.Scatter(
            x=[r["l25"], r["h75"]], y=[lbl, lbl],
            mode="lines", line=dict(color=c_main, width=7),
            opacity=0.7, showlegend=False, hoverinfo="skip",
        ))
        # Median dot
        fig.add_trace(go.Scatter(
            x=[r["med"]], y=[lbl],
            mode="markers",
            marker=dict(
                color="white",
                size=10 if is_sel else 8,
                symbol="diamond" if is_sel else "circle",
                line=dict(color=c_main, width=1.5),
            ),
            showlegend=False,
            hovertemplate=(
                f"<b>{lbl}</b><br>"
                f"Median: {fmt_m(r['med'])}<br>"
                f"50 % CI: {fmt_m(r['l25'])} – {fmt_m(r['h75'])}<br>"
                f"90 % CI: {fmt_m(r['l5'])} – {fmt_m(r['h95'])}<br>"
                f"Transfers in dataset: {r['n']}"
                "<extra></extra>"
            ),
        ))

    # Market value reference
    fig.add_vline(
        x=mv_m, line_dash="dash",
        line_color=NEUTRAL, line_width=1,
        annotation_text="Market Value",
        annotation_position="top right",
        annotation_font=dict(color=NEUTRAL, size=9),
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=CARD_BG,
        plot_bgcolor=CARD_BG,
        height=max(260, len(rows) * 52 + 70),
        margin=dict(l=20, r=20, t=20, b=40),
        xaxis=dict(
            title="Expected Transfer Fee (€M)",
            gridcolor="#1e293b",
            tickprefix="€", ticksuffix="M",
            tickfont=dict(size=10, color=TEXT_MUTED),
        ),
        yaxis=dict(
            tickfont=dict(size=10, color="#e2e8f0"),
            gridcolor="#1e293b",
            categoryorder="array",
            categoryarray=[r["label"] for r in rows],
        ),
        showlegend=False,
        hoverlabel=dict(bgcolor="#1e293b", font_size=11),
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    st.markdown(
        f"<p style='color:{TEXT_MUTED}; font-size:0.75rem; margin-top:-0.5rem;'>"
        f"◆ Diamond = selected corridor ({corridor_label(sel_corridor)}).  "
        f"Thick bar = 50 % CI · Thin bar = 90 % CI · Sorted lowest → highest median."
        f"</p>",
        unsafe_allow_html=True,
    )


# SIDEBAR + ROUTING
def main() -> None:
    post = load_posterior()
    inp  = load_model2_input()

    n_corridors = sum(1 for c in post.columns if c.startswith("gamma_"))

    with st.sidebar:
        st.markdown(
            "<div style='margin-bottom:1.75rem;'>"
            "<div style='font-size:1.15rem; font-weight:700; color:#f1f5f9;"
            " letter-spacing:-0.02em;'>Transfer Intel</div>"
            f"<div style='font-size:0.65rem; color:#475569; text-transform:uppercase;"
            f" letter-spacing:0.1em;'>DS4420 · Spring 2026</div>"
            "</div>",
            unsafe_allow_html=True,
        )

        page = st.radio(
            "page",
            options=["Project Overview", "Deal Evaluator"],
            label_visibility="collapsed",
        )

        st.markdown("---")
        st.markdown(
            f"<p style='font-size:0.7rem; color:#334155; line-height:1.8;'>"
            f"Sampler &nbsp;Gibbs (R, manual)<br>"
            f"Posterior draws &nbsp;{len(post):,}<br>"
            f"Corridors &nbsp;{n_corridors}<br>"
            f"Transfers &nbsp;{len(inp):,}<br>"
            f"Age bands &nbsp;U21 / 21–25 / 26–29 / 30+"
            f"</p>",
            unsafe_allow_html=True,
        )

    if page == "Project Overview":
        page_overview(post, inp)
    else:
        page_deal_evaluator(post, inp)


main()
