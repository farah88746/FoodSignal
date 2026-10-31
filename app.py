import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from foodshield import data as fsdata  # noqa: E402
from foodshield import screening  # noqa: E402
from foodshield.countries import country_label, country_name  # noqa: E402


# ============================================================
# CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="FoodSignal",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

PROCESSED_DIR = BASE_DIR / "data" / "processed"
DATA_FILE = PROCESSED_DIR / "food_insecurity_percentages.csv"
SUMMARY_FILE = PROCESSED_DIR / "cleaning_summary.csv"
REVIEW_FILE = PROCESSED_DIR / "records_for_review.csv"
INVALID_FILE = PROCESSED_DIR / "invalid_records.csv"

# Colour-blind-safe pair (blue / orange) instead of red / green.
DIRECTION_COLORS = {
    "Increase": "#d95f02",
    "Decrease": "#1f78b4",
    "No change": "#8c8c8c",
}
CONFLICT_COLOR = "#b2182b"


def _version_tuple(text):
    parts = []
    for piece in str(text).split(".")[:3]:
        digits = "".join(ch for ch in piece if ch.isdigit())
        parts.append(int(digits) if digits else 0)
    return tuple(parts)


# Streamlit 1.50+ uses width="stretch"; older versions use use_container_width.
# The version is checked once so neither a deprecation warning nor an
# unsupported-argument error appears.
if _version_tuple(getattr(st, "__version__", "0")) >= (1, 50, 0):
    STRETCH = {"width": "stretch"}
else:
    STRETCH = {"use_container_width": True}


# ============================================================
# VISUAL DESIGN SYSTEM
# ============================================================

st.markdown(
    """
    <style>
    /* FoodShield UI system: explicitly force light surfaces to prevent black/dark blocks */
    :root {
      color-scheme: light !important;
      --fs-bg: #f4f7fb;
      --fs-surface: #ffffff;
      --fs-surface-soft: #f7faff;
      --fs-ink: #14243b;
      --fs-muted: #5c6b80;
      --fs-blue: #1769aa;
      --fs-blue-dark: #0d4f87;
      --fs-border: #dce5ef;
      --fs-green: #247a58;
      --fs-amber: #9a5a08;
      --fs-shadow: 0 5px 18px rgba(18, 42, 72, .055);
    }
    html, body, .stApp, [data-testid="stAppViewContainer"],
    [data-testid="stMain"], [data-testid="stMainBlockContainer"],
    [data-testid="stSidebar"], [data-testid="stHeader"] {
      color-scheme: light !important;
    }
    .stApp, [data-testid="stAppViewContainer"] {
      background: var(--fs-bg) !important;
      color: var(--fs-ink) !important;
    }
    [data-testid="stMainBlockContainer"] { padding-top: 1.6rem; padding-bottom: 3rem; }
    [data-testid="stHeader"] { background: rgba(244,247,251,.96) !important; }
    [data-testid="stSidebar"] {
      background: linear-gradient(180deg, #edf4fb 0%, #f8fbff 100%) !important;
      border-right: 1px solid var(--fs-border) !important;
    }
    [data-testid="stSidebar"] > div:first-child { background: transparent !important; }
    [data-testid="stSidebar"] * { color: var(--fs-ink); }
    h1, h2, h3, h4, h5, h6, p, li, label, small, span, div, summary {
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    h1, h2, h3, h4, h5, h6 { color: var(--fs-ink) !important; letter-spacing: -.025em; }
    h1 { font-weight: 800 !important; } h2 { font-weight: 760 !important; } h3 { font-weight: 700 !important; }
    p, li, [data-testid="stCaptionContainer"] { color: var(--fs-muted); line-height: 1.58; }
    [data-testid="stMetric"] {
      background: var(--fs-surface) !important; color: var(--fs-ink) !important;
      border: 1px solid var(--fs-border) !important; border-radius: 15px !important;
      padding: 15px 17px !important; box-shadow: var(--fs-shadow);
    }
    [data-testid="stMetricLabel"], [data-testid="stMetricLabel"] p { color: var(--fs-muted) !important; }
    [data-testid="stMetricValue"], [data-testid="stMetricValue"] div { color: var(--fs-ink) !important; font-weight: 780 !important; }
    [data-testid="stVerticalBlockBorderWrapper"] {
      border-color: var(--fs-border) !important; border-radius: 16px !important; background: var(--fs-surface) !important;
    }
    [data-testid="stDataFrame"], [data-testid="stTable"] {
      border: 1px solid var(--fs-border) !important; border-radius: 12px !important;
      overflow: hidden; background: #fff !important; color: var(--fs-ink) !important;
    }
    /* Streamlit widgets can inherit dark-mode colors; explicitly style each control */
    [data-testid="stWidgetLabel"], [data-testid="stWidgetLabel"] p,
    [data-testid="stMarkdownContainer"], [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] li { color: var(--fs-ink); }
    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div,
    div[data-baseweb="textarea"] > div {
      border-radius: 10px !important; border-color: #cbd8e6 !important;
      background: #fff !important; color: var(--fs-ink) !important;
    }
    div[data-baseweb="select"] *, div[data-baseweb="input"] *, div[data-baseweb="textarea"] *,
    input, textarea { color: var(--fs-ink) !important; -webkit-text-fill-color: var(--fs-ink) !important; }
    ul[data-testid="stSelectboxVirtualDropdown"], [role="listbox"], [role="option"],
    [data-baseweb="popover"] > div, [data-baseweb="menu"] {
      background: #fff !important; color: var(--fs-ink) !important;
    }
    [role="option"]:hover { background: #eaf3fc !important; }
    .stButton > button, .stDownloadButton > button {
      border-radius: 10px !important; font-weight: 680 !important;
      border: 1px solid #c6d7e8 !important; background: #fff !important; color: var(--fs-blue-dark) !important;
      transition: all .15s ease-in-out;
    }
    .stButton > button[kind="primary"] {
      background: var(--fs-blue) !important; border-color: var(--fs-blue) !important; color: #fff !important;
    }
    .stButton > button:hover, .stDownloadButton > button:hover {
      border-color: var(--fs-blue) !important; background: #edf6ff !important; color: var(--fs-blue-dark) !important;
    }
    .stButton > button[kind="primary"]:hover { background: var(--fs-blue-dark) !important; color: #fff !important; }
    [data-testid="stAlert"] { border-radius: 12px !important; }
    [data-testid="stExpander"] { border: 1px solid var(--fs-border) !important; border-radius: 12px !important; background: #fff !important; }
    [data-testid="stExpander"] details, [data-testid="stExpander"] summary { background: #fff !important; color: var(--fs-ink) !important; }
    [data-testid="stTabs"] [role="tablist"] { gap: .35rem; }
    [data-testid="stTabs"] button[role="tab"] { color: var(--fs-muted) !important; }
    [data-testid="stTabs"] button[aria-selected="true"] { color: var(--fs-blue-dark) !important; border-bottom-color: var(--fs-blue) !important; }
    hr { border-color: var(--fs-border) !important; }
    .fs-eyebrow { color: var(--fs-blue) !important; text-transform: uppercase; letter-spacing: .13em; font-size: .73rem; font-weight: 800; margin-bottom: .5rem; }
    .fs-hero {
      padding: 1.8rem 1.9rem; border: 1px solid #d6e4f2; border-radius: 20px;
      background: radial-gradient(circle at 90% 10%, rgba(198,226,250,.55), transparent 35%), linear-gradient(120deg,#edf6ff 0%,#ffffff 62%,#f0f8f3 100%);
      box-shadow: var(--fs-shadow); margin: .35rem 0 1.35rem;
    }
    .fs-hero-title { color: #102c4e !important; font-size: clamp(1.8rem, 3vw, 2.55rem); font-weight: 850; line-height: 1.12; letter-spacing: -.045em; margin: 0 0 .65rem; }
    .fs-hero-copy { color: #4a5f78 !important; font-size: 1.02rem; line-height: 1.65; max-width: 880px; }
    .fs-step { height: 100%; min-height: 138px; padding: 1.1rem; border: 1px solid var(--fs-border); border-radius: 16px; background: #fff; box-shadow: var(--fs-shadow); }
    .fs-step-number { color: var(--fs-blue) !important; font-size: .72rem; font-weight: 850; letter-spacing: .12em; }
    .fs-step-title { color: var(--fs-ink) !important; font-weight: 760; margin: .45rem 0; }
    .fs-step-copy { color: var(--fs-muted) !important; font-size: .9rem; line-height: 1.55; }
    @media(max-width: 800px) {
      [data-testid="stMainBlockContainer"] { padding-left: 1rem; padding-right: 1rem; }
      .fs-hero { padding: 1.15rem; }
      [data-testid="stMetric"] { padding: 11px !important; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def show_chart(figure):
    # Keep Plotly surfaces and labels readable even when the browser/app prefers dark mode.
    figure.update_layout(
        template="plotly_white",
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font=dict(family="Inter, Segoe UI, sans-serif", color="#14243b", size=12),
        title_font=dict(color="#14243b", size=17),
        legend=dict(font=dict(color="#40536b")),
        margin=dict(l=20, r=20, t=55, b=25),
    )
    figure.update_xaxes(color="#52647b", gridcolor="#e7edf4", zerolinecolor="#dce5ef")
    figure.update_yaxes(color="#52647b", gridcolor="#e7edf4", zerolinecolor="#dce5ef")
    st.plotly_chart(figure, **STRETCH)


def show_table(frame, **kwargs):
    st.dataframe(frame, hide_index=True, **STRETCH, **kwargs)


def set_category_order(figure, labels_top_to_bottom):
    """Keep horizontal bars in the same order as the table (first row on top)."""
    figure.update_yaxes(
        categoryorder="array",
        categoryarray=list(reversed(list(labels_top_to_bottom))),
        automargin=True,
    )


# ============================================================
# DATA LOADING
# ============================================================

@st.cache_data(show_spinner=False)
def load_cached(file_path, modified_time):
    loaded = fsdata.load_observations(file_path)
    return loaded.data, loaded.rejected, list(loaded.missing_metadata)


if not DATA_FILE.exists():
    st.error(
        "The cleaned dataset was not found.\n\n"
        f"Expected location: `{DATA_FILE}`"
    )
    stray = BASE_DIR / DATA_FILE.name
    if stray.exists():
        st.info(
            f"A file with this name exists at `{stray}`. "
            "Move it into `data/processed/` (or run `clean_data.py`)."
        )
    st.stop()

try:
    observations, rejected, missing_metadata = load_cached(
        str(DATA_FILE), DATA_FILE.stat().st_mtime
    )
except Exception as error:  # noqa: BLE001 - show any load problem to the user
    st.error("The dataset could not be loaded.")
    st.code(str(error))
    st.stop()

if observations.empty:
    st.error("The dataset contains no valid observations.")
    if len(rejected):
        st.warning(f"{len(rejected):,} rows were rejected during validation.")
    st.stop()

coverage = fsdata.coverage_summary(observations)
dataset_years = sorted(int(y) for y in observations["year"].unique())
earliest_year, latest_year = dataset_years[0], dataset_years[-1]
country_codes = sorted(
    observations["country_code"].unique(), key=lambda code: country_name(code).lower()
)
country_total = len(country_codes)
example = fsdata.window_example(observations)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="fs-hero">
      <div class="fs-eyebrow">Humanitarian evidence workspace</div>
      <div class="fs-hero-title">FoodSignal</div>
      <div class="fs-hero-copy">Find countries that may need further review, understand the evidence behind each signal, and prepare a clear review brief — without treating historical estimates as forecasts or official assessments.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("### Start a country review")
step1, step2, step3 = st.columns(3, gap="medium")
with step1:
    st.markdown('<div class="fs-step"><div class="fs-step-number">STEP 01</div><div class="fs-step-title">Find candidates</div><div class="fs-step-copy">Filter the priority-screening list by category or search for a country.</div></div>', unsafe_allow_html=True)
with step2:
    st.markdown('<div class="fs-step"><div class="fs-step-number">STEP 02</div><div class="fs-step-title">Check the evidence</div><div class="fs-step-copy">Review the estimate year, historical direction, coverage and data limitations.</div></div>', unsafe_allow_html=True)
with step3:
    st.markdown('<div class="fs-step"><div class="fs-step-number">STEP 03</div><div class="fs-step-title">Prepare a brief</div><div class="fs-step-copy">Export the evidence, rationale and verification steps for follow-up.</div></div>', unsafe_allow_html=True)

st.caption(
    "A screening result means that further review may be useful. It is not an "
    "official needs assessment and must not be used alone to allocate aid."
)

if example:
    example_year, example_window = example
    window_sentence = f"For example, **{example_year} represents the {example_window} window**."
else:
    window_sentence = (
        "The reference window could not be read from this file, so none is shown."
    )

st.info(
    "**How to read reference years.** Each estimate is a three-year average. "
    f"{window_sentence} Neighbouring years share two of their three underlying "
    "years, so consecutive estimates are not independent observations."
)

st.warning(
    "Prototype scope: this application describes historical FAO estimates. "
    "It does not produce validated forecasts, risk scores or official assessments."
)

if missing_metadata:
    st.warning(
        "These metadata columns are missing from the dataset, so the related "
        "information shows as 'Not provided': " + ", ".join(missing_metadata)
    )

st.divider()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.markdown("## 🌾 FoodSignal")
st.sidebar.caption("Humanitarian evidence workspace")
st.sidebar.divider()
st.sidebar.markdown("**Quick context**")
st.sidebar.caption(f"{country_total:,} countries / areas · {earliest_year}–{latest_year}")
st.sidebar.caption("Country names are shown for readability; ISO codes are always kept.")

default_country = "DZA" if "DZA" in country_codes else country_codes[0]
selected_country = st.sidebar.selectbox(
    "Select a country",
    options=country_codes,
    index=country_codes.index(default_country),
    format_func=country_label,
)

selected_year = st.sidebar.selectbox(
    "Comparison year (reference year)",
    options=dataset_years,
    index=len(dataset_years) - 1,
    format_func=lambda y: f"{y}  (window {fsdata.window_label_for_year(observations, y)})",
)

st.sidebar.caption(
    "Rankings and comparisons only include countries that have an unambiguous "
    "estimate for the chosen year."
)


# ============================================================
# GLOBAL OVERVIEW AND COVERAGE
# ============================================================

st.header("Global overview")

with_latest = int(coverage["latest_is_current"].sum())
not_current = country_total - with_latest

metric1, metric2, metric3, metric4 = st.columns(4)
metric1.metric("Records", f"{len(observations):,}")
metric2.metric("Countries / areas", f"{country_total:,}")
metric3.metric("Reference years", f"{earliest_year}\u2013{latest_year}")
metric4.metric(
    f"Countries with a {latest_year} estimate",
    f"{with_latest:,} of {country_total:,}",
)

per_year = (
    observations.groupby("year")
    .agg(countries=("country_code", "nunique"), window=("window_label", "first"))
    .reset_index()
)
coverage_chart = px.bar(
    per_year,
    x="year",
    y="countries",
    text="countries",
    hover_data={"window": True, "year": True, "countries": True},
    title="Countries with an estimate, by reference year",
    labels={
        "year": "Reference year",
        "countries": "Countries / areas",
        "window": "Reference window",
    },
)
coverage_chart.update_layout(
    height=340,
    xaxis=dict(dtick=1),
    margin=dict(l=20, r=20, t=60, b=20),
)
show_chart(coverage_chart)

st.caption(
    "Coverage differs by year. Different years therefore describe different sets "
    "of countries, which is why comparisons below are restricted to countries "
    "observed in both years being compared."
)

if not_current:
    st.warning(
        f"{not_current} countries/areas have no estimate for {latest_year}. Their "
        "latest observation is earlier and must not be read as a "
        f"{latest_year} value."
    )

with st.expander("Data coverage by country (first, latest and missing years)"):
    coverage_table = fsdata.coverage_display_table(coverage)
    show_table(coverage_table)
    st.download_button(
        label="Download coverage table as CSV",
        data=coverage_table.to_csv(index=False).encode("utf-8"),
        file_name="foodsignal_coverage_by_country.csv",
        mime="text/csv",
    )

st.divider()


# ============================================================
# COUNTRY TREND
# ============================================================

st.header("Country trend")

country_data = observations[observations["country_code"] == selected_country].sort_values(
    ["year", "source"]
)
country_label_text = country_label(selected_country)
cov_row = coverage[coverage["country_code"] == selected_country].iloc[0]

unambiguous = country_data[~country_data["conflict"]].drop_duplicates("year")
conflict_rows = country_data[country_data["conflict"]]

trend1, trend2, trend3, trend4 = st.columns(4)
trend1.metric("Country", country_name(selected_country), help=f"ISO code: {selected_country}")
trend2.metric(
    "Years observed",
    f"{int(cov_row['n_years_observed'])} of {int(cov_row['years_in_dataset'])}",
)
trend3.metric(
    "Latest observation",
    f"{int(cov_row['latest_year'])}",
    help=f"Reference window: {cov_row['latest_window']}",
)
if cov_row["latest_year_conflict"]:
    trend4.metric("Latest estimate", "Conflict")
else:
    trend4.metric("Latest estimate", f"{cov_row['latest_value']:.1f}%")

if not bool(cov_row["latest_is_current"]):
    st.warning(
        f"The latest observation for {country_name(selected_country)} is "
        f"{int(cov_row['latest_year'])} (window {cov_row['latest_window']}), while the dataset "
        f"runs to {latest_year}. It is not comparable with {latest_year} values of other countries."
    )

if int(cov_row["gap_years"]) > 0:
    st.info(
        f"{int(cov_row['gap_years'])} reference year(s) between this country's first and "
        "latest observations have no estimate. The line connects the available "
        "points; it does not imply values for the gaps."
    )

if not conflict_rows.empty:
    st.error(
        f"{conflict_rows['year'].nunique()} reference year(s) have conflicting estimates for "
        f"{country_name(selected_country)}. They are shown as red crosses and are not used in "
        "the line, the change figures or the rankings. They need source-level review."
    )

change = fsdata.country_change(country_data)
if change is not None:
    st.markdown("#### Change between this country's first and latest unambiguous estimates")
    change1, change2, change3 = st.columns(3)
    change1.metric(
        f"First ({int(change['start_year'])})",
        f"{change['value_start']:.1f}%",
        help=f"Reference window: {change['label_start']}",
    )
    change2.metric(
        f"Latest ({int(change['end_year'])})",
        f"{change['value_end']:.1f}%",
        help=f"Reference window: {change['label_end']}",
    )
    change3.metric(
        "Difference",
        f"{change['change_pp']:+.1f} percentage points",
        help="Latest minus first. Descriptive only; not a forecast.",
    )
    notes = []
    if change["windows_overlap_years"] and change["windows_overlap_years"] > 0:
        notes.append(
            f"The two windows ({change['label_start']} and {change['label_end']}) overlap by "
            f"{int(change['windows_overlap_years'])} year(s), so part of the difference reflects "
            "shared underlying data."
        )
    if change["source_changed"]:
        notes.append(
            f"The source changed between these estimates ({change['source_start']} \u2192 "
            f"{change['source_end']}); differences may partly reflect the change of source."
        )
    for note in notes:
        st.caption(note)
else:
    st.caption("Fewer than two unambiguous reference years are available, so no change is shown.")

trend_chart = px.line(
    unambiguous,
    x="year",
    y="food_insecurity_percent",
    markers=True,
    title=f"Historical indicator \u2014 {country_label_text}",
    labels={
        "year": "Reference year (midpoint of the three-year window)",
        "food_insecurity_percent": "Estimated prevalence (%)",
        "window_label": "Reference window",
        "source": "Source",
        "observation_status": "Status code",
        "country_code": "ISO code",
    },
    hover_data={
        "window_label": True,
        "source": True,
        "observation_status": True,
        "country_code": True,
        "food_insecurity_percent": ":.1f",
    },
)
if not conflict_rows.empty:
    trend_chart.add_trace(
        go.Scatter(
            x=conflict_rows["year"],
            y=conflict_rows["food_insecurity_percent"],
            mode="markers",
            name="Unresolved conflict",
            marker=dict(symbol="x", size=11, color=CONFLICT_COLOR),
            customdata=conflict_rows[["window_label", "source"]].to_numpy(),
            hovertemplate=(
                "%{x}: %{y:.1f}%<br>Window %{customdata[0]}<br>%{customdata[1]}"
                "<extra>Unresolved conflict</extra>"
            ),
        )
    )
trend_chart.update_layout(
    height=430,
    xaxis=dict(dtick=1),
    yaxis=dict(rangemode="tozero", ticksuffix="%", title="Estimated prevalence (%)"),
    hovermode="x unified",
    margin=dict(l=20, r=20, t=65, b=20),
)
show_chart(trend_chart)

st.caption(
    "Each point is a three-year average. The reference year is the window midpoint; "
    "the exact window is shown on hover and in the table."
)

with st.expander("View records for this country"):
    show_table(fsdata.to_display_table(country_data))

st.divider()



# ============================================================
# RISK & EVIDENCE EXPLORER (DESCRIPTIVE, NOT A FORECAST)
# ============================================================

st.header("Food insecurity: evidence & historical signal")
st.write(
    "This explorer separates the observed burden, the historical change, and "
    "the strength of available evidence. It does not calculate a validated risk "
    "score or predict future food insecurity."
)

risk_country_data = observations[
    observations["country_code"] == selected_country
].copy()
risk_country_data = risk_country_data.sort_values(["year", "source"])

# One estimate per reference year is usable only when that country-year is not
# flagged as a conflict. Metadata duplicates are collapsed for display only;
# conflicting country-years remain excluded from the signal.
risk_clean = risk_country_data[~risk_country_data["conflict"]].drop_duplicates("year")
risk_conflicts = risk_country_data[risk_country_data["conflict"]]
risk_latest = None if risk_clean.empty else risk_clean.sort_values("year").iloc[-1]

def _parse_window_bounds(value):
    """Return (start, end) for a YYYY-YYYY reference window, or None."""
    import re as _re
    match = _re.search(r"(\d{4})\s*[-–]\s*(\d{4})", str(value))
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))

def _window_overlap_years(left, right):
    left_bounds = _parse_window_bounds(left)
    right_bounds = _parse_window_bounds(right)
    if not left_bounds or not right_bounds:
        return None
    return max(0, min(left_bounds[1], right_bounds[1]) -
               max(left_bounds[0], right_bounds[0]) + 1)

if risk_latest is None:
    st.warning("No unambiguous estimate is available for this country. Review conflicting records before interpreting its status.")
else:
    latest_year_value = int(risk_latest["year"])
    latest_window_value = str(risk_latest.get("window_label", risk_latest.get("time_detail", "Not provided")))
    latest_estimate_value = float(risk_latest["food_insecurity_percent"])
    n_observed_years = int(risk_clean["year"].nunique())
    years_since_latest = max(0, int(latest_year) - latest_year_value)
    n_sources = int(risk_clean["source"].nunique()) if "source" in risk_clean else 0

    st.subheader("1. Observed burden")
    burden_cols = st.columns(3)
    burden_cols[0].metric("Latest unambiguous estimate", f"{latest_estimate_value:.1f}%")
    burden_cols[1].metric("Reference year", str(latest_year_value), help=f"Reference window: {latest_window_value}")
    burden_cols[2].metric("Years behind dataset end", str(years_since_latest))
    st.caption(
        f"This is an observed FAO indicator estimate for {latest_window_value}, "
        "not a complete assessment of food insecurity and not a prediction."
    )

    st.subheader("2. Historical signal")
    # Prefer the latest estimate and the latest earlier estimate with a
    # non-overlapping three-year reference window.
    comparison = None
    for _, candidate in risk_clean.sort_values("year", ascending=False).iloc[1:].iterrows():
        candidate_window = candidate.get("window_label", candidate.get("time_detail", ""))
        overlap = _window_overlap_years(candidate_window, latest_window_value)
        if overlap == 0:
            comparison = candidate
            break

    if comparison is None:
        st.info(
            "Insufficient non-overlapping reference windows to describe a "
            "like-for-like historical change for this country."
        )
    else:
        previous_value = float(comparison["food_insecurity_percent"])
        previous_year_value = int(comparison["year"])
        delta_pp = latest_estimate_value - previous_value
        previous_window_value = str(comparison.get("window_label", comparison.get("time_detail", "Not provided")))
        direction = "Increased" if delta_pp > 0 else ("Decreased" if delta_pp < 0 else "No numerical change")
        trend_cols = st.columns(3)
        trend_cols[0].metric(f"Earlier estimate ({previous_year_value})", f"{previous_value:.1f}%", help=f"Window: {previous_window_value}")
        trend_cols[1].metric(f"Latest estimate ({latest_year_value})", f"{latest_estimate_value:.1f}%", help=f"Window: {latest_window_value}")
        trend_cols[2].metric("Difference", f"{delta_pp:+.1f} percentage points", help="Latest minus earlier estimate; descriptive only.")
        st.write(f"**Historical signal:** {direction}.")
        st.caption(
            "The selected windows do not overlap, reducing direct overlap between their "
            "underlying three-year periods. This comparison still does not establish "
            "causation or predict future outcomes."
        )

    st.subheader("3. Evidence available")
    evidence_cols = st.columns(4)
    evidence_cols[0].metric("Unambiguous reference years", f"{n_observed_years}")
    evidence_cols[1].metric("Distinct sources", f"{n_sources}")
    evidence_cols[2].metric("Unresolved conflict years", f"{risk_conflicts['year'].nunique()}")
    evidence_cols[3].metric("Latest observation year", str(latest_year_value))

    evidence_notes = []
    if years_since_latest > 0:
        evidence_notes.append(
            f"The latest estimate is {years_since_latest} reference year(s) older than the dataset's latest reference year ({latest_year})."
        )
    if n_observed_years < 3:
        evidence_notes.append("There are fewer than three unambiguous reference years, limiting the historical context.")
    if n_sources > 1:
        evidence_notes.append("More than one source appears in this country's unambiguous observations; source changes may affect comparability.")
    if not risk_conflicts.empty:
        evidence_notes.append("At least one country-year has conflicting records; those years are excluded from this descriptive signal.")
    if not evidence_notes:
        evidence_notes.append("The selected country has multiple unambiguous observations and its latest estimate matches the dataset's latest reference year; this does not independently validate the estimate.")
    for note in evidence_notes:
        st.markdown(f"- {note}")

    with st.expander("Records used for this interpretation"):
        show_table(fsdata.to_display_table(risk_country_data))

    st.caption(
        "Evidence facts are descriptive checks, not a calibrated confidence probability. "
        "No combined risk score is produced."
    )


# ============================================================
# TRANSPARENT PRIORITY SCREEN (RELATIVE, NOT A FORECAST)
# ============================================================

st.header("Priority screening")
st.write(
    "A transparent screening view for deciding where further review may be useful. "
    "It keeps observed burden, historical direction, and evidence availability separate; "
    "it is not a forecast, a validated risk score, or a substitute for local assessment."
)

# The screening rules live in foodshield/screening.py so they can be tested.
priority_df, screen_info = screening.build_priority_table(
    observations, int(latest_year), stale_standing=True, country_order=country_codes
)

if not priority_df.empty:
    cutoff = screen_info["current_cutoff"]
    stale_high = int((priority_df["Priority screen"] == screening.CAT_STALE_HIGH).sum())

    p1, p2, p3, p4 = st.columns(4)
    p1.metric("Countries / areas", f"{len(priority_df):,}")
    p2.metric("Comparable latest-year estimates", f"{screen_info['n_comparable']:,}")
    p3.metric("Higher relative-burden cutoff", f"{cutoff:.1f}%" if cutoff is not None else "Not calculated")
    p4.metric(
        "Older estimates, higher last-known burden",
        f"{stale_high:,}",
        help="Countries with no estimate for the latest reference year whose last estimate was in "
        "the higher quarter of the countries observed in that same earlier year.",
    )

    st.caption(
        f"Peer comparison uses only unambiguous estimates for reference year {latest_year}. "
        "Where at least four countries are available, 'higher relative burden' means the top quarter "
        "of those observed estimates (75th percentile); this is a relative comparison, not a clinical "
        "or policy threshold. Historical direction uses the latest earlier non-overlapping reference window."
    )
    st.warning(
        "Use this table to prioritize further investigation, not to allocate aid automatically. "
        "Older estimates are never ranked against latest-year estimates; they are compared only with "
        "countries observed in their own reference year."
    )
    if stale_high:
        st.info(
            f"{stale_high} countries/areas have no estimate for {latest_year}, but their last estimate "
            "was in the higher quarter of their own reference year. They are listed as "
            f"'{screening.CAT_STALE_HIGH}' so that missing recent data is not mistaken for low burden. "
            "Their present situation is unknown."
        )

    st.subheader("Find countries for further review")
    st.caption(
        "These filters change the displayed shortlist only; they do not alter the "
        "underlying screening rules or the full downloadable screening dataset."
    )

    priority_view = priority_df.copy()
    filter_col1, filter_col2 = st.columns([1, 2])
    with filter_col1:
        category_options = sorted(
            priority_view["Priority screen"].dropna().astype(str).unique().tolist()
        ) if "Priority screen" in priority_view.columns else []
        selected_categories = st.multiselect(
            "Screening categories",
            options=category_options,
            default=category_options,
            key="priority_categories_filter",
        )
    with filter_col2:
        country_search = st.text_input(
            "Search country or ISO code",
            placeholder="e.g. Algeria or DZA",
            key="priority_country_search",
        )

    if "Priority screen" in priority_view.columns:
        priority_view = priority_view[
            priority_view["Priority screen"].astype(str).isin(selected_categories)
        ]
    if country_search.strip():
        searchable_columns = [
            column for column in ("Country", "Country name", "Country / area", "Country code")
            if column in priority_view.columns
        ]
        if searchable_columns:
            search_mask = pd.Series(False, index=priority_view.index)
            query = country_search.strip().casefold()
            for column in searchable_columns:
                search_mask |= priority_view[column].fillna("").astype(str).str.casefold().str.contains(
                    query, regex=False
                )
            priority_view = priority_view[search_mask]

    st.caption(
        f"Showing {len(priority_view):,} of {len(priority_df):,} countries / areas."
    )
    _display = priority_view[screening.DISPLAY_COLUMNS].copy()
    _display["Latest estimate (%)"] = _display["Latest estimate (%)"].apply(
        lambda value: "\u2014" if pd.isna(value) else f"{float(value):.1f}%"
    )
    show_table(_display)
    with st.expander("How this screening works"):
        st.markdown(
            "- **Burden:** latest unambiguous observed estimate; peer ranking is restricted to the same latest reference year.\n"
            "- **Older estimates:** a country without a latest-year estimate is compared only with countries observed in its own last reference year. "
            "If it was in that year's higher quarter it is shown as higher last-known burden; otherwise as a generic data update. "
            "Persistence of such standing has been checked retrospectively (see `evaluation/results/`), but that is not a forecast for the country.\n"
            "- **Trajectory:** latest estimate minus the most recent earlier estimate with a non-overlapping reference window. A difference within \u00b10.05 percentage points is shown as little/no numerical change; this is a display tolerance, not a statistical significance test.\n"
            "- **Evidence availability:** a transparent coverage label based on recency and number of unambiguous reference years; it is not a probability that the estimate is correct.\n"
            "- **Conflicts:** conflicting country-year observations are excluded from the screening signal and counted separately.\n"
            "- **No forecast:** no machine-learning model, future prediction, or validated risk probability is claimed."
        )
else:
    st.info("No country-level records are available for priority screening.")

st.divider()


# ============================================================
# DECISION BRIEF EXPORT
# ============================================================
st.header("Decision brief: evidence for follow-up")
st.write(
    "Prepare a practical review record for a selected country or area. "
    "This brief supports human review; it does not prescribe aid allocation."
)

if not priority_df.empty:
    _brief_codes = priority_df["Country code"].dropna().astype(str).tolist()
    _default_brief_code = selected_country if selected_country in _brief_codes else _brief_codes[0]
    _brief_code = st.selectbox(
        "Choose a country / area for the brief",
        options=_brief_codes,
        index=_brief_codes.index(_default_brief_code),
        format_func=country_label,
        key="decision_brief_country",
    )
    _brief_row = priority_df[priority_df["Country code"] == _brief_code].iloc[0]
    _brief_latest = _brief_row["Latest estimate (%)"]
    _brief_latest_text = (
        "No unambiguous estimate available" if pd.isna(_brief_latest) else f"{float(_brief_latest):.1f}%"
    )
    _brief_year = "Not available" if pd.isna(_brief_row["Reference year"]) else str(int(_brief_row["Reference year"]))

    # Review-workflow fields personalise the exported brief. They are not saved
    # to a database; the user can retain them by downloading the generated file.
    st.subheader(f"Review workspace — {country_label(_brief_code)}")
    b1, b2, b3 = st.columns(3)
    b1.metric("Latest observed estimate", _brief_latest_text)
    b2.metric("Reference year", _brief_year)
    b3.metric("Screening category", str(_brief_row["Priority screen"]))

    _review_purpose = st.selectbox(
        "Purpose of this review",
        options=[
            "Initial evidence review",
            "Check whether data need updating",
            "Prepare questions for country-level assessment",
            "Compare evidence before a planning discussion",
        ],
        key="decision_brief_purpose",
    )
    _review_status = st.selectbox(
        "Review status",
        options=["Not started", "In progress", "Needs external verification", "Reviewed — follow-up required"],
        key="decision_brief_status",
    )
    _review_questions = st.text_area(
        "Questions or follow-up notes",
        placeholder=(
            "For example: Is a newer official estimate available? "
            "Has the source changed? What local information is needed to interpret this indicator?"
        ),
        help="These notes are included in the downloaded brief. They are not saved after the session unless you download the file.",
        key="decision_brief_notes",
    )

    st.markdown("**Why this country appears in the screening**")
    st.write(f"**Screening category:** {str(_brief_row['Priority screen'])}")
    if "Reason" in _brief_row.index and pd.notna(_brief_row["Reason"]):
        st.write(f"**Reason:** {str(_brief_row['Reason'])}")
    if "Evidence availability" in _brief_row.index and pd.notna(_brief_row["Evidence availability"]):
        st.write(f"**Evidence availability:** {str(_brief_row['Evidence availability'])}")
    if "Historical direction" in _brief_row.index and pd.notna(_brief_row["Historical direction"]):
        st.write(f"**Historical direction:** {str(_brief_row['Historical direction'])}")

    st.markdown("**Suggested follow-up steps**")
    for _step in screening.follow_up_steps(_brief_row):
        st.markdown(f"- {_step}")

    # Add a compact country-specific evidence snapshot to the export, using the
    # same validated observations that power the rest of the application.
    _brief_data = observations[observations["country_code"] == _brief_code].copy()
    _brief_clean = _brief_data[~_brief_data["conflict"]].drop_duplicates("year").sort_values("year")
    _brief_conflicts = _brief_data[_brief_data["conflict"]]
    _brief_sources = (
        int(_brief_clean["source"].nunique())
        if not _brief_clean.empty and "source" in _brief_clean.columns else 0
    )
    _brief_windows = (
        int(_brief_clean["year"].nunique())
        if not _brief_clean.empty else 0
    )
    _brief_conflict_years = int(_brief_conflicts["year"].nunique())
    _brief_latest_window = (
        str(_brief_clean.iloc[-1].get("window_label", "Not provided"))
        if not _brief_clean.empty else "Not available"
    )

    _brief_export = screening.build_brief_markdown(_brief_row).rstrip()
    _brief_export += (
        "\\n\\n## Review workspace\\n"
        f"- **Review purpose:** {_review_purpose}\\n"
        f"- **Review status:** {_review_status}\\n"
        f"- **Dataset latest reference year:** {latest_year}\\n"
        f"- **Latest reference window:** {_brief_latest_window}\\n"
        f"- **Unambiguous reference years available:** {_brief_windows}\\n"
        f"- **Distinct sources represented:** {_brief_sources}\\n"
        f"- **Years with conflicting observations:** {_brief_conflict_years}\\n"
        "\\n## Reviewer notes and questions\\n"
        + (_review_questions.strip() if _review_questions.strip() else "No reviewer notes entered.")
        + "\\n\\n## Interpretation and use\\n"
        "- This screening identifies a country for further review; it is not an official needs assessment.\\n"
        "- The indicator is historical and does not establish current conditions where data are old or missing.\\n"
        "- Confirm recent information and relevant local context before planning or allocating assistance.\\n"
        "- Reviewer notes entered here are included in this export only; this app does not persist them in a shared review system.\\n"
    )

    st.download_button(
        "Download review brief (.md)",
        data=_brief_export.encode("utf-8"),
        file_name=f"foodsignal_review_brief_{_brief_code.lower()}.md",
        mime="text/markdown",
        key="download_decision_brief",
    )
    st.download_button(
        "Download full priority screening table (.csv)",
        data=priority_df.to_csv(index=False).encode("utf-8-sig"),
        file_name="foodsignal_priority_screening.csv",
        mime="text/csv",
        key="download_priority_screening",
    )
else:
    st.info("No country-level records are available to create a decision brief.")
st.divider()

# ============================================================
# COUNTRY COMPARISON (ONE REFERENCE YEAR)
# ============================================================

st.header("Country comparison")

year_window = fsdata.window_label_for_year(observations, selected_year)
st.write(
    f"Estimates for reference year **{selected_year}** (window {year_window}). "
    "These are descriptive indicator estimates, not predictions."
)

comparable, withheld = fsdata.observations_for_year(observations, selected_year)
without_estimate = country_total - comparable["country_code"].nunique() - withheld[
    "country_code"
].nunique()

if not withheld.empty:
    st.warning(
        f"{withheld['country_code'].nunique()} country/countries have conflicting estimates for "
        f"{selected_year}. They are left out of the ranking (no value is averaged or chosen) "
        "and listed below for review."
    )
    with st.expander("Conflicting records withheld from this ranking"):
        show_table(fsdata.to_display_table(withheld))

if comparable.empty:
    st.warning(f"No unambiguous estimates are available for {selected_year}.")
else:
    comparable = comparable.sort_values(
        ["food_insecurity_percent", "country_name"], ascending=[False, True]
    ).reset_index(drop=True)
    comparable["rank"] = (
        comparable["food_insecurity_percent"].rank(method="min", ascending=False).astype(int)
    )

    chart_col, table_col = st.columns([1.4, 1], gap="large")

    with chart_col:
        top = comparable.head(min(15, len(comparable)))
        comparison_chart = px.bar(
            top,
            x="food_insecurity_percent",
            y="country_name",
            color="source",
            orientation="h",
            hover_data={
                "country_code": True,
                "window_label": True,
                "observation_status": True,
                "food_insecurity_percent": ":.1f",
                "source": False,
            },
            title=f"Highest estimates among countries with data \u2014 {selected_year}",
            labels={
                "country_name": "Country",
                "country_code": "ISO code",
                "window_label": "Reference window",
                "observation_status": "Status code",
                "food_insecurity_percent": "Estimated prevalence (%)",
                "source": "Source",
            },
        )
        comparison_chart.update_layout(
            height=540,
            xaxis=dict(rangemode="tozero", ticksuffix="%", title="Estimated prevalence (%)"),
            yaxis_title="",
            legend=dict(orientation="h", yanchor="top", y=-0.12, x=0),
            margin=dict(l=10, r=15, t=65, b=20),
        )
        set_category_order(comparison_chart, top["country_name"])
        show_chart(comparison_chart)

    with table_col:
        st.subheader("Comparison table")
        table = comparable[
            [
                "rank",
                "country_name",
                "country_code",
                "food_insecurity_percent",
                "window_label",
                "source",
                "observation_status",
            ]
        ].rename(
            columns={
                "rank": "Rank",
                "country_name": "Country",
                "country_code": "ISO code",
                "food_insecurity_percent": "Estimate (%)",
                "window_label": "Reference window",
                "source": "Source",
                "observation_status": "Status code",
            }
        )
        show_table(table, height=540)

    st.caption(
        f"{len(comparable)} countries/areas are ranked. {without_estimate} have no estimate "
        f"for {selected_year} and are not ranked; the dataset also does not cover every "
        "country, so rank is relative to the countries with data. Estimates are not a "
        "complete assessment of food insecurity or a forecast. Bars are coloured by source."
    )

st.divider()


# ============================================================
# HISTORICAL CHANGE EXPLORER (LIKE-FOR-LIKE)
# ============================================================

st.header("Historical change explorer")

st.write(
    "Compare two reference years. Only countries with an unambiguous estimate in "
    "**both** years are compared, so every bar covers the same two reference years. "
    "Countries that cannot be compared are listed with the reason."
)

default_start_target = latest_year - 4
default_start = max(
    (y for y in dataset_years if y <= default_start_target), default=dataset_years[0]
)
if default_start >= latest_year:
    default_start = dataset_years[0]

start_options = dataset_years[:-1] or dataset_years
start_index = start_options.index(default_start) if default_start in start_options else 0

select_col1, select_col2, select_col3, select_col4 = st.columns(4)
start_year = select_col1.selectbox(
    "From reference year",
    options=start_options,
    index=start_index,
)
end_options = [y for y in dataset_years if y > start_year]

if not end_options:
    st.info("Choose an earlier start year to compare two reference years.")
else:
    end_year = select_col2.selectbox(
        "To reference year",
        options=end_options,
        index=len(end_options) - 1,
    )
    mode_label = select_col3.selectbox(
        "Show",
        options=["Largest increases", "Largest decreases", "All eligible countries"],
    )
    mode = {
        "Largest increases": "increases",
        "Largest decreases": "decreases",
        "All eligible countries": "all",
    }[mode_label]
    if mode == "all":
        limit = None
        select_col4.caption("All eligible countries are shown.")
    else:
        limit = select_col4.selectbox("Number of countries", options=[10, 15, 20, 30], index=1)

    eligible, excluded = fsdata.compute_changes(observations, start_year, end_year)

    summary1, summary2, summary3 = st.columns(3)
    summary1.metric("Countries compared", f"{len(eligible):,}")
    summary2.metric("Countries not compared", f"{len(excluded):,}")
    summary3.metric(
        "Windows compared",
        f"{fsdata.window_label_for_year(observations, start_year)} vs "
        f"{fsdata.window_label_for_year(observations, end_year)}",
    )

    if eligible.empty:
        st.warning("No country has an unambiguous estimate in both selected years.")
    else:
        overlapping = int((eligible["windows_overlap_years"].fillna(0) > 0).sum())
        if overlapping:
            st.warning(
                f"For {overlapping} compared countries the two windows overlap, so part of the "
                "difference reflects shared underlying data. Choose years at least three "
                "years apart to compare non-overlapping windows."
            )
        switched = int(eligible["source_changed"].sum())
        if switched:
            st.info(
                f"{switched} compared countries changed source between the two years; "
                "see the 'Source changed' column."
            )

        plot_change = fsdata.select_changes(eligible, mode, limit)
        st.caption(f"Showing {len(plot_change)} of {len(eligible)} eligible countries.")

        change_chart = px.bar(
            plot_change,
            x="change_pp",
            y="country_name",
            color="direction",
            color_discrete_map=DIRECTION_COLORS,
            orientation="h",
            hover_data={
                "country_code": True,
                "label_start": True,
                "label_end": True,
                "value_start": ":.1f",
                "value_end": ":.1f",
                "change_pp": ":.1f",
                "direction": False,
            },
            title=f"Change in estimated prevalence, {start_year} to {end_year}",
            labels={
                "country_name": "Country",
                "change_pp": "Change (percentage points)",
                "country_code": "ISO code",
                "label_start": f"Window ({start_year})",
                "label_end": f"Window ({end_year})",
                "value_start": f"Estimate {start_year} (%)",
                "value_end": f"Estimate {end_year} (%)",
                "direction": "Direction",
            },
        )
        change_chart.update_layout(
            height=max(420, len(plot_change) * 24 + 140),
            xaxis_title="Change (percentage points)",
            yaxis_title="",
            legend=dict(orientation="h", yanchor="top", y=-0.08, x=0),
            margin=dict(l=10, r=15, t=65, b=20),
        )
        set_category_order(change_chart, plot_change["country_name"])
        show_chart(change_chart)

    st.caption(
        "This describes the difference between two estimates. It does not establish "
        "causation or predict the future. Source differences, estimate uncertainty and "
        "three-year averaging all affect interpretation."
    )

    with st.expander("View change table (all eligible countries)"):
        change_table = eligible.rename(
            columns={
                "country_name": "Country",
                "country_code": "ISO code",
                "value_start": f"Estimate {start_year} (%)",
                "value_end": f"Estimate {end_year} (%)",
                "label_start": f"Window ({start_year})",
                "label_end": f"Window ({end_year})",
                "change_pp": "Change (percentage points)",
                "direction": "Direction",
                "windows_overlap_years": "Overlapping window years",
                "source_start": f"Source ({start_year})",
                "source_end": f"Source ({end_year})",
                "source_changed": "Source changed",
            }
        )[
            [
                "Country",
                "ISO code",
                f"Estimate {start_year} (%)",
                f"Estimate {end_year} (%)",
                "Change (percentage points)",
                "Direction",
                f"Window ({start_year})",
                f"Window ({end_year})",
                "Overlapping window years",
                f"Source ({start_year})",
                f"Source ({end_year})",
                "Source changed",
            ]
        ]
        show_table(change_table)
        st.download_button(
            label="Download change table as CSV",
            data=change_table.to_csv(index=False).encode("utf-8"),
            file_name=f"foodsignal_change_{start_year}_{end_year}.csv",
            mime="text/csv",
        )

    with st.expander(f"Countries not compared ({len(excluded)}) and why"):
        if excluded.empty:
            st.write("Every country has an unambiguous estimate in both selected years.")
        else:
            show_table(
                excluded.rename(
                    columns={
                        "country_name": "Country",
                        "country_code": "ISO code",
                        "reason": "Reason",
                    }
                )
            )

st.divider()


# ============================================================
# DATA EXPLORER AND DOWNLOAD
# ============================================================

st.header("Data explorer")
st.write("Inspect the cleaned observations and download the filtered data.")

filter_col1, filter_col2, filter_col3 = st.columns(3)
with filter_col1:
    country_filter = st.selectbox(
        "Filter by country",
        options=["All countries"] + country_codes,
        format_func=lambda code: code if code == "All countries" else country_label(code),
    )
with filter_col2:
    year_filter = st.selectbox(
        "Filter by reference year",
        options=["All years"] + dataset_years,
    )
with filter_col3:
    review_only = st.checkbox("Only records flagged for review", value=False)

filtered = observations
if country_filter != "All countries":
    filtered = filtered[filtered["country_code"] == country_filter]
if year_filter != "All years":
    filtered = filtered[filtered["year"] == year_filter]
if review_only:
    filtered = filtered[filtered["record_status"] != fsdata.RECORD_SINGLE]

st.caption(f"{len(filtered):,} records match the selected filters.")
show_table(fsdata.to_display_table(filtered))

st.download_button(
    label="Download filtered data as CSV",
    data=fsdata.to_export_frame(filtered).to_csv(index=False).encode("utf-8"),
    file_name="foodsignal_filtered_data.csv",
    mime="text/csv",
)
st.caption(
    "The export keeps the ISO code, country name, original period (time_detail), "
    "source, status code and record check."
)

st.divider()


# ============================================================
# DATA QUALITY AND TRANSPARENCY
# ============================================================

st.header("Data quality & transparency")

quality_left, quality_right = st.columns(2, gap="large")

with quality_left:
    st.subheader("Cleaning pipeline statistics")

    try:
        summary = fsdata.load_cleaning_summary(SUMMARY_FILE)
    except Exception as error:  # noqa: BLE001
        summary = None
        st.error(f"cleaning_summary.csv could not be read: {error}")

    if summary is None:
        st.warning(
            "`data/processed/cleaning_summary.csv` was not found, so cleaning statistics "
            "cannot be shown. Run `python clean_data.py` with the raw FAO export "
            "to create it. The figures on the right come only from the loaded file."
        )
    else:
        card1, card2 = st.columns(2)
        card3, card4 = st.columns(2)
        for card, metric, label in (
            (card1, "invalid_rows", "Rows failing validation"),
            (card2, "exact_duplicate_rows_removed", "Exact duplicates removed"),
            (card3, "conflicting_country_year_rows", "Conflicting rows left out"),
            (card4, "clean_rows_saved", "Rows in dashboard file"),
        ):
            value = fsdata.summary_value(summary, metric)
            card.metric(label, "n/a" if value is None else f"{int(value):,}")

        show_table(fsdata.summary_for_display(summary))

        for message in fsdata.check_summary_consistency(
            summary, len(observations) + len(rejected), country_total
        ):
            st.warning(message)

        collapsed = fsdata.reconcile_cleaning_counts(summary)
        if collapsed is not None and collapsed > 0:
            st.warning(
                f"{collapsed:,} rows are not explained by the pipeline counts above. "
                "clean_data.py keeps one row per country-year in its last step, so rows with "
                "the same value but different metadata (for example a different source or "
                "period) are collapsed without being logged. They are not in this dashboard "
                "file; reviewing them requires the raw FAO export."
            )
        elif collapsed == 0:
            st.success("Pipeline counts reconcile: every excluded row is accounted for.")

    review = fsdata.load_optional_csv(REVIEW_FILE)
    invalid = fsdata.load_optional_csv(INVALID_FILE)
    st.markdown("**Review files**")
    for label, frame, path in (
        ("Conflicting country-year records", review, REVIEW_FILE),
        ("Invalid records", invalid, INVALID_FILE),
    ):
        if frame is None:
            st.caption(f"{label}: `{path.name}` not found.")
        elif frame.empty:
            st.caption(f"{label}: `{path.name}` contains no records.")
        else:
            with st.expander(f"{label}: {len(frame):,} records in `{path.name}`"):
                show_table(frame)
                st.download_button(
                    label=f"Download {path.name}",
                    data=frame.to_csv(index=False).encode("utf-8"),
                    file_name=path.name,
                    mime="text/csv",
                    key=f"download_{path.name}",
                )

with quality_right:
    st.subheader("What the loaded file contains")
    st.caption(
        "Descriptive facts about the file loaded by this app. They do not prove "
        "the underlying estimates are error-free."
    )

    conflict_keys = int(
        observations.loc[observations["conflict"], ["country_code", "year"]]
        .drop_duplicates()
        .shape[0]
    )
    repeat_rows = int((observations["record_status"] == fsdata.RECORD_REPEAT).sum())

    loaded1, loaded2 = st.columns(2)
    loaded1.metric("Rejected when loading", f"{len(rejected):,}")
    loaded2.metric("Unresolved country-year conflicts", f"{conflict_keys:,}")
    loaded3, loaded4 = st.columns(2)
    loaded3.metric("Rows repeating a same-value estimate", f"{repeat_rows:,}")
    loaded4.metric(
        "Windows centred on reference year",
        f"{int(observations['window_centered'].sum()):,} of {len(observations):,}",
    )

    if len(rejected):
        with st.expander("Rows rejected when loading"):
            show_table(rejected)

    by_source = (
        observations.groupby("source").size().rename("Records").reset_index().rename(
            columns={"source": "Source"}
        )
    )
    by_status = (
        observations.groupby("observation_status")
        .size()
        .rename("Records")
        .reset_index()
        .rename(columns={"observation_status": "Status code"})
    )
    st.markdown("**Records by source**")
    show_table(by_source)
    st.markdown("**Records by status code**")
    show_table(by_status)
    st.caption(
        "Status codes are carried over unchanged from the FAO export; consult FAO's "
        "metadata for their definitions."
    )

st.divider()

st.subheader("Interpretation limits")
st.markdown(
    """
    - The indicator is an estimated prevalence of food insecurity, not a complete picture of food security.
    - Every estimate is a three-year average; adjacent reference years share two underlying years.
    - Sources and status codes differ between records, and some countries switch source over time.
    - Coverage varies by country and year. A missing estimate does not mean food insecurity is absent.
    - Rankings only cover countries present in the dataset.
    - Conflicting estimates are withheld and flagged, never averaged.
    - Historical differences do not establish causation or predict future outcomes.
    """
)

st.subheader("Source & methodology")
st.markdown(
    """
    **Primary source:** Food and Agriculture Organization of the United Nations (FAO),
    Sustainable Development Goal indicator 2.1.2.

    [Explore FAO food-security data](https://dataexplorer.fao.org/)

    **Data preparation:** `clean_data.py` selects the total-population series, validates
    country codes, years and percentage values, and writes conflicting and invalid records to
    review files instead of silently treating them as reliable estimates.

    **Current analytical scope:** descriptive exploration of historical records. No forecasting
    or risk model is implemented or claimed.
    """
)

st.caption(
    "FoodSignal supports human analysis. It does not replace official "
    "food-security assessments or expert judgment."
)
