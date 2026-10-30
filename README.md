\# FoodShield AI

\### A Data-Driven Food Insecurity Screening Dashboard



\## Overview



FoodShield AI is a data-driven dashboard designed to help users explore historical food insecurity estimates, identify countries that may warrant further review, and examine the evidence behind screening results.



The project aims to make historical data more accessible and support evidence-informed review. It is a screening and analysis tool, not an official humanitarian assessment system.



\## Objectives



\- Explore food insecurity estimates across countries and years.

\- Examine country-level historical trends.

\- Compare countries using available estimates.

\- Identify cases that may warrant further review, including countries with outdated estimates.

\- Generate decision briefs and export screening results.

\- Communicate data quality, uncertainty, and methodological limitations.



\## Dataset


## Dataset and Data Source

FoodShield AI uses data from the Food and Agriculture Organization of the United Nations (FAO), specifically SDG Indicator 2.1.2: *Prevalence of moderate or severe food insecurity in the population, based on the Food Insecurity Experience Scale (FIES).*

**Official sources:**
- [FAO SDG Indicator 2.1.2](https://www.fao.org/sustainable-development-goals-data-portal/data/indicators/212-prevalence-of-moderate-or-severe-food-insecurity-in-the-population-based-on-the-food-insecurity-experience-scale/2/)
- [FAO Data Explorer — Indicator 2.1.2](https://dataexplorer.fao.org/)

The raw dataset file in this project is `FAO,DF_SDG_2_1_2,1.0+all.csv`. The application uses a processed dataset at `data/processed/food_insecurity_percentages.csv`.

The current processed dataset contains 1,280 observations covering 157 countries and areas over the reference years 2015–2024. These counts describe the processed subset used by the application, not the full FAO dataset.

The data include estimates of moderate or severe food insecurity. Values may be unavailable for some country-years, and the most recent reference period in the raw file may not contain an estimate for every country.

For reproducibility, the project retains data-cleaning and audit outputs under `data/processed/`, alongside evaluation documentation in `evaluation/`.


\## Methodology



FoodShield AI provides descriptive analysis of historical food insecurity estimates.



Its stale-data screening rule uses a country's latest available estimate and its relative standing among countries observed in the same reference year. When an estimate is outdated but falls within the higher quarter of that year's observed estimates, the country can be flagged as having a \*\*higher last-known burden\*\*.



This label preserves historical context. It does not establish the country's current food insecurity level.



\## Evaluation



The stale-standing rule was evaluated using countries with observations available at a later time.



For the three-year evaluation horizon, the reported results were:



| Metric | Result |

|---|---:|

| Country-year pairs | 810 |

| Flagged pairs | 201 |

| Precision | 92.0% |

| Recall | 94.4% |

| Base rate | 24.2% |

| Lift over base rate | 3.80 |



These results measure the persistence of relative standing in observed data. They do not validate humanitarian priority decisions or establish predictive performance for future crises.



The evaluation also reports a cross-year-cutoff baseline, country-clustered bootstrap intervals, and diagnostics for countries without a later observation. See `evaluation/stale\_standing\_evaluation.md` for the full results and limitations.



\## Dashboard Features



\- \*\*Global overview:\*\* Explore the overall dataset.

\- \*\*Country trend:\*\* Examine historical estimates for an individual country.

\- \*\*Evidence and historical signal:\*\* Inspect historical context.

\- \*\*FoodShield priority screening:\*\* Review cases flagged for further investigation.

\- \*\*Decision brief:\*\* Prepare a country-level summary for review.

\- \*\*Country comparison:\*\* Compare available country estimates.

\- \*\*Historical change explorer:\*\* Examine changes over time.

\- \*\*Data explorer:\*\* Inspect the underlying observations.

\- \*\*Data quality and transparency:\*\* Review data coverage and limitations.



Available exports include screening results and decision briefs.



\## Technology Stack



\- Python

\- Streamlit

\- Pandas

\- Plotly



\## Installation



\### 1. Clone or download the repository



Open a terminal in the project directory.



\### 2. Create a virtual environment



```bash

python -m venv .venv

```



\### 3. Activate the environment



On Windows PowerShell:



```powershell

.\\.venv\\Scripts\\Activate.ps1

```



\### 4. Install dependencies



```bash

python -m pip install -r requirements.txt

```



\## Run the Application



From the project root, run:



```bash

python -m streamlit run app.py

```



Streamlit will provide a local URL, usually `http://localhost:8501`, that you can open in your browser.



\## Run Tests



Run the project's automated tests with:



```bash

python -m unittest discover -s tests -v

```



\## Limitations and Responsible Use



\- Historical estimates may not represent current conditions.

\- Relative ranking is not the same as an official humanitarian priority threshold.

\- Evaluation results apply to observed countries and years; missing future observations cannot be evaluated in the same way.

\- Repeated observations from the same countries can affect statistical uncertainty.

\- The screening rule is not a forecasting model and does not establish causality.

\- Results should be reviewed alongside current information, local context, and expert judgment.



\*\*FoodShield AI is intended to support human review, not replace it.\*\* A screening flag should prompt further investigation rather than automatically determine the allocation of aid.



\## Project Status



FoodShield AI includes an interactive dashboard, historical screening logic, evaluation documentation, and automated tests. Further work may include broader usability testing, updated data integration, and evaluation on additional datasets.



\## License



A license has not yet been specified. Confirm the project's intended license before redistributing the code.



