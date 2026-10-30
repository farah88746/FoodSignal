# FoodShield AI — Data & Indicator Validation Note

## Scope
This review checks the supplied processed file `food_insecurity_percentages.csv` and the descriptive, peer-relative screening logic. It does not validate the indicator against ground truth or establish predictive accuracy.

## Dataset profile
- Rows: 1,280
- Countries/areas: 157
- Reference years: 2015–2024
- Distinct source labels: 2
- Duplicate country-year keys: 0
- Percentage range: 1.0%–89.8%
- Observation status counts: {'E': 1154, 'A': 126}

## Key findings
1. **No duplicate country-year keys** were found in the supplied processed file.
2. **All values are within 0–100** and the file spans reference years 2015–2024.
3. **Coverage is uneven.** Countries have different numbers of observations and different last-observed years. A country with a high estimate but sparse or old data should not be treated as directly equivalent to a country with current, more complete data.
4. **The latest-year peer cutoff is relative, not absolute.** The 75th percentile for year 2024 is 42.6%. This is a statistical grouping within the observed peer set, not a validated humanitarian threshold.
5. **Trend labels are descriptive.** Three-year moving-average reference windows overlap from year to year. Comparisons should use non-overlapping windows, and short series should be flagged as insufficient rather than assigned a confident direction.
6. **Source labels do not prove independent corroboration.** A count of distinct source labels is not equivalent to multiple independent measurements.
7. **No predictive performance can be claimed** from this dataset alone. No out-of-sample evaluation against a defined outcome has been performed.

## Recommended changes before competition
- Keep burden, historical direction, and evidence coverage as separate dimensions.
- Exclude stale or non-comparable observations from same-year peer rankings; list them in a data-review queue.
- Display the exact reference window beside each percentage.
- Add unit tests for missing years, duplicate keys, invalid percentages, sparse series, and conflicting observations.
- Describe the tool as a transparent screening / decision-support prototype, not an AI forecast or an automatic aid allocator.
- In the demo, choose one country with a clear historical pattern and one with limited evidence to demonstrate how the app avoids false certainty.

## Files
- `data_validation_checks.csv`: basic data integrity checks.
- `country_data_coverage.csv`: observation coverage per country.
- `latest_year_peer_comparison.csv`: comparable latest-year peer comparison.
- `country_review_flags.csv`: cases requiring data recency / coverage review.

## Limitations
This is a file-level and logic-level review based on the supplied processed CSV and displayed app behavior. It does not independently verify the underlying FAO records, assess survey methodology, or measure predictive accuracy.
