## Project Overview: Vacation Rental Pricing recommendation 

Synthetic-data, end-to-end dynamic pricing pipeline for the Dynamic Pricing, Demand forecasting & Revenue management portfolio.

## 1. Business objective

MyHolidayRentals's business goal is to maximise **gross merchandise value (GMV)** — the booking value flowing through the platform.

```
GMV (per booking) = nights × shown_price
```

Platform monetization (take-rate × GMV + per-booking processing fees) and channel-distribution costs are **out of scope for v1** — they don't change the optimum under a constant % take rate, so the argmax is identical. A fees-sensitivity memo is a deferred extension.

**Optimization objective for Phase 5** — expected net GMV per shopping event:

```
E[net GMV] = nights × shown_price × P(book | price, context) × (1 − P(cancel | price, context))
```

Every recommendation in the deliverable CSV is chosen to maximise this, and `regret_pct` is computed against the DGP-truth optimum of the same quantity.

## 2. Final deliverable

A single CSV: **`predictions/2027_price_recommendations.csv`**

| column | description |
|---|---|
| `property_id` | synthetic vacation-rental ID |
| `stay_date` | a date in 2027 |
| `assumed_los_nights` | representative length-of-stay used for the recommendation (property's modal LOS) |
| `recommended_base_price` | output of the optimization step (host-set listing price, per night) |
| `recommended_discount_pct` | optimization-chosen promotional discount, if any |
| `recommended_shown_price` | `recommended_base_price × (1 − recommended_discount_pct)` (per night) |
| `expected_booking_prob` | P(book) at the recommended price |
| `expected_cancel_prob` | P(cancel \| booked) at the recommended price |
| `expected_per_booking_gmv` | `assumed_los_nights × recommended_shown_price × expected_booking_prob × (1 − expected_cancel_prob)` |
| `true_optimal_shown_price` | from the known DGP — synthetic-only luxury |
| `true_optimal_per_booking_gmv` | DGP-truth net GMV at the true optimum |
| `regret_pct` | `(true_optimal_per_booking_gmv − expected_per_booking_gmv) / true_optimal_per_booking_gmv` |

Plus a short memo on regret distribution by region / quality tier / lead-time bucket. The presence of `true_optimal_shown_price` and `regret_pct` is the whole reason we go synthetic — we can *grade* the policy, not just defend it.

## 3. Why synthetic only

- We own the data-generating process → known true elasticity → causal estimator is *falsifiable*. On real data, "I estimated β = −1.4" is unverifiable.
- We can bake in vacation-rental dynamics that map to MyHolidayRentals (per-property heterogeneity, LOS, lead time, regional seasonality, weekend premium, host pricing styles).
- We avoid selection-on-shown-price and missing-data issues that limit public hotel datasets.
- Tradeoff to acknowledge in the README: synthetic risks "too clean." Mitigation: deliberately inject confounding, noise, and a host-pricing rule that creates real identification problems.

## 4. Pipeline — 5 phases

### Phase 1 — Synthetic DGP
Parameterised generator under `src/synthetic/`. Outputs to `data/synthetic/`.

**Entities**
- Properties (~300) — modelled like a MyHolidayRentals listing card. Static attributes:
  - **Identity**: `property_id`, `region`, `country`, `quality_tier ∈ {budget, mid, premium}`, `host_pricing_style ∈ {static, dynamic, premium}`, `base_price` (per-night reference)
  - **Type & size**: `property_type ∈ {apartment, house, villa, cottage, cabin, studio}`, `bedrooms`, `bathrooms`, `capacity` (sleeps N)
  - **Amenities (multi-hot bools)**: `wifi`, `pool`, `parking`, `ac`, `kitchen`, `washing_machine`, `garden`, `bbq`
  - **Audience flags**: `kid_friendly`, `pet_friendly`
  - **View / location**: `view ∈ {sea, mountain, garden, city, none}`, `distance_to_beach_km`, `distance_to_transit_km`, `distance_to_center_km`
  - **Reputation**: `rating` (0–5, drawn around quality_tier mean), `review_count`
  - **Policy**: `cancel_policy ∈ {flexible, moderate, strict}`
  - All attributes drawn correlated with `quality_tier` (premium properties skew higher rating, more amenities, sea view, lower distance-to-center, etc.) so the entity is internally consistent.
- Calendar: 2024-01-01 → 2027-12-31 (3 yrs train + 1 yr forecast horizon)
- Observations: one row per `(property × stay_date × lead_time)` shopping event

**Demand drivers (the latent truth)**

*Date-level (time-varying)*
- Annual seasonality (sinusoidal, region-specific peaks)
- Holiday/event spikes (regional)
- Weekday/weekend effect
- Lead-time decay + last-minute segment
- Length-of-stay multiplier

*Property-level (static attributes feeding demand)*
- Rating + review_count → trust signal, increases P(book) at any price
- Amenities & view → quality signal
- Kid-friendly × school-holiday season → interaction effect (family demand spikes)
- Distance to beach / transit → location appeal
- Cancel policy: flexible → higher P(book), but also higher P(cancel)
- Per-property quality residual on top of observable attributes

**Treatment — decomposed price**
Shown price has two components, generated separately so each carries its own elasticity:

- `base_price` — host's listing price; slow-moving, set per `(property × season)` by a host pricing rule that depends on seasonality, weekend, region, quality tier
- `discount_pct` — promotional reduction layered on top, triggered by *observable* rules:
  - **Last-minute discount** (lead_time ≤ 7d) — moderate cut
  - **Early-bird discount** (lead_time ≥ 90d) — small cut
  - **Length-of-stay discount** (LOS ≥ 7 nights) — moderate cut
  - **Host promo** (random property-week campaigns, drawn from observable property attributes)
- `shown_price = base_price × (1 − discount_pct)`

The split lets us recover **two distinct elasticities** in Phase 4: `β_base` (level effect) vs `β_discount` (framing/promo effect). In RM these are not the same.

**Outcomes — two-stage**

1. **Booking**: Bernoulli with logit linear in `log(base_price)`, `discount_pct`, demand drivers, property FE
   - True `β_base` heterogeneous by `quality_tier` (e.g., premium less elastic than budget)
   - True `β_discount` heterogeneous by `lead_time bucket` and `LOS` (last-minute and long-stay discounts more potent)
2. **Cancellation**: conditional on booking, Bernoulli with logit linear in `log(shown_price)`, `lead_time`, `LOS`, deposit signal
   - True `β_cancel` small but non-zero — directionally: higher price → slightly higher cancel rate (buyer's remorse). This breaks the "lowest price wins" trap in optimization.

All confounders driving price/discount/cancellation are observable → unconfoundedness still holds by construction.

**Identification strategy (observational only)**
- **No randomized-price slice, no IV.** DML must do the work.
- **Design constraint:** every variable that the host pricing rule depends on must also be observable to DML as a control. No unobserved demand shocks driving price. This makes unconfoundedness hold by construction — confounding can be arbitrarily strong, but it has to be explainable by observables.
- Naive regression of `bookings ~ price` will still give the wrong answer (because controls aren't included); DML's job is to recover the truth by partialling out the observed confounders.

**DGP acceptance criteria**
- Aggregate booking rate in a realistic range (15–30% of shown events)
- Seasonality, weekend, lead-time effects visible in EDA plots
- Naive `bookings ~ price` regression yields the **wrong sign or magnitude** vs the true β — so the causal phase has something real to fix

### Phase 2 — EDA & DGP sanity
- Aggregate stats by season, region, lead time, LOS, weekend
- Side-by-side: naive correlational elasticity vs. known truth (sets up the causal motivation)
- A short notebook that doubles as the EDA section of the final write-up

### Phase 3 — Booking + cancellation probability models (two-stage)
Two predictive models, both calibrated, feeding the optimization step.

**3a. Booking model** — predicts `P(booking)`
- Target: `booking_bool` per `(property × stay_date × base_price × discount_pct × lead_time × LOS)`
- Features:
  - *Pricing*: `log(base_price)`, `discount_pct`
  - *Date-level*: lead_time, dow, month, holiday flag, school-holiday flag, LOS
  - *Property attributes*: property_type, bedrooms, bathrooms, capacity, amenities (multi-hot), kid_friendly, pet_friendly, view, distance_to_beach_km, distance_to_transit_km, distance_to_center_km, rating, review_count, cancel_policy, region
  - *Property identity*: property embedding / FE
  - *Host*: host_pricing_style
- Models: **LogReg (baseline) → LightGBM → XGBoost → calibrated (isotonic)**. Compare all three; LogReg is the interpretable benchmark, LGBM/XGB the production candidates.

**3b. Cancellation model** — predicts `P(cancel | booked)`
- Target: `cancelled_bool` conditional on a booking
- Features: `log(shown_price)`, lead_time, LOS, cancel_policy, rating, review_count, region, property FE
- Models: **LogReg → LightGBM → XGBoost → calibrated**

**Eval (both)**: AUC, Brier, reliability plot. Time-based split — see §8.

### Phase 4 — Causal elasticity (three estimands)
Observational identification only — no IV, no randomized slice. Relies on unconfoundedness given observed controls (which the DGP guarantees by construction).

Three separate causal questions, each estimated with Double ML and validated against DGP truth:

Controls in every DML estimand include both date-level demand drivers (seasonality, weekend, lead time, LOS, holidays) **and property attributes** (property_type, amenities, view, rating, review_count, distances, cancel_policy, etc.) — all are observable confounders since the host pricing rule depends on them.

1. **`β_base` — booking elasticity to base price**
   - Outcome: booking, treatment: `log(base_price)`, controls: date-level drivers + property attributes + `discount_pct` + property FE
2. **`β_discount` — booking elasticity to discount %**
   - Outcome: booking, treatment: `discount_pct`, controls: date-level drivers + property attributes + `log(base_price)` + property FE
   - Compare β_base and β_discount: are they the same, or does discount framing have its own effect?
3. **`β_cancel` — cancellation elasticity to shown price**
   - Outcome: cancellation (conditional on booking), treatment: `log(shown_price)`, controls: lead_time, LOS, cancel_policy, rating, property attributes, property FE

**Heterogeneity (Causal Forest)**: each estimand gets a heterogeneity pass over `quality_tier`, `region`, `lead_time bucket`, `LOS bucket`, `rating bucket`, `kid_friendly`, `view`. Output: HTE plots showing where elasticity is steepest. Likely findings to flag in the memo: low-rating properties more elastic; kid-friendly properties less elastic during school holidays.

**Validation**: estimated β vs known true β per segment → bias + CI coverage. This is the differentiator slide.

**Sensitivity check**: rerun DML with one control deliberately omitted (e.g., drop seasonality) to show how bias creeps back in — turns the unconfoundedness assumption into a discussion point rather than a hidden caveat.

### Phase 5 — Price optimization → 2027 recommendations
For each `(property × stay_date in 2027)`, fix a representative lead-time and LOS context (property's modal LOS), then jointly choose `(base_price, discount_pct)` to maximise expected net GMV per booking:

1. Predicted demand curve from Phase 3 + Phase 4 (booking + cancellation, both elasticities)
2. Solve
   ```
   (base*, disc*) = argmax  nights · shown_price · P(book | base, disc, ctx) · (1 − P(cancel | shown_price, ctx))
   ```
   under bounds: `base ∈ [0.5 × base_default, 1.5 × base_default]`, `disc ∈ [0, 0.30]`
3. Compare against the DGP's true optimal `(base, disc)` and true net GMV → `regret_pct`
4. Write the deliverable CSV
5. Memo: regret by region, quality_tier, lead_time bucket. Where does the policy win, where does it lose, and why. Highlight cases where the cheapest price is *not* the most profitable (cancellation effect dominates).

## 5. Repo layout

```
src/
  components/                        # reusable building blocks (no orchestration)
    data_loader.py                   # load_split / load_all / load_synthetic_events
    features.py                      # build_features: drop leakage, one-hot, derive
    synthetic/                       # DGP generator
      properties.py                  # static property table + outcome simulator
    booking_model/                   # Phase 3a primitives
      train_book.py                  # train_models, evaluate, save_models
    cancellation_model/              # Phase 3b primitives
    causal/                          # Phase 4 — DML × {β_base, β_discount, β_cancel} + Causal Forest
    optimize/                        # Phase 5 — joint (base, discount) optimization
    reporting/                       # Phase 5 reporting helpers
  pipeline/                          # thin orchestrators (one entry-point per phase)
    generate_synthetic_data.py       # Phase 1
    split_data.py                    # Phase 1.5 — time-based train/val/test split
    train_booking_model.py           # Phase 3a (load → features → train → eval → save)
    train_cancellation_model.py      # Phase 3b
    estimate_causal_effects.py       # Phase 4
    optimize_prices_2027.py          # Phase 5 (optimization)
    build_report.py                  # Phase 5 (CSV + memo)
notebooks/                           # EDA + per-phase walkthrough (eda.ipynb)
data/
  synthetic/                         # raw DGP output (synthetic_demand_events.parquet)
  processed/                         # train.parquet / val.parquet / test.parquet
models/                              # joblib artefacts: booking/, cancellation/
predictions/                         # 2027_price_recommendations.csv
reports/                             # short memos per phase
config/config.yaml                   # data paths, split cutoffs, model params
```

**Layering rule:** `components/` holds pure functions and stateless logic — no
file paths, no `main()`, no I/O orchestration. `pipeline/` scripts wire the
components together, read paths/params from `config.yaml`, and serve as the
`uv run python -m src.pipeline.<phase>` entry-points. Following this rule keeps
every component reusable across phases (e.g., `data_loader` and `features` are
called from Phase 3a, 3b, 4, and 5 alike).

## 6. General Pricing Data Scientist JD mapping

| JD bullet | Phase |
|---|---|
| Demand forecasting | Phase 3a (booking prob conditional on context) |
| Price elasticity (causal) | **Phase 4 — three estimands: β_base, β_discount, β_cancel** |
| Conversion probability (booking → stay) | Phase 3b (cancellation model) + Phase 4 (β_cancel) |
| Dynamic pricing strategy | Phase 5 — joint (base price, discount) optimization on **expected net GMV** |
| Experimentation | Memo on how the policy *would* be A/B tested in prod (MDE, switchback, guardrails) — pure observational identification in the build itself |
| Monitoring | Light drift comparison 2026 vs 2024 (stretch) |
| Communication | Per-phase memo + final CSV memo |

## 7. Build order

1. **Phase 1 DGP** — highest-leverage step; everything downstream is judged against it
2. **Phase 1.5 Data split** — time-based train/val/test on `stay_date` (see §8)
3. **Phase 2 EDA** — validates the DGP (run on train only — keep val/test held out)
4. **Phase 3a booking model**, then **3b cancellation model**
5. **Phase 4 causal** — DML × 3 estimands, validate each against DGP truth
6. **Phase 5 optimization** (joint base + discount on net GMV) → 2027 CSV
7. **Reporting** — memos + README write-up

## 8. Locked decisions (dated 2026-04-28)

**Geometry & elasticities**
1. **Geography**: ~300 properties across 5 regions / 3 countries.
2. **True booking elasticity by tier (β_base)**: `β_premium = −0.8`, `β_mid = −1.2`, `β_budget = −1.6`.
3. **Inventory**: skip for v1 — each shopping event independent. Revisit later.
4. **Confounding strength**: strong host-pricing response to seasonality / weekend / lead time.

**Discount mechanics**
5. **Discount triggers**:
   - Last-minute (lead_time ≤ 7d): 10–15%
   - Early-bird (lead_time ≥ 90d): 5–10%
   - LOS (≥ 7 nights): 8–12%
   - Host promo (random property-week campaign): 15–25%
   - Triggers stack additively, capped at 30%.
6. **`β_discount` ≈ 1.5 × |`β_base`|** in magnitude, **positive sign** (discount framing more potent than equivalent level change; higher `discount_pct` increases P(book)). So `β_discount_premium ≈ +1.2`, `β_discount_mid ≈ +1.8`, `β_discount_budget ≈ +2.4`. Sign convention: `β_base` is negative because the booking utility is linear in `log(date_base_price)`; `β_discount` is positive because the booking utility is linear in raw `discount_pct` (a discount entering as a *negative* log-price term would carry the same negative sign as `β_base`, but the DGP uses the linear form for simplicity, so the sign flips).

**Cancellation**
7. **Base cancellation rate**: ~20%.
8. **`β_cancel`**: mildly *positive* — price up ~10% → cancel rate up ~1pp (buyer's-remorse story). Breaks the "lowest price wins" trap in optimization.

**Data split (dated 2026-04-29)**

Time-based split on `stay_date`. Realism framing: today is 2026-04-29, so only stay dates ≤ today are "observed"; everything later is the forecast horizon.

| split | range | rows | role |
|---|---|---|---|
| train | 2024-01-01 → 2025-04-30 | 145,800 (33%) | fit booking + cancel models, fit DML nuisances |
| val   | 2025-05-01 → 2026-04-29 | 109,200 (25%) | hyperparam tuning, calibration, model selection |
| test  | 2026-04-30 → 2027-12-31 | 183,300 (42%) | held-out eval + Phase 5 optimization input (2027 deliverable) |

- **Trailing-12-month val** ensures val captures a full annual seasonality cycle (incl. summer 2025) — critical because GMV is concentrated at peak.
- **Val ends "today"** → matches a realistic deployment posture. Test = strict future-from-today.
- **All 300 properties present in all three splits** by construction (the DGP cross-joins properties × dates). Required because Phase 5 must price every property × stay_date in 2027 and Phase 3 uses property fixed effects.
- Cold-start ("leave-properties-out") evaluation is explicitly out of scope — flagged as a future-extension memo.
- Cutoffs live in `config/config.yaml → data_split`. Run via `uv run python -m src.pipeline.split_data`. Outputs `data/processed/{train,val,test}.parquet`.

## 9. Data dictionary — `synthetic_demand_events.parquet`

One row per `(property_id × stay_date)` shopping event. Generated by `src/pipeline/generate_synthetic_data.py`.

### 9.1 Property identity & static attributes 

| column | type | description | explanation |
|---|---|---|---|
| `property_id` | str (`P0000`–`P0299`) | Stable property identifier | Generated as `f"P{i:04d}"` in creation order; immutable across dates |
| `region` | enum {Brittany, Provence, Catalonia, Andalusia, Tuscany} | Coarse geography of the listing | Drawn uniformly across 5 regions; each region carries a `price_factor` used inside `base_price` |
| `country` | enum {France, Spain, Italy} | Country containing the region | Lookup from region (Brittany/Provence→France, Catalonia/Andalusia→Spain, Tuscany→Italy) |
| `quality_tier` | enum {budget, mid, premium} | Latent quality label that **drives almost every other attribute** | Drawn from {budget: 0.4, mid: 0.4, premium: 0.2}; deterministic input to all downstream draws |
| `host_pricing_style` | enum {static, dynamic, premium} | Host's pricing posture | Drawn from {static: 0.5, dynamic: 0.3, premium: 0.2}; placeholder for future heterogeneous pricing rules — no effect on outcomes today |
| `property_type` | enum {apartment, house, villa, cottage, cabin, studio} | Listing format | Drawn per tier — budget skews studio/apartment, premium skews villa/house |
| `bedrooms` | int [1, 8] | Number of bedrooms | Poisson(λ_tier), λ ∈ {1.5, 2.5, 3.5}; clipped to [1, 8]; studios forced to 1 |
| `bathrooms` | int [1, bedrooms] | Number of bathrooms | `round(bedrooms/2 + N(0, 0.3))`; clipped to [1, bedrooms] |
| `capacity` | int [1, 16] | How many guests the listing sleeps | `2 × bedrooms + Uniform({-1, 0, 1})`; clipped to [1, 16]; studios capped at 2 |
| `wifi`, `pool`, `parking`, `ac`, `kitchen`, `washing_machine`, `garden`, `bbq` | binary {0,1} | Amenity flags | Bernoulli(p_tier_amenity); per-tier probabilities (e.g. pool: 0.05/0.20/0.60 for budget/mid/premium) |
| `kid_friendly` | binary {0,1} | Suitable-for-children flag | Bernoulli with p ∈ {budget: 0.30, mid: 0.50, premium: 0.40} |
| `pet_friendly` | binary {0,1} | Pets-allowed flag | Bernoulli with p ∈ {budget: 0.40, mid: 0.30, premium: 0.20} |
| `view` | enum {sea, mountain, garden, city, none} | Type of view from the listing | Drawn per tier — premium ~45% sea, budget ~50% none |
| `distance_to_beach_km` | float ≥ 0 | km to the nearest beach | Lognormal, median ∈ {budget: 8, mid: 5, premium: 2}, σ = 0.5 |
| `distance_to_transit_km` | float ≥ 0 | km to the nearest public transit | Lognormal, median ∈ {budget: 1.5, mid: 1.2, premium: 2.0}, σ = 0.5 |
| `distance_to_center_km` | float ≥ 0 | km to city/town centre | Lognormal, median ∈ {budget: 2.0, mid: 2.5, premium: 3.0}, σ = 0.5 |
| `rating` | float [1.0, 5.0] | Average review score | Normal(μ_tier, σ_tier), μ ∈ {3.8, 4.2, 4.6}; clipped to [1, 5] and rounded to 1 decimal |
| `review_count` | int ≥ 0 | Number of reviews | Lognormal(μ_tier, σ_tier), μ ∈ {3.0, 3.5, 4.0} |
| `cancel_policy` | enum {flexible, moderate, strict} | Host-set cancellation policy | Drawn per tier — budget 40/40/20, premium 20/40/40 |
| `base_price` | float (€) | Per-night listing reference price; **static per property** | `tier_anchor + (capacity − 2) × 15` then `× (1 + amenity_uplifts) × view_uplift × region_factor × lognormal_noise(σ = 0.10)`. Tier anchors: budget 60, mid 120, premium 280 |

### 9.2 Calendar / shopping context — *from `generate_synthetic_data.py:generate_calendar_events`*

| column | type | description | explanation |
|---|---|---|---|
| `stay_date` | date (2024-01-01 → 2027-12-31) | Target stay date for the shopping event | Cartesian product: every property × every day in `[START_DATE, END_DATE]` |
| `lead_time` | int [0, 180] | Days between booking moment and stay | Exponential(scale = 30); clipped to [0, 180] — most searches close to the date, long tail of early-birds |
| `los` | int [1, 14] | Length of stay in nights | Per-property *modal* LOS ∈ {2, 3, 7} drawn at p = {0.3, 0.4, 0.3}; jittered by Uniform({-1, 0, 1}) per row; clipped to [1, 14] |
| `month` | int [1, 12] | Calendar month of `stay_date` | `stay_date.dt.month` |
| `day_of_week` | int [0, 6] | Day of week (Mon=0 … Sun=6) | `stay_date.dt.dayofweek` |
| `is_weekend` | binary {0,1} | Fri/Sat/Sun flag | `day_of_week ∈ {4, 5, 6}` — vacation-rental "weekend" convention |

### 9.3 Pricing (treatment) — *from `generate_synthetic_data.py:apply_host_pricing_logic`*

| column | type | description | explanation |
|---|---|---|---|
| `seasonal_mult` | float ≈ [0.7, 1.3] | Annual seasonality multiplier on price | `1 + 0.3 × cos(2π × (month − 7) / 12)` — peaks at 1.30 in July, troughs at 0.70 in January |
| `weekend_mult` | float ∈ {1.00, 1.15} | Weekend markup multiplier | `1 + 0.15 × is_weekend` — flat 15% bump on Fri/Sat/Sun |
| `host_price_shock` | float ~ N(1.0, 0.05) | Exogenous "host whim" noise | Pure random multiplier per row — **the only source of identifying variation for β_base** under DML (after partialling out month/weekend) |
| `date_base_price` | float (€) | Per-night base price for *this* stay date (host's pre-discount quote) | `base_price × seasonal_mult × weekend_mult × host_price_shock` |
| `discount_pct` | float [0, 0.30] | Promotional discount layered on top | Sum of triggers, then clipped to [0, 0.30]: +0.15 if `lead_time ≤ 7`, +0.10 if `los ≥ 7`, +0.10 with p = 0.05 (random property campaign — **the identifying variation for β_discount**) |
| `shown_price` | float (€) | Customer-facing per-night price; **treatment for β_cancel** | `date_base_price × (1 − discount_pct)` |

### 9.4 Outcomes — *from `properties.py:generate_outcomes`*

| column | type | description | explanation |
|---|---|---|---|
| `booking_bool` | binary {0,1} | Did the shopping event convert to a booking? | `Bernoulli(expected_booking_prob)`. The probability is `sigmoid(5 + quality_effect + seasonal_utility + price_utility + N(0, 0.1))`, where `price_utility = β_base · log(date_base_price) + β_discount · discount_pct` |
| `cancelled_bool` | binary {0,1} | Was the booking cancelled? | `Bernoulli(expected_cancel_prob)` if `booking_bool == 1`, else 0. Probability is `sigmoid(−0.5 + 0.15 · log(shown_price) + policy_effect)` where `policy_effect ∈ {flexible: +0.5, moderate: 0, strict: −0.5}` |

### 9.5 DGP truth — **must NOT be used as features in Phase 3 / 4**

These columns expose the latent generative process. They are the "answer key" used by Phase 2 (sanity), Phase 4 (estimator validation), and Phase 5 (regret computation). Any model that conditions on them is leaking ground truth.

| column | type | description | explanation |
|---|---|---|---|
| `beta_base` | float | True per-row booking elasticity to `log(base_price)` | Mapped from `quality_tier`: budget = −1.6, mid = −1.2, premium = −0.8 |
| `beta_discount` | float | True per-row booking elasticity to `discount_pct` (positive — higher discount → higher P(book)) | `−1.5 × beta_base` (equivalently `+1.5 × |beta_base|`); discount framing assumed 1.5× more potent than the equivalent base-price drop |
| `expected_booking_prob` | float [0, 1] | DGP-true `P(book)` for this row | `sigmoid(5 + quality_effect + seasonal_utility + price_utility + N(0, 0.1))` — the latent probability before the Bernoulli draw |
| `expected_cancel_prob` | float [0, 1] | DGP-true `P(cancel \| booked)` for this row | `sigmoid(−0.5 + 0.15 · log(shown_price) + policy_effect)` — the latent probability before the conditional Bernoulli draw |

**Rule of thumb:** before training any Phase-3 model, drop the §9.5 columns from the feature matrix. They are kept in the parquet because Phase 2 (DGP sanity vs. observed rates) and Phase 4 (β̂ vs. true β by tier) both need them.
