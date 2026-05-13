"""Property generator for the synthetic vacation-rental DGP.

Produces ~300 properties with attributes correlated to quality_tier, matching the
entity spec in project_overview.md §4 Phase 1 (MyHolidayRentals-listing shape).

Parameters live as module-level constants for now; will be moved to
config/config.yaml once the Phase 1 schema is stable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------- Categorical domains ----------

TIERS = ["budget", "mid", "premium"]
PROPERTY_TYPES = ["apartment", "house", "villa", "cottage", "cabin", "studio"]
HOST_STYLES = ["static", "dynamic", "premium"]
VIEWS = ["sea", "mountain", "garden", "city", "none"]
CANCEL_POLICIES = ["flexible", "moderate", "strict"]
AMENITIES = ["wifi", "pool", "parking", "ac", "kitchen", "washing_machine", "garden", "bbq"]


# ---------- Geography (5 regions / 3 countries) ----------

REGIONS = [
    {"name": "Brittany",  "country": "France", "weight": 0.20, "price_factor": 1.00},
    {"name": "Provence",  "country": "France", "weight": 0.20, "price_factor": 1.15},
    {"name": "Catalonia", "country": "Spain",  "weight": 0.20, "price_factor": 1.05},
    {"name": "Andalusia", "country": "Spain",  "weight": 0.20, "price_factor": 0.95},
    {"name": "Tuscany",   "country": "Italy",  "weight": 0.20, "price_factor": 1.20},
]


# ---------- Mixes ----------

TIER_MIX = {"budget": 0.4, "mid": 0.4, "premium": 0.2}
HOST_STYLE_MIX = {"static": 0.5, "dynamic": 0.3, "premium": 0.2}


# ---------- Property type & size, conditional on tier ----------

PROPERTY_TYPE_P = {
    "budget":  {"apartment": 0.50, "studio": 0.30, "cabin": 0.10, "cottage": 0.10, "house": 0.00, "villa": 0.00},
    "mid":     {"apartment": 0.30, "studio": 0.10, "cabin": 0.10, "cottage": 0.20, "house": 0.25, "villa": 0.05},
    "premium": {"apartment": 0.10, "studio": 0.00, "cabin": 0.00, "cottage": 0.10, "house": 0.40, "villa": 0.40},
}

BEDROOMS_MEAN = {"budget": 1.5, "mid": 2.5, "premium": 3.5}  # Poisson, clipped [1, 8]


# ---------- Amenities, conditional on tier ----------

AMENITY_P = {
    "budget":  {"wifi": 0.85, "pool": 0.05, "parking": 0.40, "ac": 0.30, "kitchen": 0.70, "washing_machine": 0.40, "garden": 0.20, "bbq": 0.10},
    "mid":     {"wifi": 0.95, "pool": 0.20, "parking": 0.70, "ac": 0.60, "kitchen": 0.95, "washing_machine": 0.70, "garden": 0.50, "bbq": 0.40},
    "premium": {"wifi": 1.00, "pool": 0.60, "parking": 0.95, "ac": 0.95, "kitchen": 1.00, "washing_machine": 0.95, "garden": 0.85, "bbq": 0.80},
}


# ---------- Audience flags, conditional on tier ----------

KID_FRIENDLY_P = {"budget": 0.30, "mid": 0.50, "premium": 0.40}
PET_FRIENDLY_P = {"budget": 0.40, "mid": 0.30, "premium": 0.20}


# ---------- View, conditional on tier ----------

VIEW_P = {
    "budget":  {"none": 0.50, "garden": 0.20, "city": 0.20, "mountain": 0.05, "sea": 0.05},
    "mid":     {"none": 0.20, "garden": 0.30, "city": 0.15, "mountain": 0.15, "sea": 0.20},
    "premium": {"none": 0.05, "garden": 0.20, "city": 0.05, "mountain": 0.25, "sea": 0.45},
}


# ---------- Distances (km), conditional on tier ----------

DISTANCES_MEDIAN_KM = {
    "budget":  {"beach": 8.0, "transit": 1.5, "center": 2.0},
    "mid":     {"beach": 5.0, "transit": 1.2, "center": 2.5},
    "premium": {"beach": 2.0, "transit": 2.0, "center": 3.0},
}
DISTANCE_SIGMA = 0.5  # lognormal sigma; median per tier comes from DISTANCES_MEDIAN_KM


# ---------- Reputation, conditional on tier ----------

RATING_NORMAL = {
    "budget":  {"mean": 3.8, "sd": 0.40},
    "mid":     {"mean": 4.2, "sd": 0.30},
    "premium": {"mean": 4.6, "sd": 0.20},
}
REVIEW_COUNT_LOGNORMAL = {
    "budget":  {"mu": 3.0, "sigma": 0.8},
    "mid":     {"mu": 3.5, "sigma": 0.7},
    "premium": {"mu": 4.0, "sigma": 0.6},
}


# ---------- Cancel policy, conditional on tier ----------

CANCEL_POLICY_P = {
    "budget":  {"flexible": 0.40, "moderate": 0.40, "strict": 0.20},
    "mid":     {"flexible": 0.30, "moderate": 0.50, "strict": 0.20},
    "premium": {"flexible": 0.20, "moderate": 0.40, "strict": 0.40},
}


# ---------- Base-price formula ----------

TIER_ANCHOR_EUR = {"budget": 60.0, "mid": 120.0, "premium": 280.0}  # per night, capacity=2, no amenity uplift
PER_EXTRA_GUEST_EUR = 15.0
AMENITY_UPLIFT = {"pool": 0.15, "ac": 0.05, "garden": 0.05, "bbq": 0.03}  # only amenities with material price impact
VIEW_UPLIFT = {"sea": 0.30, "mountain": 0.15, "garden": 0.05, "city": 0.0, "none": 0.0}
BASE_PRICE_NOISE_SIGMA = 0.10  # lognormal noise on final price


# ---------- Public API ----------

def generate_properties(n: int = 300, seed: int = 42) -> pd.DataFrame:
    """Generate a property table for the synthetic DGP.

    Each property's attributes (type, size, amenities, view, distances, rating,
    review_count, cancel_policy, base_price) are drawn correlated with `quality_tier`
    so the entity is internally consistent.
    """
    rng = np.random.default_rng(seed)

    region, country = _draw_geography(rng, n)
    quality_tier = _draw_categorical(rng, TIERS, TIER_MIX, n)
    host_pricing_style = _draw_categorical(rng, HOST_STYLES, HOST_STYLE_MIX, n)

    property_type = _draw_categorical_by_tier(rng, quality_tier, PROPERTY_TYPES, PROPERTY_TYPE_P)
    bedrooms = _draw_poisson_by_tier(rng, quality_tier, BEDROOMS_MEAN, lo=1, hi=8)
    bathrooms = np.clip(np.round(bedrooms / 2 + rng.normal(0, 0.3, size=n)), 1, bedrooms).astype(int)
    capacity = np.clip(bedrooms * 2 + rng.integers(-1, 2, size=n), 1, 16).astype(int)

    # Studios are 1-bedroom by definition; cap their capacity at 2
    studio_mask = property_type == "studio"
    bedrooms[studio_mask] = 1
    bathrooms[studio_mask] = 1
    capacity[studio_mask] = np.minimum(capacity[studio_mask], 2)

    amenities = {
        amen: (rng.uniform(size=n) < np.array([AMENITY_P[t][amen] for t in quality_tier])).astype(int)
        for amen in AMENITIES
    }

    kid_friendly = (rng.uniform(size=n) < np.array([KID_FRIENDLY_P[t] for t in quality_tier])).astype(int)
    pet_friendly = (rng.uniform(size=n) < np.array([PET_FRIENDLY_P[t] for t in quality_tier])).astype(int)

    view = _draw_categorical_by_tier(rng, quality_tier, VIEWS, VIEW_P)

    distance_to_beach_km = _draw_lognormal_median_by_tier(rng, quality_tier, DISTANCES_MEDIAN_KM, "beach")
    distance_to_transit_km = _draw_lognormal_median_by_tier(rng, quality_tier, DISTANCES_MEDIAN_KM, "transit")
    distance_to_center_km = _draw_lognormal_median_by_tier(rng, quality_tier, DISTANCES_MEDIAN_KM, "center")

    rating = np.clip(_draw_normal_by_tier(rng, quality_tier, RATING_NORMAL), 1.0, 5.0).round(1)
    review_count = _draw_lognormal_raw_by_tier(rng, quality_tier, REVIEW_COUNT_LOGNORMAL).astype(int)

    cancel_policy = _draw_categorical_by_tier(rng, quality_tier, CANCEL_POLICIES, CANCEL_POLICY_P)

    base_price = _compute_base_price(rng, quality_tier, capacity, amenities, view, region)

    return pd.DataFrame({
        "property_id": [f"P{i:04d}" for i in range(n)],
        "region": region,
        "country": country,
        "quality_tier": quality_tier,
        "host_pricing_style": host_pricing_style,
        "property_type": property_type,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "capacity": capacity,
        **amenities,
        "kid_friendly": kid_friendly,
        "pet_friendly": pet_friendly,
        "view": view,
        "distance_to_beach_km": np.round(distance_to_beach_km, 2),
        "distance_to_transit_km": np.round(distance_to_transit_km, 2),
        "distance_to_center_km": np.round(distance_to_center_km, 2),
        "rating": rating,
        "review_count": review_count,
        "cancel_policy": cancel_policy,
        "base_price": np.round(base_price, 2),
    })


# ---------- Internal helpers ----------

def _draw_geography(rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray]:
    names = [r["name"] for r in REGIONS]
    weights = np.array([r["weight"] for r in REGIONS])
    weights = weights / weights.sum()
    region = rng.choice(names, size=n, p=weights)
    region_to_country = {r["name"]: r["country"] for r in REGIONS}
    country = np.array([region_to_country[r] for r in region])
    return region, country


def _draw_categorical(
    rng: np.random.Generator,
    categories: list[str],
    p_map: dict[str, float],
    n: int,
) -> np.ndarray:
    p = np.array([p_map[c] for c in categories])
    p = p / p.sum()
    return rng.choice(categories, size=n, p=p)


def _draw_categorical_by_tier(
    rng: np.random.Generator,
    tier: np.ndarray,
    categories: list[str],
    p_by_tier: dict[str, dict[str, float]],
) -> np.ndarray:
    out = np.empty(len(tier), dtype=object)
    for t in TIERS:
        mask = tier == t
        if not mask.any():
            continue
        p = np.array([p_by_tier[t].get(c, 0.0) for c in categories])
        p = p / p.sum()
        out[mask] = rng.choice(categories, size=mask.sum(), p=p)
    return out


def _draw_poisson_by_tier(
    rng: np.random.Generator,
    tier: np.ndarray,
    mean_by_tier: dict[str, float],
    lo: int,
    hi: int,
) -> np.ndarray:
    out = np.zeros(len(tier), dtype=int)
    for t in TIERS:
        mask = tier == t
        if not mask.any():
            continue
        out[mask] = rng.poisson(lam=mean_by_tier[t], size=mask.sum())
    return np.clip(out, lo, hi)


def _draw_normal_by_tier(
    rng: np.random.Generator,
    tier: np.ndarray,
    params_by_tier: dict[str, dict[str, float]],
) -> np.ndarray:
    out = np.zeros(len(tier))
    for t in TIERS:
        mask = tier == t
        if not mask.any():
            continue
        out[mask] = rng.normal(params_by_tier[t]["mean"], params_by_tier[t]["sd"], size=mask.sum())
    return out


def _draw_lognormal_median_by_tier(
    rng: np.random.Generator,
    tier: np.ndarray,
    median_by_tier: dict[str, dict[str, float]],
    key: str,
) -> np.ndarray:
    """Lognormal draws whose per-tier median equals `median_by_tier[tier][key]`.

    median(LN(mu, sigma)) = exp(mu), so mu = log(median).
    """
    out = np.zeros(len(tier))
    for t in TIERS:
        mask = tier == t
        if not mask.any():
            continue
        mu = np.log(median_by_tier[t][key])
        out[mask] = rng.lognormal(mu, DISTANCE_SIGMA, size=mask.sum())
    return out


def _draw_lognormal_raw_by_tier(
    rng: np.random.Generator,
    tier: np.ndarray,
    params_by_tier: dict[str, dict[str, float]],
) -> np.ndarray:
    out = np.zeros(len(tier))
    for t in TIERS:
        mask = tier == t
        if not mask.any():
            continue
        out[mask] = rng.lognormal(params_by_tier[t]["mu"], params_by_tier[t]["sigma"], size=mask.sum())
    return out


def _compute_base_price(
    rng: np.random.Generator,
    quality_tier: np.ndarray,
    capacity: np.ndarray,
    amenities: dict[str, np.ndarray],
    view: np.ndarray,
    region: np.ndarray,
) -> np.ndarray:
    n = len(quality_tier)
    tier_anchor = np.array([TIER_ANCHOR_EUR[t] for t in quality_tier])
    base_pre = tier_anchor + (capacity - 2) * PER_EXTRA_GUEST_EUR

    amenity_uplift = np.ones(n)
    for amen, mult in AMENITY_UPLIFT.items():
        amenity_uplift *= 1 + mult * amenities[amen]

    view_uplift = np.array([1 + VIEW_UPLIFT[v] for v in view])

    region_to_factor = {r["name"]: r["price_factor"] for r in REGIONS}
    region_factor = np.array([region_to_factor[r] for r in region])

    noise = rng.lognormal(mean=0.0, sigma=BASE_PRICE_NOISE_SIGMA, size=n)
    return base_pre * amenity_uplift * view_uplift * region_factor * noise

def generate_outcomes(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """
    Simulates the 'Latent Truth' of customer behavior.
    
    Stage 1: Booking (Bernoulli) - Dependent on price, quality, and seasonality.
    Stage 2: Cancellation (Bernoulli) - Conditional on booking; dependent on price 
             and cancel_policy.
    """
    rng = np.random.default_rng(seed)
    n = len(df)

    # --- 1. Latent Demand Drivers (The 'Signal') ---
    # We define a 'Utility' score. If Utility > 0, the likelihood of booking increases.
    
    # Quality Signal: Better ratings and more bedrooms increase utility
    quality_effect = 0.5 * (df["rating"] - 3.5) + 0.2 * df["bedrooms"]
    
    # Seasonality Signal: Natural demand spikes in summer/weekends
    # Note: This is why we need DML; seasonality drives both price (Phase 1) and utility here.
    seasonal_utility = 1.2 * df["seasonal_mult"] + 0.5 * df["is_weekend"]
    
    # --- 2. The Treatment Effects (Elasticities) ---
    # We use log(price) to represent constant elasticity
    # β_base is heterogeneous by tier (from project_overview §8)
    beta_map = {"budget": -1.6, "mid": -1.2, "premium": -0.8}
    df["beta_base"] = df["quality_tier"].map(beta_map)
    
    # Framing effect: |β_discount| is ~1.5x stronger than |β_base|.
    # Positive sign — a higher discount_pct INCREASES booking probability
    # (β_base is negative because higher log(price) decreases booking).
    df["beta_discount"] = -df["beta_base"] * 1.5
    
    price_utility = (
        df["beta_base"] * np.log(df["date_base_price"]) + 
        df["beta_discount"] * df["discount_pct"]
    )

    # --- 3. Booking Probability (P_book) ---
    # Intercept adjusted to keep aggregate booking rate around 20-25%
    logit_book = 3 + quality_effect + seasonal_utility + price_utility + rng.normal(0, 0.1, n)
    df["expected_booking_prob"] = 1 / (1 + np.exp(-logit_book))
    df["booking_bool"] = (rng.uniform(size=n) < df["expected_booking_prob"]).astype(int)

    # --- 4. Cancellation Probability (P_cancel | Booked) ---
    # Only applies if booking_bool == 1
    # β_cancel is mildly positive: Higher prices lead to higher buyer's remorse
    cancel_beta_price = 0.15 
    
    # Policy effect: Flexible = +5pp cancel rate, Strict = -5pp
    policy_map = {"flexible": 0.5, "moderate": 0.0, "strict": -0.5}
    policy_effect = df["cancel_policy"].map(policy_map)
    
    # Base cancellation logit (centered around ~20%)
    logit_cancel = -2 + (cancel_beta_price * np.log(df["shown_price"])) + policy_effect
    df["expected_cancel_prob"] = 1 / (1 + np.exp(-logit_cancel))
    
    # Draw cancellation only for those who booked
    df["cancelled_bool"] = 0
    booked_mask = df["booking_bool"] == 1
    df.loc[booked_mask, "cancelled_bool"] = (
        rng.uniform(size=booked_mask.sum()) < df.loc[booked_mask, "expected_cancel_prob"]
    ).astype(int)

    return df