"""
Phase 1: Synthetic Data Generation Pipeline

This script orchestrates the creation of a vacation rental marketplace. 
It transforms static property definitions into a longitudinal 'shopping event' 
dataset. 

Key Components:
1. Entity Generation: Creates ~300 unique property listings.
2. Temporal Expansion: Generates a 4-year calendar for every property.
3. Pricing Treatment: Implements a host pricing heuristic that introduces
   realistic confounding (price correlates with season/weekend).

The resulting 'synthetic_demand_events.parquet' serves as the foundation for
Phase 3 (Predictive Modeling) and Phase 4 (Causal Inference).
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

# Local import from the project structure
from src.components.synthetic.properties import generate_properties

# --- Configuration Constants ---
# These define the 'God Mode' parameters of our synthetic universe
N_PROPERTIES = 300
START_DATE = "2024-01-01"
END_DATE = "2027-12-31"  # 3 years of historical training + 1 year of forecast horizon
OUTPUT_DIR = Path("data/synthetic")
SEED = 42

def generate_calendar_events(properties_df: pd.DataFrame) -> pd.DataFrame:
    """
    Expands the 1D property table into a 2D Time-Series (Panel Data).
    
    This simulates the 'Context' of a booking request: who is looking, 
    how far in advance (lead time), and for how long (LOS).
    """
    rng = np.random.default_rng(SEED)
    
    # 1. Create a continuous date range for the entire study period
    dates = pd.date_range(start=START_DATE, end=END_DATE, freq="D")
    
    # 2. Perform a Cartesian Product (Cross Join)
    # This maps every property to every date, creating the 'Long' format 
    # required for demand forecasting models.
    calendar = properties_df.assign(key=1).merge(
        pd.DataFrame({"stay_date": dates, "key": 1}), on="key"
    ).drop("key", axis=1)

    # 3. Simulate Search Context: Lead Time
    # We use an exponential distribution because most searches happen 
    # close to the stay date, with a long tail of 'early birds'.
    calendar["lead_time"] = rng.exponential(scale=30, size=len(calendar)).astype(int)
    calendar["lead_time"] = np.clip(calendar["lead_time"], 0, 180)
    
    # 4. Simulate Search Context: Length of Stay (LOS)
    # Real-world listings have 'modal' behaviors (e.g., city studios = 2 nights, 
    # coastal villas = 7 nights). We map a mode to each property ID.
    prop_los_mode = rng.choice([2, 3, 7], size=len(properties_df), p=[0.3, 0.4, 0.3])
    los_map = dict(zip(properties_df["property_id"], prop_los_mode))
    
    calendar["los"] = calendar["property_id"].map(los_map)
    # Add idiosyncratic variance to LOS so not every stay for Prop A is identical
    calendar["los"] = np.clip(calendar["los"] + rng.integers(-1, 2, size=len(calendar)), 1, 14)

    # 5. Extract Time-Based Features (The 'Demand Drivers')
    calendar["month"] = calendar["stay_date"].dt.month
    calendar["day_of_week"] = calendar["stay_date"].dt.dayofweek
    # Weekend flag: Friday, Saturday, Sunday (Common for vacation rentals)
    calendar["is_weekend"] = calendar["day_of_week"].isin([4, 5, 6]).astype(int)
    
    return calendar

def apply_host_pricing_logic(df: pd.DataFrame) -> pd.DataFrame:
    """
    Simulates the 'Treatment' (Price) as a function of 'Context' (Season, Weekend).
    
    In observational data, hosts raise prices when demand is high. 
    This function bakes in that 'Confounding' so that we can later test if 
    Double ML can unmix the price effect from the seasonal effect.
    """
    rng = np.random.default_rng(SEED)
    
    # A. Seasonality Multiplier
    # A cosine wave with a 12-month period, centered to peak in July (month 7).
    # This simulates higher 'Base Prices' during summer peak seasons.
    df["seasonal_mult"] = 1 + 0.3 * np.cos(2 * np.pi * (df["month"] - 7) / 12)
    
    # B. Weekend Premium
    # Hosts typically apply a fixed markup for high-demand weekend stay dates.
    df["weekend_mult"] = 1 + (df["is_weekend"] * 0.15)
    
    # C. Dynamic Pricing Noise (Exogenous Variation)
    # Crucial for Causal Inference: This represents the 'Host's Whim'.
    # It ensures price isn't 100% predictable by the model, providing the 
    # variation needed to identify the price elasticity.
    df["host_price_shock"] = rng.normal(1.0, 0.05, size=len(df))
    
    # Calculate the un-discounted price for this specific stay event
    df["date_base_price"] = (
        df["base_price"] * df["seasonal_mult"] * df["weekend_mult"] * df["host_price_shock"]
    )

    # D. Promotional Discount Layering
    # Discounts are 'framed' differently than base price changes.
    df["discount_pct"] = 0.0
    
    # Trigger 1: Last Minute (Urgency framing)
    df.loc[df["lead_time"] <= 7, "discount_pct"] += 0.15
    # Trigger 2: Long Stay (Bulk volume framing)
    df.loc[df["los"] >= 7, "discount_pct"] += 0.10
    # Trigger 3: Random Campaigns (Property-level marketing)
    promo_mask = rng.uniform(size=len(df)) < 0.05
    df.loc[promo_mask, "discount_pct"] += 0.10
    
    # Enforce realistic bounds (max 30% off)
    df["discount_pct"] = df["discount_pct"].clip(0, 0.30)
    
    # The 'Shown Price' is what the customer actually sees on the website
    df["shown_price"] = df["date_base_price"] * (1 - df["discount_pct"])
    
    return df

def main():
    """
    Main execution loop.
    Steps: Generate static entities -> Expand to calendar -> Apply pricing.
    """
    print(f"--- Launching Phase 1: Synthetic DGP ---")
    
    # Ensure directory exists before writing
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Generate Static Properties (The 'Registry')
    print(f"Generating {N_PROPERTIES} properties...")
    props = generate_properties(n=N_PROPERTIES, seed=SEED)
    props.to_parquet(OUTPUT_DIR / "static_properties.parquet")
    
    # 2. Generate Search/Shopping Context (The 'Environment')
    print(f"Expanding to calendar ({START_DATE} to {END_DATE})...")
    events = generate_calendar_events(props)
    
    # 3. Apply Host Pricing Logic (The 'Treatment')
    print("Applying host pricing logic...")
    events = apply_host_pricing_logic(events)

    # 4. Generate Bookings & Cancellations (The 'Outcome')
    # This is where we call the new function from properties.py
    print("Simulating customer behavior (Bookings & Cancellations)...")
    from src.components.synthetic.properties import generate_outcomes
    events = generate_outcomes(events, seed=SEED)
    
    # Save the final dataset
    # We use Parquet for efficient storage of high-row-count data.
    events.to_parquet(OUTPUT_DIR / "synthetic_demand_events.parquet")
    
    print(f"Success. Dataset saved to {OUTPUT_DIR}")
    print(f"Total Rows Generated: {len(events):,}")

if __name__ == "__main__":
    main()