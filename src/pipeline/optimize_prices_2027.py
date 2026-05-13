"""Phase 5 — joint (base_price, discount_pct) optimization for 2027 stays.

For each (property x stay_date in 2027), solves
    argmax  nights * shown_price * P(book | base, disc, ctx) * (1 - P(cancel | shown_price, ctx))
under the locked bounds, using the Phase 3 models and Phase 4 elasticities.

To be implemented.
"""
