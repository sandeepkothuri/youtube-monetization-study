"""
YouTube Channel Monetization Study
Multiple Linear Regression Analysis
Sandeep K | CSULB MS Information Systems
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ── Load Data ───────────────────────────────────────────────────────────────
df = pd.read_csv("/home/claude/projects/youtube-monetization-study/data/youtube_channels.csv")
print(f"Loaded {len(df)} channels\n")

# ── Log-transform revenue for regression (skewed distribution) ─────────────
df['log_revenue'] = np.log1p(df['annual_revenue_usd'])
df['log_subscribers'] = np.log1p(df['subscribers'])
df['log_total_views'] = np.log1p(df['total_views'])

features = [
    'log_subscribers',
    'recent_subscriber_growth_rate',
    'log_total_views',
    'engagement_rate',
    'video_count',
    'upload_frequency_days',
    'years_active',
    'sponsorship_deals_per_year',
]

X = df[features]
y = df['log_revenue']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

model = LinearRegression()
model.fit(X_train_sc, y_train)

y_pred = model.predict(X_test_sc)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(np.expm1(y_test), np.expm1(y_pred))

print("=" * 55)
print("  MULTIPLE LINEAR REGRESSION RESULTS")
print("=" * 55)
print(f"  R² Score (explained variance): {r2:.4f}  ({r2*100:.1f}%)")
print(f"  Mean Absolute Error:           ${mae:,.0f}")
print(f"  Training samples:              {len(X_train)}")
print(f"  Test samples:                  {len(X_test)}")
print()

# ── Feature Coefficients ───────────────────────────────────────────────────
coef_df = pd.DataFrame({
    'Feature': features,
    'Coefficient': model.coef_,
    'Abs_Impact': np.abs(model.coef_)
}).sort_values('Abs_Impact', ascending=False)

print("  FEATURE IMPORTANCE (standardized coefficients):")
print("  " + "-" * 50)
for _, row in coef_df.iterrows():
    direction = "+" if row['Coefficient'] > 0 else "-"
    bar = "█" * int(row['Abs_Impact'] * 10)
    print(f"  {direction} {row['Feature']:<38} {bar}")
print()

# ── Descriptive Stats ──────────────────────────────────────────────────────
print("  TOP EARNER INSIGHTS:")
print("  " + "-" * 50)
top10 = df.nlargest(10, 'annual_revenue_usd')
print(f"  Max revenue:     ${df['annual_revenue_usd'].max():>12,.0f}")
print(f"  Top 10 avg:      ${top10['annual_revenue_usd'].mean():>12,.0f}")
print(f"  Median revenue:  ${df['annual_revenue_usd'].median():>12,.0f}")
print(f"  Channels >$1M:   {(df['annual_revenue_usd'] > 1_000_000).sum()}")
print(f"  Channels >$10M:  {(df['annual_revenue_usd'] > 10_000_000).sum()}")
print()

# ── Category Breakdown ─────────────────────────────────────────────────────
cat_stats = df.groupby('category')['annual_revenue_usd'].agg(['mean', 'median', 'count'])
cat_stats.columns = ['Mean Revenue', 'Median Revenue', 'Channel Count']
cat_stats = cat_stats.sort_values('Mean Revenue', ascending=False)
print("  REVENUE BY CATEGORY (Mean):")
print("  " + "-" * 50)
for cat, row in cat_stats.iterrows():
    print(f"  {cat:<15}  ${row['Mean Revenue']:>10,.0f}   ({int(row['Channel Count'])} channels)")

# ── Save results ───────────────────────────────────────────────────────────
results = {
    'r2_score': round(r2, 4),
    'mae_usd': round(mae, 2),
    'n_channels': len(df),
    'top_earner_usd': round(df['annual_revenue_usd'].max(), 2),
    'median_revenue_usd': round(df['annual_revenue_usd'].median(), 2),
    'most_influential_feature': coef_df.iloc[0]['Feature']
}

import json
with open("/home/claude/projects/youtube-monetization-study/data/model_results.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\n  Most influential factor: {coef_df.iloc[0]['Feature']}")
print("=" * 55)
print("  Analysis complete. Results saved.")
print("=" * 55)
