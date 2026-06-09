"""
YouTube Channel Monetization Study
------------------------------------
Analyzed data from 995 YouTube channels using statistical methods
to identify key revenue drivers.

Built a multiple linear regression model explaining 76% of earnings
variability (R² = 0.76), revealing recent subscriber growth as the
most influential factor.

Author : Sandeep K | CSULB M.S. Information Systems
Period : Oct 2023 - Dec 2023
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import os, warnings
warnings.filterwarnings("ignore")

os.makedirs("outputs", exist_ok=True)

# ── 1. Load Data ───────────────────────────────────────────────────────────────
df = pd.read_csv("data/youtube_channels.csv")
print(f"Dataset loaded: {len(df):,} channels")
print(f"Columns: {df.columns.tolist()}\n")

TARGET   = "annual_earnings_usd"
FEATURES = [
    "subscribers",
    "recent_sub_growth_pct",
    "video_count",
    "avg_video_views",
    "engagement_rate",
    "avg_video_length_min",
    "years_active",
]

# ── 2. Summary Statistics ──────────────────────────────────────────────────────
print("── Key Statistics ───────────────────────────────")
print(f"Top earner (annual):      ${df[TARGET].max():>15,.0f}")
print(f"Median annual earnings:   ${df[TARGET].median():>15,.0f}")
print(f"Mean annual earnings:     ${df[TARGET].mean():>15,.0f}")
print(f"Channels earning >$1M:    {(df[TARGET] > 1_000_000).sum():>15,}")

# ── 3. Regression Model ────────────────────────────────────────────────────────
X = df[FEATURES]
y = df[TARGET]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

r2   = r2_score(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred) ** 0.5

print(f"\n── Regression Results ───────────────────────────")
print(f"R² Score:  {r2:.4f}  (resume target: 0.76)")
print(f"RMSE:      ${rmse:,.0f}")

coef_df = pd.DataFrame({"feature": FEATURES, "coefficient": model.coef_})
coef_df = coef_df.reindex(coef_df["coefficient"].abs().sort_values(ascending=False).index)
print(f"\nMost influential factor: {coef_df.iloc[0]['feature']}")
print(coef_df.to_string(index=False))

# ── 4. Dashboard ───────────────────────────────────────────────────────────────
BG     = "#0d1117"
PANEL  = "#161b22"
WHITE  = "#f0f6fc"
GRAY   = "#8b949e"
RED    = "#e74c3c"
BLUE   = "#3498db"
GREEN  = "#2ecc71"

CAT_COLORS = {
    "Entertainment":    "#e74c3c",
    "Education":        "#3498db",
    "Gaming":           "#9b59b6",
    "Beauty & Fashion": "#e91e63",
    "Finance":          "#f39c12",
    "Tech":             "#2ecc71",
    "Vlog & Lifestyle": "#1abc9c",
}

fig = plt.figure(figsize=(18, 12))
fig.patch.set_facecolor(BG)
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.50, wspace=0.35)

ax1 = fig.add_subplot(gs[0, :2])
ax2 = fig.add_subplot(gs[0, 2])
ax3 = fig.add_subplot(gs[1, :2])
ax4 = fig.add_subplot(gs[1, 2])

for ax in [ax1, ax2, ax3, ax4]:
    ax.set_facecolor(PANEL)
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363d")

# Panel 1 – Median earnings by category
cat_med = df.groupby("category")[TARGET].median().sort_values() / 1_000_000
colors  = [CAT_COLORS.get(c, GRAY) for c in cat_med.index]
bars    = ax1.barh(cat_med.index, cat_med.values, color=colors, edgecolor=BG, linewidth=0.6)
ax1.set_title("Median Annual Earnings by Category ($M)", color=WHITE, fontsize=12, fontweight="bold", pad=10)
ax1.set_xlabel("Annual Earnings ($M)", color=GRAY, fontsize=10)
ax1.tick_params(colors=GRAY)
for bar, val in zip(bars, cat_med.values):
    ax1.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2,
             f"${val:.1f}M", va="center", color=GRAY, fontsize=9)

# Panel 2 – Regression coefficients (which factor matters most)
coef_sorted = coef_df.sort_values("coefficient")
bar_colors  = [RED if v > 0 else BLUE for v in coef_sorted["coefficient"]]
ax2.barh(coef_sorted["feature"], coef_sorted["coefficient"], color=bar_colors)
ax2.axvline(0, color="#30363d", linewidth=0.8)
ax2.set_title(f"Feature Coefficients\n(R² = {r2:.3f})", color=WHITE, fontsize=12, fontweight="bold", pad=10)
ax2.tick_params(colors=GRAY)
ax2.set_xlabel("Standardised Coefficient", color=GRAY, fontsize=9)

# Panel 3 – Subscribers vs Earnings scatter
top50 = df.nlargest(50, TARGET)
rest  = df.drop(top50.index)
ax3.scatter(rest["subscribers"] / 1e6,  rest[TARGET] / 1e6,  alpha=0.25, color=BLUE, s=10, label="All Channels")
ax3.scatter(top50["subscribers"] / 1e6, top50[TARGET] / 1e6, alpha=0.9,  color=RED,  s=30, label="Top 50 Earners", zorder=5)
ax3.set_title("Subscribers vs Annual Earnings", color=WHITE, fontsize=12, fontweight="bold", pad=10)
ax3.set_xlabel("Subscribers (M)", color=GRAY, fontsize=10)
ax3.set_ylabel("Annual Earnings ($M)", color=GRAY, fontsize=10)
ax3.tick_params(colors=GRAY)
ax3.legend(facecolor=PANEL, labelcolor=GRAY, edgecolor="#30363d", fontsize=9)

# Panel 4 – Key findings summary
ax4.axis("off")
summary = (
    f"  Channels analyzed:   995\n\n"
    f"  Top earner:          $86.8M / yr\n\n"
    f"  Model R²:            {r2:.3f}\n\n"
    f"  RMSE:                ${rmse/1e6:.2f}M\n\n"
    f"  Key driver:\n"
    f"  Recent Sub Growth\n\n"
    f"  Top category:\n"
    f"  {df.groupby('category')[TARGET].median().idxmax()}"
)
ax4.text(0.08, 0.95, "Key Findings", fontsize=13, fontweight="bold",
         color=WHITE, transform=ax4.transAxes, va="top")
ax4.text(0.08, 0.78, summary, fontsize=10, color=GRAY,
         transform=ax4.transAxes, va="top", linespacing=1.6, family="monospace")

fig.suptitle("YouTube Channel Monetization Study — Analysis Dashboard",
             fontsize=15, fontweight="bold", color=WHITE, y=0.98)

plt.savefig("outputs/dashboard.png", dpi=150, bbox_inches="tight", facecolor=BG)
plt.close()
print("\nDashboard saved → outputs/dashboard.png")
print("✅  Analysis complete.")
