"""
YouTube Channel Monetization Study - Visualizations
Generates all charts for the project dashboard
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

# Color palette
YOUTUBE_RED = '#FF0000'
DARK_BG = '#1a1a2e'
CARD_BG = '#16213e'
ACCENT = '#e94560'
GOLD = '#f5a623'
TEAL = '#0f3460'
WHITE = '#e8e8e8'
GRAY = '#888888'

df = pd.read_csv("/home/claude/projects/youtube-monetization-study/data/youtube_channels.csv")
df['log_revenue'] = np.log1p(df['annual_revenue_usd'])
df['log_subscribers'] = np.log1p(df['subscribers'])
df['log_total_views'] = np.log1p(df['total_views'])

features = ['log_subscribers', 'recent_subscriber_growth_rate', 'log_total_views',
            'engagement_rate', 'video_count', 'upload_frequency_days',
            'years_active', 'sponsorship_deals_per_year']
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

coef_df = pd.DataFrame({'Feature': features, 'Coefficient': model.coef_}).sort_values('Coefficient', key=abs, ascending=False)

fig = plt.figure(figsize=(18, 14), facecolor=DARK_BG)
fig.suptitle('YouTube Channel Monetization Study\n995 Channels · Multiple Linear Regression Analysis',
             color=WHITE, fontsize=18, fontweight='bold', y=0.98)

gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35,
                       top=0.92, bottom=0.06, left=0.06, right=0.97)

# ── KPI Cards ──────────────────────────────────────────────────────────────
kpis = [
    ("R² Score", f"{r2:.2%}", "variance explained"),
    ("Top Earner", "$36.3M", "annual revenue"),
    ("Channels Analyzed", "995", "unique channels"),
]
for i, (label, val, sub) in enumerate(kpis):
    ax = fig.add_subplot(gs[0, i])
    ax.set_facecolor(CARD_BG)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.axis('off')
    ax.text(0.5, 0.72, val, ha='center', va='center', color=ACCENT,
            fontsize=26, fontweight='bold')
    ax.text(0.5, 0.42, label, ha='center', va='center', color=WHITE,
            fontsize=11, fontweight='semibold')
    ax.text(0.5, 0.22, sub, ha='center', va='center', color=GRAY, fontsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor(ACCENT); spine.set_linewidth(1.5)

# ── Feature Importance ─────────────────────────────────────────────────────
ax2 = fig.add_subplot(gs[1, :2])
ax2.set_facecolor(CARD_BG)
clean_names = {
    'log_total_views': 'Total Views (log)',
    'log_subscribers': 'Subscribers (log)',
    'sponsorship_deals_per_year': 'Sponsorship Deals/yr',
    'recent_subscriber_growth_rate': 'Recent Sub Growth ★',
    'years_active': 'Years Active',
    'video_count': 'Video Count',
    'upload_frequency_days': 'Upload Frequency',
    'engagement_rate': 'Engagement Rate'
}
coef_df['Clean'] = coef_df['Feature'].map(clean_names)
colors = [ACCENT if c > 0 else '#4a9eff' for c in coef_df['Coefficient']]
bars = ax2.barh(coef_df['Clean'], coef_df['Coefficient'], color=colors, edgecolor='none', height=0.65)
ax2.axvline(0, color=WHITE, lw=0.8, alpha=0.4)
ax2.set_title('Feature Coefficients (Standardized)', color=WHITE, fontsize=12, pad=8)
ax2.set_xlabel('Standardized Coefficient', color=GRAY, fontsize=9)
ax2.tick_params(colors=WHITE, labelsize=9)
ax2.spines['bottom'].set_color(GRAY); ax2.spines['left'].set_color(GRAY)
ax2.spines['top'].set_visible(False); ax2.spines['right'].set_visible(False)
ax2.set_facecolor(CARD_BG)
for bar, val in zip(bars, coef_df['Coefficient']):
    ax2.text(val + (0.005 if val > 0 else -0.005), bar.get_y() + bar.get_height()/2,
             f'{val:.3f}', va='center', ha='left' if val > 0 else 'right', color=WHITE, fontsize=8)

# ── Actual vs Predicted ────────────────────────────────────────────────────
ax3 = fig.add_subplot(gs[1, 2])
ax3.set_facecolor(CARD_BG)
ax3.scatter(np.expm1(y_test)/1e6, np.expm1(y_pred)/1e6,
            alpha=0.35, s=18, color=ACCENT, edgecolors='none')
max_val = max(np.expm1(y_test).max(), np.expm1(y_pred).max()) / 1e6
ax3.plot([0, max_val], [0, max_val], color=GOLD, lw=1.5, linestyle='--', label='Perfect fit')
ax3.set_title(f'Actual vs Predicted Revenue\nR² = {r2:.3f}', color=WHITE, fontsize=11, pad=8)
ax3.set_xlabel('Actual Revenue ($M)', color=GRAY, fontsize=9)
ax3.set_ylabel('Predicted Revenue ($M)', color=GRAY, fontsize=9)
ax3.tick_params(colors=WHITE, labelsize=8)
for s in ['top','right']: ax3.spines[s].set_visible(False)
for s in ['bottom','left']: ax3.spines[s].set_color(GRAY)
ax3.legend(fontsize=8, labelcolor=WHITE, facecolor=CARD_BG, edgecolor=GRAY)

# ── Revenue Distribution ───────────────────────────────────────────────────
ax4 = fig.add_subplot(gs[2, 0])
ax4.set_facecolor(CARD_BG)
ax4.hist(df['annual_revenue_usd'] / 1e6, bins=50, color=ACCENT, edgecolor='none', alpha=0.8)
ax4.set_title('Revenue Distribution', color=WHITE, fontsize=11, pad=8)
ax4.set_xlabel('Annual Revenue ($M)', color=GRAY, fontsize=9)
ax4.set_ylabel('Channel Count', color=GRAY, fontsize=9)
ax4.tick_params(colors=WHITE, labelsize=8)
for s in ['top','right']: ax4.spines[s].set_visible(False)
for s in ['bottom','left']: ax4.spines[s].set_color(GRAY)
ax4.axvline(df['annual_revenue_usd'].median()/1e6, color=GOLD, lw=1.5,
            linestyle='--', label=f"Median ${df['annual_revenue_usd'].median()/1e6:.2f}M")
ax4.legend(fontsize=8, labelcolor=WHITE, facecolor=CARD_BG, edgecolor=GRAY)

# ── Revenue by Category ────────────────────────────────────────────────────
ax5 = fig.add_subplot(gs[2, 1])
ax5.set_facecolor(CARD_BG)
cat_mean = df.groupby('category')['annual_revenue_usd'].mean().sort_values(ascending=True) / 1e6
cat_colors = [ACCENT if v == cat_mean.max() else '#4a9eff' for v in cat_mean.values]
ax5.barh(cat_mean.index, cat_mean.values, color=cat_colors, height=0.65, edgecolor='none')
ax5.set_title('Avg Revenue by Category', color=WHITE, fontsize=11, pad=8)
ax5.set_xlabel('Avg Annual Revenue ($M)', color=GRAY, fontsize=9)
ax5.tick_params(colors=WHITE, labelsize=8)
for s in ['top','right']: ax5.spines[s].set_visible(False)
for s in ['bottom','left']: ax5.spines[s].set_color(GRAY)

# ── Subscribers vs Revenue ─────────────────────────────────────────────────
ax6 = fig.add_subplot(gs[2, 2])
ax6.set_facecolor(CARD_BG)
scatter = ax6.scatter(df['subscribers']/1e6, df['annual_revenue_usd']/1e6,
                      c=df['recent_subscriber_growth_rate'], cmap='RdYlGn',
                      alpha=0.4, s=12, edgecolors='none')
cbar = plt.colorbar(scatter, ax=ax6, fraction=0.035, pad=0.04)
cbar.set_label('Sub Growth Rate', color=WHITE, fontsize=8)
cbar.ax.tick_params(colors=WHITE, labelsize=7)
ax6.set_title('Subscribers vs Revenue\n(color = growth rate)', color=WHITE, fontsize=11, pad=8)
ax6.set_xlabel('Subscribers (M)', color=GRAY, fontsize=9)
ax6.set_ylabel('Revenue ($M)', color=GRAY, fontsize=9)
ax6.tick_params(colors=WHITE, labelsize=8)
for s in ['top','right']: ax6.spines[s].set_visible(False)
for s in ['bottom','left']: ax6.spines[s].set_color(GRAY)

plt.savefig("/home/claude/projects/youtube-monetization-study/visualizations/dashboard.png",
            dpi=150, bbox_inches='tight', facecolor=DARK_BG)
print("Dashboard saved.")
