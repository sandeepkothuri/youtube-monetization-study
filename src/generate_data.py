import pandas as pd
import numpy as np

np.random.seed(42)
n = 995

categories = ["Music", "Gaming", "Education", "Tech", "Lifestyle", "Comedy", "Finance", "Sports", "Health", "Travel"]
countries = ["US", "UK", "India", "Brazil", "Canada", "Australia", "Germany", "France", "Japan", "Mexico"]

subscribers = np.random.lognormal(mean=13.5, sigma=1.8, size=n).astype(int)
subscribers = np.clip(subscribers, 1000, 50_000_000)

recent_sub_growth = np.random.uniform(0.01, 0.45, size=n)
total_views = (subscribers * np.random.uniform(15, 120, size=n)).astype(int)
avg_video_views = (total_views / np.random.randint(10, 500, size=n)).astype(int)
upload_frequency = np.random.choice([1, 2, 3, 4, 5, 7, 14, 30], size=n)
engagement_rate = np.random.uniform(0.01, 0.12, size=n)
video_count = np.random.randint(10, 1500, size=n)
years_active = np.random.uniform(0.5, 12, size=n)
category = np.random.choice(categories, size=n)
country = np.random.choice(countries, size=n)
sponsorship_deals = (subscribers > 100_000).astype(int) * np.random.randint(0, 12, size=n)
merchandise_sales = (subscribers > 500_000).astype(int) * np.random.uniform(0, 500_000, size=n)

ad_revenue = (total_views / 1000) * np.random.uniform(1.5, 5.5, size=n)
sponsorship_revenue = sponsorship_deals * np.random.uniform(2000, 50000, size=n)
merch_revenue = merchandise_sales * np.random.uniform(0.1, 0.4, size=n)
other_revenue = np.random.uniform(0, 50000, size=n)

growth_multiplier = 1 + (recent_sub_growth * 3.5)
annual_revenue = (ad_revenue + sponsorship_revenue + merch_revenue + other_revenue) * growth_multiplier
annual_revenue = np.clip(annual_revenue, 1000, 86_800_000)

df = pd.DataFrame({
    "channel_id": [f"UC{i:08d}" for i in range(1, n+1)],
    "category": category,
    "country": country,
    "subscribers": subscribers,
    "recent_subscriber_growth_rate": recent_sub_growth.round(4),
    "total_views": total_views,
    "avg_video_views": avg_video_views,
    "video_count": video_count,
    "upload_frequency_days": upload_frequency,
    "engagement_rate": engagement_rate.round(4),
    "years_active": years_active.round(2),
    "sponsorship_deals_per_year": sponsorship_deals,
    "merchandise_revenue_usd": merchandise_sales.round(2),
    "annual_revenue_usd": annual_revenue.round(2)
})

df.to_csv("/home/claude/projects/youtube-monetization-study/data/youtube_channels.csv", index=False)
print(f"Dataset created: {len(df)} channels")
print(f"Top earner: ${df['annual_revenue_usd'].max():,.0f}")
print(f"Median revenue: ${df['annual_revenue_usd'].median():,.0f}")
