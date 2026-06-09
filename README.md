# 📊 YouTube Channel Monetization Study

> Identifying key revenue drivers across 995 YouTube channels using multiple linear regression.

![Dashboard](assets/dashboard.png)

## Overview

This study analyzes a dataset of **995 YouTube channels** to uncover what actually drives earnings on the platform. Using statistical modeling and data visualization, the project quantifies the relationship between channel metrics and monthly revenue, revealing that **real-time audience engagement** (recent subscriber growth) is the single most influential predictor of monetization success.

**Key finding:** The regression model explains **76% of earnings variability (R² = 0.76)**, with top earners grossing over **$86.8M annually**.

---

## Project Structure

```
youtube-monetization-study/
├── analysis.py              # Main analysis script (EDA + regression)
├── generate_charts.py       # Dashboard generation script
├── requirements.txt
├── assets/
│   └── dashboard.png        # Auto-generated analysis dashboard
├── data/
│   └── youtube_channels.csv
└── README.md
```

---

## Features Analyzed

| Feature | Description |
|---|---|
| `subscribers` | Total subscriber count |
| `recent_subscriber_growth_rate` | % subscriber growth in last 90 days |
| `avg_video_views` | Average views per upload |
| `video_count` | Total videos on channel |
| `upload_frequency_days` | Days between uploads |
| `engagement_rate` | (Likes + Comments) / Views |

---

## Results

| Metric | Value |
|---|---|
| R² (test set) | **0.76** |
| Top Earner (annual) | **$86.8M+** |
| Most Influential Factor | **Recent Subscriber Growth** |
| Dataset Size | **995 channels** |

### Key Insights

1. **Recent subscriber growth** is the strongest predictor of earnings, outperforming raw subscriber count.
2. **Engagement rate** matters more than total views for high-earning channels.
3. The top 1% of channels earn significantly disproportionate revenue.
4. Video length has minimal independent impact once engagement is accounted for.

---

## Setup & Run

```bash
git clone https://github.com/sandeepkothuri/youtube-monetization-study.git
cd youtube-monetization-study
pip install -r requirements.txt
python analysis.py
```

---

## Tech Stack

`Python` `pandas` `numpy` `scikit-learn` `matplotlib` `seaborn`

---

## Author

**Sandeep K** · [LinkedIn](https://www.linkedin.com/in/sandeep-kothuri-9b99142b6/) · [GitHub](https://github.com/sandeepkothuri) · [Portfolio](https://sandeepkothuri.github.io)

*CSULB – M.S. Information Systems | Oct–Dec 2023*
