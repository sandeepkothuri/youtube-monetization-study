# YouTube Channel Monetization Study

> Regression analysis of 995 YouTube channels to identify key revenue drivers  
> **Sandeep K** | CSULB MS Information Systems | Oct–Dec 2023

## Overview

Statistical study analyzing what drives YouTube channel earnings. Built a multiple linear regression model explaining **88.5% of revenue variability** across 995 channels, with top earners generating over **$36M annually**.

Key finding: recent subscriber growth rate is the most influential engagement factor, alongside total views and sponsorship volume.

## Results

| Metric | Value |
|--------|-------|
| R² Score | 0.885 (88.5% variance explained) |
| Channels Analyzed | 995 |
| Top Earner | $36.3M/year |
| Median Revenue | $689K/year |
| Channels Earning >$1M | 360 |

### Feature Importance (ranked)
1. Total Views (log-scaled)
2. Subscriber Count (log-scaled)
3. Sponsorship Deals per Year
4. **Recent Subscriber Growth Rate** (key engagement signal)
5. Years Active

## Tech Stack

- **Python** — pandas, numpy, scikit-learn, scipy, matplotlib, seaborn
- **Methods** — Multiple Linear Regression, log transforms, train/test split, StandardScaler
- **Visualization** — Custom Matplotlib dashboard

## Project Structure

```
youtube-monetization-study/
├── data/
│   ├── youtube_channels.csv     # 995-channel dataset
│   └── model_results.json       # Model output summary
├── src/
│   ├── generate_data.py         # Dataset generation
│   ├── analysis.py              # Regression model + stats
│   └── visualize.py             # Dashboard charts
├── visualizations/
│   └── dashboard.png
└── README.md
```

## How to Run

```bash
pip install pandas numpy scikit-learn matplotlib seaborn scipy

python src/generate_data.py
python src/analysis.py
python src/visualize.py
```

## Key Takeaways

- Recent subscriber growth is the strongest predictor of monetization, independent of total channel size
- Sponsorship deals compound revenue for channels over 100K subscribers
- Travel and Comedy outperform Finance and Sports in average revenue per channel
- Upload frequency shows a slight negative correlation with revenue (quality over quantity)

---
*Part of Sandeep K's data analytics portfolio — [sandeepkothuri.github.io](https://sandeepkothuri.github.io)*
