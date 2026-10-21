# Gender Labels in the News

Code and data for the master's thesis *Gender Labels in the News: Headline Framing and Audience Responses on Weibo*.

The study compares negative expression, gender language, factual language, and vocabulary across female-gendered, gender-neutral, and male-gendered headlines.

## Repository Contents

### Weibo Scrape

- `weibo_config_template.json`: Configuration template for search keywords, collection dates, and scraper settings.
- `weibo_scraper.py`: Collects Weibo posts and comments and saves them as CSV files.

### Comment_analysis

- `01_comment_cleaning.R`: Cleans raw comments by removing invalid entries, spam, and other platform noise.
- `02_sentiment_analysis.py`: Processes comment text and generates classification labels. The final dataset also incorporates subsequent label review and corrections.
- `03_statistical_analysis.R`: Compares comment indicators across headline groups using chi-squared tests and logistic regression, with regression standard errors clustered by post.
- `04_keyness_analysis.py`: Identifies distinctive vocabulary across headline groups and topics using log-likelihood keyness analysis.
- `05_descriptive_statistics.R`: Produces descriptive tables for posts, comments, and topic distributions.

### Data Files

- `Weibo_post_data_cleaning.csv`: Cleaned post data, including headline groups, topics, and publisher information.
- `comments_sentiment.csv`: Final comment dataset used for analysis, including negative-expression, gender-language, and factual-language labels.

## Recommended Workflow

1. Configure and run the scraper to collect posts and comments.
2. Run `01_comment_cleaning.R` to clean the comments.
3. Run `02_sentiment_analysis.py` and complete the label-review process.
4. Link comments to the corresponding post information using the post ID.
5. Run `03_statistical_analysis.R`, `04_keyness_analysis.py`, and `05_descriptive_statistics.R` using the finalized dataset.

To analyze the supplied final data, start with the three analysis scripts in Step 5. These scripts can be run independently.

## Setup

The scripts use Python and R. Install the packages imported by each script and update input and output paths before running. The scraper also requires the settings specified in its configuration template.

The supplied CSV files contain the finalized research data. Rerunning the initial classification script alone may not reproduce labels that were subsequently reviewed and corrected.
