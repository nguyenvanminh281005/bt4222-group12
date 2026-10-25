task 

Feature Dictionary & Descriptive Statistics (TAN & MINH)

Provide structured tables summarizing features you plan to utilize or engineer. For example,

Feature Metadata: Feature name, description, and data type/unit (e.g., seconds, word count, categorical ID).
Numeric Features: Min, max, mean, median, standard deviation, and missing value counts.
Categorical Features: Distinct category counts, representative values, and missing value counts.
Note: If the dataset contains over 50 features, group them by category/domain and report representative features for each category.
Text / Corpus / Media (if applicable): Distribution of sequence/text lengths, vocabulary sizes, or image metadata.

1.Business ML problem

Best performing articles to get “Spotlight slots” on website to encourage higher CTR
Home webpage has limited space, we need to find the best articles to present on this page to get user to click 
When to rotate article based on timing
→ Optimizing the topK problem 
→ Time period (presumption)

Personalisation payoff evaluation, how much user history is needed before personalisation beats popularity
Store large amounts of user history leads to additional costs and more complex/less transparent models → How much better does a personalised model works over a generic model that focuses on overall article popularity
Look at users with a rich history, what happens if we only care about their recent X articles, will model still perform the same (especially for users with over 500 articles in their history)
Evaluate so we stick to simple model of just popular articles, or complex model for full user personalisation, or somewhere in between

→ Data Analysis 
 
Sample project: Making better personalised impression for each user


2.Other Business ML Problem

Business objective: Identify which patterns in article titles and abstracts drive genuine reader engagement (not just clicks), so that editorial/headline-writing guidelines and the recommender's ranking function can be tuned to reward substantive engagement over clickbait → improving long-term retention rather than short-term CTR.
CTR prediction using title/abstract text as features

Possible Approach: Shortlist a category (eg Sports, Food, Fashion…) with users that are historically interested in them. Compare the articles that were part of a longer session vs click-and-leave articles? 

→ Session record : 8am in 2nd Sep → 8:01am 2nd Sep → 8am in 2nd Sep 9am in 2nd Sep 

3.Business problem: turn casual users (only click on one article) to avid users (users to click to multiple articles)
Group the data in categories Sport (which headlines result in more clicks)
Editorial headlines → How to improve them
Temporal → Our system recommends certain articles based on timing 

By improving editorial headlines + have analysing temporal trends, we can hook users to click and interact with more articles

Business ML problem: Breaking Filter Bubbles via Diversity Re-Ranking 
Echo chamber mitigation to boost long-term retention:
Recommenders optimized purely for short-term CTR trap users in single-topic feedback loops (filter bubbles).
Over-exposure to identical categories causes reader fatigue and increases churn → How to trade off marginal click probability to surface diverse topics and keep users retained.
Fixing user interest calibration (Primary vs. Secondary interests):
Browsing histories are heavily imbalanced (e.g., 70% sports, 30% finance).
Standard rankers over-amplify dominant topics and extinguish secondary interests entirely over time.
Re-rank recommendation slates so final suggestions proportionally reflect a user's multi-faceted interest profile rather than a single obsession.
Optimal diversity trade-off evaluation  Sweet Spot):
How much ranking relevance (NDCG/AUC) is lost when enforcing semantic and category diversity (ILD/Coverage)?
Find the optimal balance point () where the slate gains significant topic variety without hurting click engagement.



RQ1 : Spotlight (top-k problem) → model-related problem 
RQ3: Balance the model between rich user history and cold-start or near that (is there any significant degradation in the performance of the model. If yes so how much history is good enough ) (model itself)
RQ5: To recommend documents that are similar to user profile but diverse in topics. Model-related problem
