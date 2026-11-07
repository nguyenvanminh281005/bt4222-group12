## **2.3 Feature Dictionary & Descriptive Statistics**

Following our empirical findings on slate distributions, extreme popularity skew, feedback sparsity, and diurnal cycles (Section 2.2), this section establishes the comprehensive feature space engineered to power our recommendation pipeline. Rather than accumulating an unmanageable quantity of marginal features, we deliberately engineered a streamlined set of core features organized into four business-aligned domains:

1. Content Semantics: Dense text representations, knowledge graph grounding, and categorical taxonomy to solve article cold-start.  
2. Freshness & Popularity Dynamics: Continuous recency decay and historical engagement conversion to balance viral content against breaking news.  
3. User Reading Profile & State: Sequential reading embeddings, topical diversity entropy, category calibration, and cold-start fallback gates.  
4. Temporal & Circadian Dynamics: Hour-of-day clock cycles, day-of-week context, and inactivity gaps governing dynamic spotlight slate rotation.

All descriptive statistics below are computed across the complete merged dataset combining train, validation and test splits of MIND-small (230,117 impression sessions, 94,057 unique users, 65,238 unique articles, and 8,584,442 total exposures). Crucially, in strict adherence to data hygiene protocols, all features are computed using only interactions preceding session timestamp t, preventing future data leakage into validation and test evaluation.

### **2.3.1 Feature Metadata Dictionary**

Table 4 details the metadata for all engineered features, including their description, data type, representation dimension/unit, and source table.

| Feature Name | Feature Type | Domain | Data Type / Unit | Source Table |
| :---- | :---- | :---- | :---- | :---- |
| article\_content\_emb | Text / Corpus & Embedding | Content Semantics | Dense Vector (384) | news.tsv |
| **Description** | An article embedding vector representing the semantic meaning of the article. |  |  |  |
| **Justification** | Article content embeddings capture the semantic content of news articles and enable the model to identify topical relevance between readers and candidate stories. This supports the business objective of maximising spotlight engagement by promoting articles that align with audience interests, even when limited historical interaction data exists. |  |  |  |
|  |  |  |  |  |
| kg\_entity\_emb | Knowledge Graph & Embedding | Content Semantics | Dense Vector (100) | news.tsv & entity\_embedding.vec |
| **Description** | Average 100-dimensional TransE entity embeddings of linked Wikidata knowledge graph entities identified in the title and abstract. |  |  |  |
| **Justification** | Knowledge graph entity embeddings encode relationships among people, organisations, locations, and events mentioned within articles. These representations help identify relevant breaking stories and support the timely promotion of fresh content, reducing reliance on historical popularity signals alone. |  |  |  |
|  |  |  |  |  |
| category\_id | Categorical | Content Semantics | 18 categories / 16-dim embedding | news.tsv |
| **Description** | Top-level editorial category of an article. |  |  |  |
| **Justification** | Article category information enables the model to capture category-specific engagement patterns and diurnal consumption trends. This is particularly important for dynamic spotlight rotation, as user preferences may shift across time periods, such as increased consumption of business news during daytime hours and sports or entertainment content during evening periods. |  |  |  |
|  |  |  |  |  |
| subcategory\_id | Categorical | Content Semantics | 270 categories / 32-dim embedding | news.tsv |
| **Description** | Fine-grained editorial subcategory of an article. |  |  |  |
| **Justification** | Captures more specific reader interests within broad editorial categories. |  |  |  |
|  |  |  |  |  |
| article\_recency | Numeric | Freshness & Popularity | Continuous (hours) | behaviors.tsv |
| **Description** | Elapsed time (in hours) between an article's initial publication/appearance in logs and the current impression session timestamp t. |  |  |  |
| **Justification** | *article\_recency* supports freshness-popularity balance and dynamic temporal rotation as it provides an anti-stagnation mechanism that allows fresher news to overcome the initial click deficit against the older and established articles, ensuring that the homepage recommendations remain dynamic and fresh.  The time that a news article is recommended on the spotlight slot is critical to its engagement as user engagement is heavily concentrated during morning surges between 8–9am (Fig.10). *article\_recency* ensures that high-traffic morning windows are populated with newly published morning news rather than decaying news from the previous day.  |  |  |  |
|  |  |  |  |  |
| historical\_ctr | Numeric | Freshness & Popularity | Continuous \[0, 1\] | behaviors.tsv |
| **Description** | Empirical click-through rate at the article exposure level, calculated strictly from historical exposures prior to session timestamp t. |  |  |  |
| **Justification** | Article clicks follow an extreme Pareto law (1% of articles drive 54.3% of clicks). Historical CTR provides a strong engagement baseline. However, as 60% of articles have zero clicks, it must be paired with recency to avoid permanently penalizing newly published articles. |  |  |  |
|  |  |  |  |  |
| user\_history\_emb | User Profile & Embedding | User Profile | Dense Vector (384) | behaviors.tsv & news.tsv |
| **Description** | Multi-head attention-weighted average of content embeddings from the user's previously clicked articles prior to session timestamp t. |  |  |  |
| **Justification** | *user\_history\_emb* allows recommendations to move beyond generic global popularity by tailoring spotlight candidates to an individual reader's specific preferences. *user\_history\_emb* represents the user reading history *u* in dense vector space and is used to compute similarity against candidate article content embedding (article\_content\_emb) to better model *P(click \= 1 \| u, d, t)* for recommendations. *user\_history\_emb* can help to address sparsity issues. As distribution of user click history length is heavily right-skewed (Fig.8) with 17.91% of users having fewer than 5 clicked articles and 2.26% having zero history, user\_history\_emb helps to map sparse user's few clicked articles into continuous dense vectors to generalize across shared topic spaces even with minimal interaction history. *user\_history\_emb* also helps to surface the single most engaging article to the top of an impression slate. As impression slates are large (mean of 37.3 articles per slate, max 299), a single impression generally contains multiple globally popular news articles. However, as 72.1% of impressions yield only a single click, users are highly selective and will ignore the remaining 36+ recommendations. Hence, selecting the single top recommendation using global popularity alone is insufficient as it assigns uniformly high scores to generic viral articles without any individual signal to break ties. Matching candidate article content embeddings (*article\_content\_emb*) against *user\_history\_emb* provides the granular and individual signal required to surface the single most engaging article to the top of a 37-article slate. |  |  |  |
|  |  |  |  |  |
| category\_affinity | Continuous Vector | User Profile & Interaction | Continuous Vector (18)  | behaviors.tsv & news.tsv |
| **Description** | Normalized 18-dim frequency vector of historical category reads (category\_affinity) |  |  |  |
| **Justification** | *category\_affinity* captures how strongly each user prefers out of the different topics, so Spotlight slots can be personalized instead of showing the same popular articles to everyone. An article’s category is known at publish time, so new articles with no click history can still be scored for relevance.  |  |  |  |
|  |  |  |  |  |
| category\_match | Binary categorical | User Reading Profile | Binary {0, 1} | behaviors.tsv, news.tsv |
| **Description** | Binary flag indicating match with user's top category. Indicates whether a candidate belongs to the user's historically most-read category. |  |  |  |
| **Justification** | *category\_match* flags whether a candidate article is the user’s most read category, which is a strong predictor of a click.  As it only uses article category, it works for newly published articles with no click history which supports cold-start ranking. However, when used alone, it risks a filter bubble so it should be paired with *reading\_entropy* to control how strongly it is applied per user. |  |  |  |
|  |  |  |  |  |
| reading\_entropy | Numeric | User Profile | Continuous \[0, ln(18) \~ 2.89\] | behaviors.tsv & news.tsv |
| **Description** | Shannon entropy of user's historical category distribution. It measures topical breadth vs. specialization. |  |  |  |
| **Justification**  | Measures how broadly a user reads, which tells the model how much to trust category signals. Users with lower entropy should be recommended articles that are aligned with their preferred topics, while users with higher entropy respond better to diverse articles with more fresh content. |  |  |  |
|  |  |  |  |  |
| history\_length | Numeric | User Reading Profile | Integer count | behaviors.tsv |
| **Description** | Number of historical clicked articles available for the user prior to the current impression. |  |  |  |
| **Justification** | *history\_length* indicates how reliable the user’s profile is. Features such as *category\_affinity* and *reading\_entropy* built from a small number of clicks are noisy, as compared to features being built from many clicks.  This allows the model to weigh personalisation strongly for users with richer history and lean on general article popularity for lighter users. Conversely, this feature is also able to navigate the model when to handle new cold-start users. |  |  |  |
|  |  |  |  |  |
| has\_history | Binary categorical | User Reading Profile | Binary {0, 1} | behaviors.tsv |
| **Description** | Indicates whether the user has at least one historical clicked article. |  |  |  |
| **Justification** | For new users, the category features are all zero. By clearly indicating whether the user has history, it prevents the model from misreading it as a real preference signal. *has\_history* tells the model to ignore personalisation signals for such users with no history and fall back on general article popularity and freshness. |  |  |  |
|  |  |  |  |  |
| session\_hour | Temporal / Categorical | Temporal Dynamics | Integer / Cyclic \[0, 23\] (16-emb) | behaviors.tsv |
| **Description** | Hour of the day (0 ... 23\) in local time when the impression occurred. |  |  |  |
| **Justification** | Captures within-day variation in reader activity to support dynamic temporal rotation of spotlight content.  EDA shows a clear diurnal activity cycle, with peaks at around 8-9am. Encoding session hour therefore provide the model with temporal context *t* when estimating *P(click \= 1 \| u, d, t)*. This would allow the article rankings to adapt across different periods of the day rather than remaining static. |  |  |  |
|  |  |  |  |  |
| session\_dow | Categorical | Temporal Dynamics | Categorical \[0, 6\] (8-emb) | behaviors.tsv |
| **Description** | Day of the week (*0 \= Monday ... 6 \= Sunday*). |  |  |  |
| **Justification** | Provides a broader temporal context beyond hour of day, allowing the model to account for potential reader activity and content demands across days. This feature supports the project’s dynamic temporal rotation objective by allowing rank decisions to condition on both the current day and current hour.  However, as MIND-small covers only 7 days, this feature is treated as contextual information rather than evidence of persistent weekly behaviour trends. |  |  |  |
|  |  |  |  |  |
| inactivity\_gap | Numeric & Categorical | Temporal Dynamics | Continuous (hours) & 4-Class Bins | behaviors.tsv |
| **Description** | Continuous hours elapsed since previous recorded impression (time\_since\_last\_click), binned into \<1h, 1h–6h, 6h–24h, \>=24h. |  |  |  |
| **Justification** | Captures the user’s current temporal state rather than absolute clock time. EDA shows substantial variation in return intervals, with a median gap of 7.2 hours. Concentrations of returning activity within short intervals and around 24-hour intervals were also shown in EDA.  This helps distinguish users continuing their recent activity from those returning after longer inactivity, allowing our spotlight rankings to adapt to changes in the recency of observed user behaviour.  |  |  |  |
|  |  |  |  |  |
| slate\_length | Numeric | Session Context | Integer \[2, 299\] | behaviors.tsv |
| **Description** | Total number of candidate news articles exposed in the current impression slate. |  |  |  |
| **Justification** | Captures the competitive context of each impression, as an article competes with a varying number of alternatives for user attention.  EDA shows substantial variation in slate size, with a median of 24, mean of 37.3 and a large range of 2-299 articles. Since each recorded impression contains at least 1 click, exposure-level CTR decreases as slate length increases. Thus, including slate\_length provides context on the number of competing articles when estimating click probability.  |  |  |  |

**Table 4: Feature Metadata Dictionary**

### **2.3.2 Numeric Features & Descriptive Statistics**

Table 5 reports descriptive statistics for the ***scalar numeric features*** defined in Table 4\. In accordance with the interim report requirements, *minimum, maximum, mean, median, standard deviation, and missing-value counts* are reported.

| Feature Name | Count (N) | Mean ± Std Dev | Median  \[Q1, Q3\] | Min | Max | Missing (%) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| *slate\_length (sessions)* | 230,117 | 37.30 ± 38.90 | 24.00  \[10.00, 51.00\] | 2.00 | 299.00 | 0 (0.00%) |
| *history\_length (user-level)* | 94,057 | 21.66 ± 27.78 | 12.00  \[6.00, 26.00\] | 0.00 | 558.00 | 2,122 (2.26%)\* |
| *history\_length (session-level)* | 230,117 | 32.46 ± 39.91 | 19.00  \[8.00, 42.00\] | 0.00 | 558.00 | 5,452 (2.37%)\* |
| *historical\_ctr (articles)* | 22,771 | 0.0432 ± 0.1165 | 0.00  \[0.0000, 0.0356\] | 0.00 | 1.00 | 0 (0.00%) |
| *article\_recency (hours)* | 200,000\*\* | 23.42 ± 23.14 | 16.34  \[10.00, 26.08\] | 0.00 | 143.48 | 0 (0.00%) |
| *inactivity\_gap (hours)* | 136,060 | 16.70 ± 21.04 | 7.24  \[1.67, 23.86\] | 0.00 | 165.28 | 94,057 (40.87%)\*\*\* |
| *reading\_entropy (user)* | 91,935 | 1.43 ± 0.51 | 1.50  \[1.10, 1.82\] | 0.00 | 2.54 | 2,122 (2.26%)\*\*\*\* |

**Table 5: Descriptive Statistics for Numeric Features**

**Note:**  
\* A zero *history\_length* count represents a valid cold-start state rather than an unavailable observation.  
\*\* *article\_recency* statistics were estimated from a sample of 200,000 logged exposure events.  
\*\*\* *inactivity\_gap* is undefined for the first recorded impression of each user because no previous session exists.  
\*\*\*\* *reading\_entropy* cannot be calculated for users without historical interactions.

Moreover, vector representation of *category\_affinity, article\_content\_emb, kg\_entity\_emb, and user\_history\_emb* are not summarized using scalar descriptive statistics because individual vector dimensions do not have directly interpretable units.

### **2.3.3 Categorical Features & Descriptive Statistics**

Table 6 summarises the categorical features defined in Table 4\. For each feature, the number of distinct categories, representative values and their frequencies, and missing-value counts are reported.

| Feature Name | Cardinality | Missing (%) | Planned Representation   |
| :---: | :---: | :---: | :---: |
| *category\_id* | 18 | 0 (0.00%) | Learned 16-dim embedding |
| *subcategory\_id* | 270 | 0 (0.00%) | Learned 32-dim embedding |
| *session\_hour* | 24 | 0 (0.00%) | Cyclic encoding / learned embedding |
| *session\_dow* | 7 | 0 (0.00%) | Learned embedding / one-hot |
| *inactivity\_bin* | 4 \+ new-user state | 94,057 (40.87%)\* | One-hot / learned embedding with \[NEW\_USER\] state |
| *category\_match* | 2 | 0 (0.00%) | Binary |
| *has\_history* | 2 | 0 (0.00%) | Binary |

**Table 6: Descriptive Statistics for Categorical Features**

**Note:**  
\* The missing *inactivity\_bin* observations correspond to each user's first recorded impression and are assigned a dedicated \[NEW\_USER\] state during modelling.

**![][image12]**  
**Fig.12: Distributions for category\_id and subcategory\_id**

The *category\_id* distribution is strongly concentrated: news and sports together account for 60.41% of catalogue articles. The much larger number of subcategories (270) provides finer topical distinctions that can support more specific personalization.  
**![][image13]**  
**Fig.13: Distributions for session\_dow and inactivity\_bin**

Temporal categorical variables further reveal substantial variation across the observation window. *session\_hour* captures within-day behavior, while *session\_dow* captures differences across the seven-day cycle. *inactivity\_bin* complements these absolute temporal features by indicating whether a user is continuing a recent browsing episode or returning after a longer absence.  
**—-----------\-![][image14]**  
**Fig.14: Distributions for category\_match and has\_history**

The *category\_match* distribution also shows that only 22.06% of evaluated candidate exposures belong to a user's historically dominant category. Together with the observed variation in *reading\_entropy*, this provides useful information for calibrating recommendation diversity rather than restricting recommendations to a single dominant interest.

### **2.3.4 Text / Corpus & Knowledge Graph Statistics**

The principal non-tabular content features in Table 4 are *article\_content\_emb* and *kg\_entity\_emb*. Before these representations are generated, their underlying raw text and entity inputs are characterized in Table 7\. The project guidelines specifically recommend reporting text-length distributions and vocabulary sizes for text/corpus data.

| Text / Entity Attribute | Related Feature | Median \[Q1; Q3\] | Mean | Range \[Min; Max\] | Std. Dev | Corpus / Entity Coverage | Missing (%)   |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Title Word Count | *article\_content\_emb* | 10 \[9; 13\] | 10.74 | \[1; 57\] | 3.25 | 728,856 tokens / 34,299 unique words | 0 (0.00%) |
| Title Character Length | *article\_content\_emb* | 64 \[54; 78\] | 66.28 | \[11; 333\] | 19.13 | \- | 0 (0.00%) |
| Abstract Word Count | *article\_content\_emb* | 26 \[17; 65\] | 36.79 | \[1; 474\] | 26.03 | 2,352,371 tokens / 56,577 unique words | 3,415 (5.23%) |
| Abstract Character Length | *article\_content\_emb* | 150 \[101; 395\] | 219.69 | \[1; 2,603\] | 154.74 | \- | 3,415 (5.23%) |
| Combined Text Vocabulary | *article\_content\_emb* | \- | \- | \- | \- | 3,081,227 tokens / 61,000 unique words | \- |
| Title KG Entities | *kg\_entity\_emb* | 2 \[1; 3\] | 2.14 | \[0; 18\] | 1.76 | 26,904 precomputed TransE vectors | 17,543 (26.89%) |
| Abstract KG Entities | *kg\_entity\_emb* | 1 \[0; 3\] | 1.89 | \[0; 24\] | 1.95 | 100-dimensional TransE representations | 17,522 (26.86%) |
| Total KG Entities / Article | *kg\_entity\_emb* | 3 \[1; 4\] | 3.08 | \[0; 32\] | 2.35 | 57,051 articles (87.45%) contain ≥1 linked entity | 8,187 (12.55%) |

**Table 7: Text, Corpus & Knowledge Graph Descriptive Statistics**

***Data Imputation & Processing Conventions:***

1. The text corpus consists of relatively short news content. Titles contain a median of 10 words, while abstracts contain a median of 26 words. This motivates using pretrained semantic representations rather than relying solely on sparse lexical overlap between user histories and candidate articles.  
2. For the 3,415 articles (5.23%) with missing abstracts, the title is duplicated into the abstract field prior to text encoding. This provides a non-empty textual representation while retaining the available headline semantics.  
3. Knowledge graph coverage is also substantial: 57,051 articles (87.45%) contain at least one linked entity, with a median of three entities per article. For the 8,187 articles (12.55%) without linked entities, *kg\_entity\_emb* is initialized as a zero vector 0 ∈ ℝ100, allowing the recommendation model to rely on the article's textual representation when knowledge graph information is unavailable.