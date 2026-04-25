# Part B: Business Case Analysis
## Scenario: Promotion Effectiveness at a Fashion Retail Chain

### B1. Problem Formulation
#### B1(a) – ML problem definition
We want to learn, for each store and month, which promotion maximises the number of items sold.

Target variable: monthly items sold for a given store under a given promotion (e.g., items_sold aggregated at store–month–promotion level).

Candidate input features (examples):

Store characteristics: store_id (encoded), store_size, location_type (urban, semi-urban, rural), historical average items_sold, typical footfall, competition_density.

Promotion variables: promotion_type (Flat Discount, BOGO, Free Gift, Category Offer, Loyalty Points), discount depth/category, promotion duration.

Time and calendar: year, month, season, is_weekend share, number of festival days in the month.

Local context: broader economic indicators, local events (if available).

This is a supervised regression problem, because the target items_sold is a continuous numerical outcome and we have historical labeled data mapping feature combinations to realised sales volumes.

#### B1(b) – Why use items sold instead of revenue?
Total sales revenue combines both unit volume and price, so it can be heavily influenced by temporary price changes (e.g., deep discounts) or mix shifts towards high-priced items.

Two promotions might yield similar revenue but very different unit volumes; for a retailer focused on clearing inventory, driving store footfall, or upselling add-ons, unit volume is a more direct measure of promotional effectiveness than revenue.

Moreover, revenue can be distorted by external pricing decisions (e.g., margin strategy, inflation, tax changes) that are not under the control of the promotion algorithm and may confound model learning.

Using items sold as the target focuses the model on customer response to promotions, independent of price level, and thus yields more stable and interpretable relationships over time.

The broader principle is: choose a target variable that directly reflects the business objective and is as stable and causally close to the decision as possible, rather than a noisy proxy influenced by many unrelated factors.

#### B1(c) – Beyond a single global model
A single global model across all 50 stores assumes that the relationship between promotions and items sold is the same everywhere, ignoring local differences in demographics, competition, and store format.

A better strategy is a hierarchical / multi-level approach, for example:

Train a global base model that uses all stores’ data but includes store-level features (store_size, location_type, competition_density) and possibly store_id as encoded features.

Allow for store- or segment-specific effects, such as:

Separate models for clusters of similar stores (e.g., urban vs rural vs semi-urban, or segments based on historical response patterns).

A global model plus store-level bias terms (random effects) learned via mixed-effects models or store embeddings.

This approach leverages the scale of data across all stores while still allowing different locations to respond differently to the same promotion, improving both accuracy and business trust.

### B2. Data and EDA Strategy
#### B2(a) – Joining tables and defining the grain
We have four tables:

Transactions: one row per individual transaction (date, store_id, promotion applied, items_sold, basket details).

Store attributes: one row per store (store_id, store_size, location_type, typical monthly footfall, competition_density).

Promotion details: one row per promotion instance (promotion_id or store–date–promotion_type, depth, category, mechanics).

Calendar: one row per date (transaction_date, is_weekend, is_festival, month, year, etc.).

Join keys:

Transactions ↔ Store attributes on store_id.

Transactions ↔ Calendar on transaction_date.

Transactions ↔ Promotion details on a promotion key (e.g., store_id + transaction_date + promotion_type, or promotion_id if present).

For modelling promotion effectiveness at a planning horizon of one month per store, the grain of the final dataset should be:

One row per store per month per promotion_type (or, if only one promotion runs per store per month, one row per store per month).

Aggregations before modelling:

From transactions to store–month–promotion level:

Sum of items_sold (target).

Total revenue, average basket size, distinct customers (for potential features).

Counts or proportions of days with promotions vs non-promotion days in that month.

From calendar:

Number of weekends, number of festival days per month.

From store attributes:

Direct join; no aggregation needed (they are already at store level).

This yields a clean modelling table where each row is a potential decision point: “Store S in month M under promotion P had X items sold”.

#### B2(b) – EDA before modelling
You can structure EDA around at least four analyses/charts:

Promotion-type vs items_sold distribution

Chart: Boxplot or violin plot of items_sold by promotion_type.

Look for: Which promotions tend to produce higher median and upper-tail volumes, and whether the effect differs by location_type or store_size.

Influence: If one promotion consistently underperforms, consider whether it should be modelled differently, or whether to include interaction features like promotion_type × location_type.

Time-series plots of monthly items_sold by store

Chart: Line plots of items_sold over time for a sample of stores or aggregated by store segment.

Look for: Seasonality (e.g., festive months), trends, abrupt shifts after new competitors enter.

Influence: Leads to engineered features for year, month, season, festivals, and possibly lag features (e.g., last month’s sales) if you extend to time-series modelling.

Relationship between competition_density and items_sold

Chart: Scatter plot or binned line plot of items_sold vs competition_density, possibly separated by location_type.

Look for: Whether dense competition depresses promotion effectiveness and whether certain promotions work better in competitive areas.

Influence: Motivate interaction features such as promotion_type × competition_density or location_type × competition_density.

Store-size and location-type effects

Chart: Grouped bar charts or boxplots of items_sold by store_size and location_type.

Look for: Systematic differences between small, medium, large stores and between urban vs rural stores.

Influence: Supports the earlier decision to include store_size and location_type as key features and potentially to segment the model or apply different thresholds.

Promotion coverage and missingness (optional fifth)

Chart: Heatmap of promotion_type usage by store and month, or bar chart of proportion of days with promotions.

Look for: Whether some stores rarely run certain promotions (data sparsity), which can weaken estimates for those combinations.

Influence: May suggest regularisation, partial pooling (hierarchical models), or down-weighting rare configurations, and informs which promotion–store combinations we can trust.

#### B2(c) – Imbalance: 80% transactions without promotion
If 80% of transactions occur without any promotion, the model may mostly learn patterns of baseline demand and underfit the incremental effect of promotions.

For example, a model might infer that “no promotion” is the safest prediction and fail to capture how specific promotions shift items_sold relative to baseline.

Mitigation steps:

Change the framing: Model uplift or incremental lift (items_sold with promotion minus expected items_sold without promotion) instead of raw volume, where feasible.

Rebalance the training data: Sample more promotion transactions or use weighting so that promotion events get higher importance in the loss function.

Separate sub-models: Train a baseline demand model on non-promotion data and a second model that estimates incremental uplift for each promotion_type, then combine them at decision time.

These steps ensure the algorithm pays sufficient attention to the rarer but business-critical promotion periods.

### B3. Model Evaluation and Deployment
#### B3(a) – Train-test split and metrics
We have three years of monthly data per store, so the data is a panel: store × month.

A random split is inappropriate because it would mix past and future months in both train and test sets. This leads to temporal leakage (learning from future data), over-optimistic performance, and unrealistic evaluation for forecasting.

A better setup:

Temporal split:

Use the first 2.5 years as training (older months across all stores).

Use the last 0.5 year as the test period (most recent months for all stores).

Optionally, within training, use rolling or expanding-window cross-validation (time-series CV) to tune hyperparameters and assess stability over time.

Evaluation metrics and interpretation:

RMSE (Root Mean Squared Error): Measures the typical magnitude of prediction errors in items_sold; penalises large mistakes more heavily. A lower RMSE indicates better overall accuracy. In this context, it reflects how far predicted monthly sales per store can be from actual sales.

MAE (Mean Absolute Error): Average absolute prediction error in items_sold, easy to explain to business teams (“on average we are off by X items per store per month”).

Optionally MAPE (Mean Absolute Percentage Error) if you need relative percentage errors, but be cautious when items_sold is very small.

These metrics together tell you whether the model is accurate enough to be useful, and how much deviation the retailer should expect when following the recommended promotions.

#### B3(b) – Explaining different recommendations via feature importance
The model recommends Loyalty Points Bonus in December and Flat Discount in March for Store 12.

To explain this, we can examine both global and local feature importance:

Global feature importance (e.g., from a tree-based model):

Identify which features generally influence items_sold most: promotion_type, month, is_festival, competition_density, store_size, etc.

This tells marketing that overall, the model finds these factors to be key drivers of promotion success.

Local explanation for Store 12 / specific months:

Use techniques like SHAP values or permutation importance computed for the December row and the March row for Store 12.

For December, the explanation may show large positive contributions from features like month=December, high festival intensity, high historical responsiveness to loyalty points at this store, making Loyalty Points Bonus the best option.

For March, feature contributions might show that no festivals, higher price sensitivity, or heavier competition make Flat Discount more effective.

When communicating to the marketing team, avoid raw technical details and instead summarise:

“In December, Store 12’s customers respond strongly to loyalty rewards and festival-driven shopping, so a Loyalty Points Bonus yields more items sold. In March, there are fewer festivals and customers become more price-sensitive, so Flat Discount is more effective. The model captures these patterns from past data, where similar conditions led to higher volumes under those promotions.”

This uses feature importance as a storytelling tool to tie recommendations back to intuitive drivers.

#### B3(c) – End-to-end deployment and monitoring
Saving and deploying the model

After training and validating the chosen model (e.g., a pipeline with preprocessing plus a regression/decision model), save the entire pipeline object (preprocessing + model) using a serialisation format such as joblib or pickle.

Deploy the saved pipeline to a production environment (e.g., a scheduled batch job, an API service, or a data pipeline in the cloud).

Monthly data preparation and scoring

At the start of each month:

Gather the latest store attributes (store_size, competition_density), promotion candidates for that month, and calendar features (month, festival days, weekends).

Construct the same feature table as in training: one row per store per candidate promotion_type for the coming month.

Pass this feature table through the saved pipeline’s .predict() method to obtain predicted items_sold for each store–promotion combination.

For each store, select the promotion with the highest predicted items_sold as the recommendation and send it to the marketing team or promotion-planning system.

Monitoring and retraining

Performance monitoring:

Log actual items_sold each month per store and compare against predictions (RMSE/MAE over time) and against baseline strategies (e.g., last year’s promotion).

Track key diagnostics: error distribution by store segment, by promotion_type, and by season.

Data and concept drift checks:

Monitor changes in input distributions (e.g., competition_density, store footfall, share of online sales) that may indicate the environment has changed.

If errors trend upward over several months or drift metrics exceed thresholds, mark the model as degraded.

Retraining policy:

Retrain on the most recent 2–3 years of data on a fixed schedule (e.g., yearly or quarterly), or sooner if monitoring flags performance degradation.

After retraining, validate the new model, compare against the existing one with backtesting, and only replace the production model once it clearly outperforms the old one.

This process ensures the model generates timely monthly recommendations without retraining every cycle, while still adapting periodically as store behaviour and market conditions evolve.