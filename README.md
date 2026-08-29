# Offer Personalization ML

An advanced machine learning solution for predicting and personalizing customer offers. This project leverages LightGBM classifiers, SHAP explainability, and interest score prediction to recommend the most relevant offers to customers.

## Project Overview

This project implements an end-to-end machine learning pipeline for offer personalization:

1. **Data Preprocessing**: Loads and cleans customer-offer interaction data with 43+ features
2. **Feature Engineering**: Organizes features into categories:
   - Interest Scores (var_1 to var_12)
   - Engagement Features (var_13 to var_16)
   - Spend/Debit Patterns (var_17 to var_30)
   - Offer Metadata (var_32 to var_38)
   - Rolling CTR/Engagement (var_39 to var_43)
   - Offer Categories (var_44 to var_50)
3. **Interest Modeling**: Predicts 12 customer interest scores using LGBMRegressor models
4. **Redemption Modeling**: Trains a classifier to predict offer redemption probability
5. **Offer Recommendation**: Generates personalized top-3 offer recommendations per customer
6. **Explainability**: Uses SHAP to explain predictions and feature contributions
7. **Reporting**: Generates PDF reports with waterfall plots and SHAP explanations

## Key Features

### Data Processing
- Handles missing values and outliers
- Log scaling for monetary variables
- CTR normalization (0-1 clipping)
- Train-test split by date or random stratified split

### Modeling
- **Interest Models**: 12 separate LGBMRegressor models predict customer interest in categories
- **Redemption Classifier**: LGBMClassifier predicts offer acceptance probability
- **Hyperparameters**: Tuned for balanced performance with class weighting
- **Metrics**: ROC-AUC and PR-AUC for evaluation

### Explainability
- SHAP Tree Explainers for both interest and redemption models
- Waterfall plots showing feature contributions
- Comparison plots for multiple offers
- Per-customer PDF reports with explanations
- JSON-friendly explanation outputs

### Recommendation Engine
- `recommend_k()`: Returns top-k offers for a customer
- `explain_customer_interest()`: Shows SHAP contributions for each interest score
- `explain_redemption()`: Shows what drives offer redemption
- `unified_offer_explanation()`: Complete explanation with positive/negative drivers

## Files

- **offer_personalisation.ipynb**: Main Jupyter notebook with full pipeline
- **customer_interest_averages (1).csv**: Input data with customer interest scores (var_1-var_12)
- **output files** (generated during execution):
  - `sample_1000.csv`: Sample of 1000 rows for quick testing
  - `predictions_offer.csv`: Redemption probabilities for all customer-offer pairs
  - `top3_offers.csv`: Top 3 offers per customer with scores
  - `customer_interest_averages.csv`: Aggregated customer interest profiles
  - `waterfall.png`: SHAP waterfall plot for a single prediction
  - `compare_offers.png`: SHAP comparison across multiple offers
  - `cust0_report.pdf`: Example customer report with explanations

## Installation

### Requirements
```bash
pip install pandas numpy scikit-learn lightgbm shap matplotlib
```

Or with conda:
```bash
conda install pandas numpy scikit-learn lightgbm shap matplotlib
```

**Note**: The notebook is designed for Google Colab. To run locally:
1. Remove the Google Drive mounting code (cell 2)
2. Update file paths to your local directory

## Usage

### Running the Full Pipeline
```python
# 1. Load data
df = pd.read_parquet(filepath)

# 2. Preprocess features
# (automatic in cells 5-7)

# 3. Train models
# - Interest models (cell 31)
# - Redemption classifier (cell 43)

# 4. Generate recommendations
customer = df_pred.iloc[0]
top3 = recommend_k(customer, offers_unique, k=3)
print(top3)

# 5. Explain predictions
exp = unified_offer_explanation(customer, offer)
print(json.dumps(exp, indent=2))

# 6. Generate reports
generate_customer_pdf_report(customer_idx=0, filename="report.pdf", top_k_offers=3)
```

### Key Functions

#### Recommendation
- `recommend_k(customer_row, offers_df, k=3)`: Get top-k offers
- `compare_offers_shap(customer_row, offers_df, explainer, clf_feature_list, top_k=10, save_path=None)`: Compare offers visually

#### Explanation
- `explain_customer_interest(customer_row)`: SHAP for interest scores
- `explain_redemption(row)`: SHAP for redemption probability
- `unified_offer_explanation(customer_row, offer_row)`: Complete explanation
- `plot_shap_waterfall_for_row(row, explainer, clf_feature_list)`: Force plot

#### Utilities
- `fix_input_row(row, clf_feature_list)`: Prepare data for model
- `get_shap_matrix(shap_out, positive_class_index=1)`: Normalize SHAP output
- `normalize_shap_output(shap_output)`: Handle different SHAP formats

## Model Performance

The models are evaluated using:
- **ROC-AUC**: Measures overall discrimination ability
- **PR-AUC**: Emphasizes performance on positive class (redeemed offers)
- **Per-interest-model metrics**: RMSE, MAE, R² for each interest score

*Note*: Specific metrics depend on your dataset. Run the notebook to see actual performance.

## Architecture

### Data Flow
```
Raw Data → Preprocessing → Feature Engineering → 
  ├─→ Interest Model Training → Predictions
  ├─→ Redemption Classifier Training → Probabilities
  └─→ Offer Recommendation & SHAP Explanations
```

### Feature Categories
| Category | Range | Purpose |
|----------|-------|---------|
| Interest Scores | var_1-var_12 | Customer category preferences |
| Engagement | var_13-var_16 | Historical interaction patterns |
| Spend Patterns | var_17-var_30 | Monetary behavior features |
| Offer Metadata | var_32-var_38 | Offer characteristics |
| CTR Features | var_39-var_43 | Click-through rate metrics |
| Categories | var_44-var_50 | Offer categories (one-hot encoded) |

## Troubleshooting

### Common Issues

**Issue**: `KeyError: 'offer_id_encoded'` in recommendation functions
- **Solution**: Ensure `offer_le.fit_transform()` has been executed before calling recommendation functions

**Issue**: SHAP values are `None` or empty
- **Solution**: Verify the classifier `clf` has been trained and `explainer_clf` initialized

**Issue**: Missing columns in predictions
- **Solution**: Verify all expected columns exist in input data; use `fix_input_row()` helper

**Issue**: Notebook runs but PDF generation fails
- **Solution**: Install matplotlib: `pip install matplotlib`

## Advanced Usage

### Custom Recommendations
```python
# Get offers for a specific customer ID
customer_data = df_pred[df_pred['customer_id'] == '1000125'].iloc[0]
top_offers = recommend_k(customer_data, offers_unique, k=5)
```

### Batch Explanations
```python
# Explain top-3 offers for multiple customers
for idx in range(100):
    customer = df_pred.iloc[idx]
    exp = unified_offer_explanation(customer, offers_unique.iloc[0])
    print(f"Customer {idx}: {exp['probability']:.3f} probability")
```

### Export Results
```python
# Save all recommendations
df_out.to_csv('all_offers_scored.csv', index=False)

# Save top-3 aggregated by customer
top3_offers.to_csv('personalized_offers.csv', index=False)

# Save customer profiles
df_avg.to_csv('customer_profiles.csv', index=False)
```

## Output Interpretation

### `predictions_offer.csv`
- `customer_id`: Customer identifier
- `offer_id`: Offer identifier
- `p_redeem`: Predicted redemption probability (0-1)

### `top3_offers.csv`
- Best offers for each customer ranked by probability
- One row per (customer, offer) pair, up to 3 per customer

### `customer_interest_averages.csv`
- Aggregated customer interest profiles
- Columns: `customer_id`, `var_1` to `var_12`
- Values: Average interest scores (0-100 range)

## Performance Optimization

For large datasets:
1. **Reduce SHAP sample size**: Change `5000` to smaller value in sampling cells
2. **Use tree_limit**: Add `tree_limit` parameter to SHAP for faster computation
3. **Batch processing**: Process customers in chunks instead of all at once
4. **Disable PDF generation**: Skip report generation for speed


## License

This project is open-source. Modify and use as needed for your offer personalization needs.

## Contact & Support

For issues or questions, please refer to the notebook comments or open an issue in the repository.