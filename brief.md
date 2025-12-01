# Philips Supplier Sustainability Analytics Brief

## Executive Summary
This analysis evaluates the effectiveness of General Information (GI) and Self-Assessment Questionnaire (SAQ) data in predicting supplier sustainability performance (`Val_Score`). Using a supplier-stratified validation approach, we found that augmenting GI data with SAQ responses significantly improves predictive power, nearly doubling the R² score and increasing AUC by 11 points. We recommend a two-stage risk segmentation framework to prioritize interventions, while noting a potential fairness concern regarding higher false negative rates for suppliers in China.

## Methodology
- **Data:** 1,236 assessments from 463 suppliers (2016-2025).
- **Validation:** Stratified split by `Supplier ID` (80/20) to prevent data leakage.
- **Models:** Random Forest Regressor (for scoring) and Classifier (for BIC identification).
- **Feature Sets:**
    1.  **GI-only:** Publicly observable data (Activities, Facilities, Workforce).
    2.  **GI+SAQ:** GI data plus detailed questionnaire responses.

## Results Comparison

The addition of SAQ data provides a substantial lift in model performance, particularly for regression tasks (predicting the exact score).

| Metric | GI-Only Model | GI + SAQ Model | Improvement |
| :--- | :--- | :--- | :--- |
| **RMSE** (Lower is better) | 0.177 | **0.154** | -13% |
| **R²** (Higher is better) | 0.277 | **0.456** | +65% |
| **AUC** (Higher is better) | 0.839 | **0.874** | +4 pts |
| **Accuracy** | 92.1% | 92.5% | +0.4% |

> [!NOTE]
> While accuracy remains stable (likely due to class imbalance), the AUC score demonstrates that the GI+SAQ model is significantly better at ranking suppliers and distinguishing Best-in-Class (BIC) performers.

## Feature Importance & Interpretation

### GI-Only Drivers
The most influential GI features relate to the assessment context and workforce scale.
![GI-Only Feature Importance](analysis_outputs/shap_gi_only.png)
*   **Assessment Year:** Strongest driver, likely reflecting rising sustainability standards over time.
*   **Workforce Size (Total/Male):** Larger workforce counts correlate with score variance, possibly indicating that larger suppliers face more scrutiny or have more resources for compliance.

### GI + SAQ Drivers
When SAQ data is available, specific questionnaire responses become the primary predictors.
![GI+SAQ Feature Importance](analysis_outputs/shap_gi_plus_saq.png)
*   **Q1297 (Health & Safety - Implementation):** A top predictor, suggesting that tangible H&S practices are a strong proxy for overall sustainability maturity.
*   **Q1224 (Environment - Procedures):** Indicates that formalized environmental procedures are a key differentiator.
*   **Missing Indicators:** The presence of missing data (e.g., `missingindicator_Q301_6`) is predictive, often signaling lower maturity or lack of transparency.

## Risk Segmentation Framework

We propose a 5-tier risk segmentation based on the predicted `Val_Score` to guide procurement actions.

![Risk Segmentation](analysis_outputs/risk_segmentation.png)

| Risk Tier | Score Range | Recommended Action |
| :--- | :--- | :--- |
| **Critical Risk** | < 0.50 | **Immediate Audit:** High priority for on-site assessment and corrective action plans. |
| **High Risk** | 0.50 - 0.60 | **Targeted Review:** Request specific documentation for weak areas; schedule audit if unsatisfactory. |
| **Medium Risk** | 0.60 - 0.70 | **Desktop Review:** Monitor via SAQ updates; verify improvements in next cycle. |
| **Moderate Risk** | 0.70 - 0.80 | **Maintain:** Standard monitoring cadence. |
| **Low Risk** | ≥ 0.80 | **Reward/Partner:** Consider for "Best-in-Class" recognition and strategic partnership. |

## Fairness Analysis

We evaluated the model for geographic bias by comparing the False Negative Rate (FNR) – the rate at which actual BIC suppliers are incorrectly classified as non-BIC.

![Fairness Check](analysis_outputs/fairness_check.png)

> [!WARNING]
> **Potential Bias Detected:** The model shows a high False Negative Rate for suppliers in China (~86%). This means high-performing Chinese suppliers are frequently underestimated by the model. This could be due to regional differences in reporting or assessment standards.
> **Mitigation:** Manual review is recommended for Chinese suppliers near the BIC threshold (e.g., predicted score 0.70-0.75) to avoid missing potential BIC candidates.

## Recommendations
1.  **Adopt GI+SAQ Model:** The significant performance lift justifies the effort of collecting SAQ data.
2.  **Focus on "Implementation" Questions:** Prioritize validating answers to key drivers like Q1297 (H&S Implementation) during audits, as these are strong indicators of overall performance.
3.  **Address Regional Bias:** Calibrate the model or apply a specific adjustment factor for Chinese suppliers to reduce the False Negative Rate.
4.  **Data Quality:** Investigate why certain questions (e.g., Q301) have high missing rates that predict poor performance; consider making these mandatory.
