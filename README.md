# Philips Supplier Sustainability Analytics - Data Package

## Quick Start

In this repository, you will find the brief in the markdown `brief.md` and the Python notebook in the file `starter_notebook.ipynb`.

### 1. Install Required Packages
```bash
pip install -r requirements.txt
```

### 2. Load Data
```python
df = pd.read_excel('data/SSP_Data.xlsx')
```

---

## Assignment Requirements

**Main Deliverable:** 4-6 page brief

**Core Tasks:**
1. Build GI-only models (observable supplier characteristics)
2. Build GI+SAQ models (with detailed questionnaires)
3. Compare both approaches
4. Create risk segmentation framework
5. Check fairness across regions

---

## Data File

### SSP_Data.xlsx - Data Sheet
- **1,236 assessments** from **463 suppliers** (2016-2025)
- **Target variable:** `Val_Score` (Overall Sustainability score, 0-1 scale)
- **Topic scores:** Val_Environment, Val_Health and Safety, Val_Business Ethics, Val_Labor and Human Rights
- **GI features (31):** Activities, Facilities, Number of workers, Country
- **SAQ features (~450):** Q-code columns (filter out columns with >50% missing)

### SSP_Data.xlsx - Column Explanation
- Maps Q-codes to topics and chapters
- Example: Q1272 → Environment - Corrective Action Approach
- Use this to interpret SAQ feature importance

---

## Critical Requirements

### ⚠️ Train/Test Split
**Split by supplier ID, not randomly.**

Random split causes data leakage (same supplier in train and test).

---

## Data Notes

### Q-Codes (SAQ Features)
- Question text is confidential (Philips proprietary)
- Q-code → Topic → Chapter mapping
- Example: "Q1272 ranks #1" → "Environment Corrective Action matters most"

### Missing Values
- Median imputation for numeric features and mode imputation for categorical features. Drop columns with more than 95% missing values.

### Geographic Distribution
- Check for fairness across regions.

---

## Deliverable Checklist

Brief includes:

- Two-stage results comparison (GI-only vs GI+SAQ)
- Feature importance for both stages with business interpretation
- Risk segmentation framework (3-5 tiers with recommended actions)
- Fairness analysis (check performance across regions)
- Two visualizations
- Business recommendations

---

## Resources

**Reference:** Tan et al. (2024) "Stop Auditing and Start to CARE" - INFORMS Journal on Applied Analytics

---
