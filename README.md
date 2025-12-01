# Philips Supplier Sustainability Analytics - Data Package

## Quick Start

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

**Main Deliverable:** 4-6 page brief (slide deck or written report)

**Core Tasks:**
1. Build GI-only models (observable supplier characteristics)
2. Build GI+SAQ models (with detailed questionnaires)
3. Compare both approaches
4. Create risk segmentation framework
5. Check fairness across regions

**Expected Timeline:** 10-14 hours

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
**You MUST split by supplier ID, not randomly.**

Random split causes data leakage (same supplier in train and test).

---

## Data Notes

### Q-Codes (SAQ Features)
- Question text is confidential (Philips proprietary)
- You have: Q-code → Topic → Chapter mapping
- This is sufficient for interpretation
- Example: "Q1272 ranks #1" → "Environment Corrective Action matters most"

### Missing Values
- Document your approach

### Geographic Distribution
- Check for fairness across regions in your analysis

---

## Expected Performance Ranges

If your results differ but methodology is sound, that's acceptable.

---

## Deliverable Checklist

Your brief should include:

- [ ] Two-stage results comparison (GI-only vs GI+SAQ)
- [ ] Feature importance for both stages with business interpretation
- [ ] Risk segmentation framework (3-5 tiers with recommended actions)
- [ ] Fairness analysis (check performance across regions)
- [ ] Two visualizations: feature importance chart + calibration/comparison plot
- [ ] Business recommendations

---

## Common Mistakes to Avoid

❌ Random train/test split (data leakage)  
❌ Skipping GI-only analysis  
❌ Listing Q-codes without interpreting topics  
❌ No fairness check  
❌ Notebook-focused instead of brief-focused  

---

## Resources

**Reference:** Tan et al. (2024) "Stop Auditing and Start to CARE" - INFORMS Journal on Applied Analytics

---

## Academic Integrity

- Use dataset for coursework only
- Work independently
- Cite any external resources used
- Do not share code with classmates

---

**Good luck! Focus on the business insights, not just the models.**
