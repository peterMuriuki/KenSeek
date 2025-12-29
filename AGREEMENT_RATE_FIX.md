# Agreement Rate Fix - Summary

## Issue

The `agreement_rate` column in the `extraction_results` table was not being populated correctly. It was storing `0.0` instead of the actual agreement rate (e.g., 0.67, 1.00).

## Root Cause

In `database.py`, the `save_extraction_results()` function was using incorrect keys when extracting values from the `confidence_metrics` dictionary:

### Before (Incorrect)

```python
cursor.execute("""
    INSERT INTO extraction_results
    (attempt_id, metric_name, consensus_value, confidence,
     agreement_rate, vote_distribution)
    VALUES (?, ?, ?, ?, ?, ?)
""", (
    attempt_id,
    metric_name,
    consensus_value,
    confidence_info.get('confidence', 'unknown'),      # ✗ Key doesn't exist
    confidence_info.get('agreement', 0.0),             # ✗ This is a string like "2/3"
    json.dumps(confidence_info.get('votes', {}))       # ✗ Key doesn't exist
))
```

### Confidence Metrics Structure

From `_compute_consensus()` in `pdf_extractor.py`:

```python
confidence_metrics[metric] = {
    "final_value": most_common_value,
    "confidence_score": round(confidence, 2),    # ← Agreement rate as decimal (0.67, 1.00)
    "confidence_level": confidence_level,         # ← "high", "medium", "low"
    "agreement": f"{count}/{len(samples)}",       # ← String representation "2/3"
    "samples": values,
    "vote_distribution": dict(value_counts)       # ← Dictionary of vote counts
}
```

## Fix

Updated `database.py` to use the correct dictionary keys:

### After (Correct)

```python
cursor.execute("""
    INSERT INTO extraction_results
    (attempt_id, metric_name, consensus_value, confidence,
     agreement_rate, vote_distribution)
    VALUES (?, ?, ?, ?, ?, ?)
""", (
    attempt_id,
    metric_name,
    consensus_value,
    confidence_info.get('confidence_level', 'unknown'),     # ✓ Correct key
    confidence_info.get('confidence_score', 0.0),           # ✓ Correct key (decimal rate)
    json.dumps(confidence_info.get('vote_distribution', {})) # ✓ Correct key
))
```

## Changes Made

**File: `database.py`** (Lines 339-341)

| Column | Before | After | Description |
|--------|--------|-------|-------------|
| `confidence` | `confidence_info.get('confidence', 'unknown')` | `confidence_info.get('confidence_level', 'unknown')` | Stores "high", "medium", or "low" |
| `agreement_rate` | `confidence_info.get('agreement', 0.0)` | `confidence_info.get('confidence_score', 0.0)` | Stores decimal rate (0.00 to 1.00) |
| `vote_distribution` | `confidence_info.get('votes', {})` | `confidence_info.get('vote_distribution', {})` | Stores vote counts |

## Impact

### Before Fix
- `agreement_rate` column stored `0.0` for all records
- CSV export would show `0.00` for all confidence columns
- Couldn't determine which extractions were high/low quality

### After Fix
- `agreement_rate` column stores actual decimal rates (e.g., 0.67, 1.00)
- CSV export shows correct agreement rates
- Can filter/identify low-confidence extractions

## Examples

### Database Storage

**Before:**
```sql
metric_name    | consensus_value | confidence | agreement_rate
---------------|-----------------|------------|---------------
net_sales      | 1000000        | unknown    | 0.0
credit_sales   | 500000         | unknown    | 0.0
```

**After:**
```sql
metric_name    | consensus_value | confidence | agreement_rate
---------------|-----------------|------------|---------------
net_sales      | 1000000        | high       | 1.00
credit_sales   | 500000         | medium     | 0.67
debtors        | 200000         | low        | 0.33
```

### CSV Export (with --confidence flag)

**Before:**
```csv
Company,Year,Net_Sales,Net_Sales_Confidence
Test Co,2023,1000000,0.00
```

**After:**
```csv
Company,Year,Net_Sales,Net_Sales_Confidence
Test Co,2023,1000000,1.00
```

## Validation

Created `test_agreement_rate_fix.py` to verify the fix:

```bash
$ python3 test_agreement_rate_fix.py
======================================================================
AGREEMENT RATE STORAGE TEST
======================================================================

Expected agreement rates:
  net_sales: 1.00 (3/3 agreement)
  credit_sales: 0.67 (2/3 agreement)
  debtors: 0.33 (1/3 agreement)

Actual values from database:
  ✓ credit_sales: 0.67 (expected: 0.67)
  ✓ debtors: 0.33 (expected: 0.33)
  ✓ net_sales: 1.0 (expected: 1.0)

======================================================================
ALL TESTS PASSED ✓
Agreement rates are correctly stored as decimals!
======================================================================
```

## Agreement Rate Interpretation

| Rate | Meaning | Confidence |
|------|---------|------------|
| 1.00 | All samples agreed (e.g., 3/3) | High |
| 0.67 | Most samples agreed (e.g., 2/3) | Medium |
| 0.33 | Low agreement (e.g., 1/3) | Low |

For `num_samples=3` (default):
- `1.00` = All 3 samples extracted the same value
- `0.67` = 2 out of 3 samples agreed
- `0.33` = Each sample gave a different value (no consensus)

For `num_samples=5`:
- `1.00` = All 5 samples agreed
- `0.80` = 4 out of 5 agreed
- `0.60` = 3 out of 5 agreed
- `0.40` = 2 out of 5 agreed
- `0.20` = No clear consensus

## Usage

The agreement rate is now correctly available in:

### 1. Database Queries

```sql
-- Find low-confidence extractions
SELECT company_name, financial_year, metric_name, consensus_value, agreement_rate
FROM extraction_results er
JOIN extraction_attempts ea ON er.attempt_id = ea.id
WHERE agreement_rate < 0.6
ORDER BY agreement_rate;
```

### 2. CSV Export

```bash
# Export with confidence columns
python pdf_extractor.py --export results.csv --confidence

# The CSV will include columns like:
# - Net_Sales_Confidence (e.g., 1.00)
# - Credit_Sales_Confidence (e.g., 0.67)
# - Debtors_Confidence (e.g., 0.33)
```

### 3. Python API

```python
from database import ExtractionDatabase

db = ExtractionDatabase("./extractions.db")
extractions = db.get_all_successful_extractions()

for extraction in extractions:
    company = extraction['company_name']
    year = extraction['financial_year']

    for metric, conf in extraction['confidence_metrics'].items():
        rate = conf['agreement_rate']
        if rate < 0.6:
            print(f"Low confidence: {company} {year} - {metric}: {rate}")
```

## Notes

- **Existing database records** with incorrect agreement_rate values will remain unchanged
- **New extractions** will have correct agreement_rate values
- To fix existing records, you would need to re-run extractions
- The CSV exporter correctly reads from `agreement_rate` column (no changes needed there)
- The fix is backward compatible (old records with 0.0 will just show as 0.00 in CSV)

## Status

✅ **Fixed and Validated**
- Database storage corrected
- Test coverage added
- All tests passing
- Ready for production use

New extractions will now correctly store and export agreement rates in CSV format.
