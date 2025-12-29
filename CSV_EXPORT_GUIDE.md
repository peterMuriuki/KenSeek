# CSV Export Guide

This guide explains how to export extraction results from the database to CSV format with derived financial metrics.

## Overview

The CSV export functionality:
- Exports all successful extractions from the database
- Calculates derived financial metrics (ratios, percentages, etc.)
- Formats data to match the required CSV structure
- Optionally includes confidence rates for each extracted value
- Supports filtering by company and/or year

## Quick Start

### Basic Export

Export all extractions to CSV:

```bash
python pdf_extractor.py --export results.csv
```

### Export with Confidence Rates

Include agreement confidence rates for each metric (adds 18 columns):

```bash
python pdf_extractor.py --export results.csv --confidence
```

### Filtered Exports

Export specific company:

```bash
python pdf_extractor.py --export results.csv --company "SGL Limited"
```

Export specific year:

```bash
python pdf_extractor.py --export results.csv --year "2023"
```

Export specific company and year:

```bash
python pdf_extractor.py --export results.csv \
  --company "WILLIAMSON TEA KENYA" \
  --year "2023"
```

### With Metadata Fields

Add sector and code information:

```bash
python pdf_extractor.py --export results.csv \
  --sector "Manufacturing" \
  --sector-short "MFG" \
  --code "001"
```

## CSV Column Structure

### Base Columns (37 total)

The exported CSV includes the following columns:

#### Metadata Columns
1. `code` - Record code (user-provided)
2. `Sector` - Sector name (user-provided)
3. `SectorS` - Short sector name (user-provided)
4. `Company` - Company name (from extraction)
5. `Cos` - Company short name (empty)
6. `Year` - Financial year (from extraction)

#### Extracted Financial Data (in KES)
7. `Credit_Sales (kes)` - Total credit sales
8. `Net_Sales (Kes)` - Net revenue/sales
9. `Debtors (Kes)` - Accounts receivable
10. `Impaired_Debts (Kes)` - Bad debts/provisions
11. `Current_Assets (Kes)` - Total current assets
12. `Current_Liabilities (Kes)` - Total current liabilities
13. `Cash (Kes)` - Cash and cash equivalents
14. `Marketable_Securities (Kes)` - Short-term investments
15. `Total_liabilities (Kes)` - Total liabilities
16. `Total_assets (Kes)` - Total assets
17. `EBIT (Kes)` - Earnings before interest and tax
18. `Interest (Kes)` - Interest expense
19. `Labour cost (Kes)` - Employee costs
20. `Operating expenses (Kes)` - Operating expenses
21. `R&D_cost (Kes)` - Research and development costs
22. `Tax (Kes)` - Income tax expense
23. `Equity (Kes)` - Shareholders' equity
24. `Netincome (Kes)` - Net profit

#### Derived Metrics (Calculated)
25. `Trade_credit_exposure_rate` - Debtors / Net Sales
26. `Default_Rate` - Impaired Debts / Debtors
27. `Current_ratio` - Current Assets / Current Liabilities
28. `C+MS (Kes)` - Cash + Marketable Securities
29. `Cash_Ratio` - (Cash + MS) / Current Liabilities
30. `Debt_ratio` - Total Liabilities / Total Assets
31. `EBIT - Interest (Kes)` - EBIT minus Interest
32. `DFL` - Degree of Financial Leverage (EBIT / (EBIT - Interest))
33. `LCOR (Kes)` - Labour Cost / Operating Expenses
34. `R&D_cost_ratio` - R&D Cost / Net Sales
35. `Firm_Size (Ln Assets)` - Natural log of Total Assets
36. `ROE` - Return on Equity (Net Income / Equity)
37. `ROS` - Return on Sales (Net Income / Net Sales)

### Confidence Columns (18 additional, optional)

When `--confidence` flag is used, these columns are added:

1. `Credit_Sales_Confidence` - Agreement rate (0.00-1.00)
2. `Net_Sales_Confidence`
3. `Debtors_Confidence`
4. `Impaired_Debts_Confidence`
5. `Current_Assets_Confidence`
6. `Current_Liabilities_Confidence`
7. `Cash_Confidence`
8. `Marketable_Securities_Confidence`
9. `Total_liabilities_Confidence`
10. `Total_assets_Confidence`
11. `EBIT_Confidence`
12. `Interest_Confidence`
13. `Labour_cost_Confidence`
14. `Operating_expenses_Confidence`
15. `R&D_cost_Confidence`
16. `Tax_Confidence`
17. `Equity_Confidence`
18. `Net_income_Confidence`

**Confidence Rate Interpretation:**
- `1.00` = 100% agreement (all samples agreed)
- `0.67` = 67% agreement (2 out of 3 samples agreed)
- `0.33` = 33% agreement (only 1 out of 3 samples agreed)
- `NA` = No confidence data available

## Data Formatting

### Currency Values
- Formatted with thousand separators
- No decimal places
- Example: `2,590,416`
- `NA` for missing/unavailable values

### Ratio/Percentage Values
- Formatted with 4 decimal places
- Example: `0.1234`
- `NA` for missing/incalculable values

### Special Cases
- Division by zero → `NA`
- Missing source data → `NA`
- Negative values are preserved (e.g., negative EBIT)

## Derived Metric Formulas

| Metric | Formula | Notes |
|--------|---------|-------|
| Trade Credit Exposure Rate | Debtors / Net Sales | |
| Default Rate | Impaired Debts / Debtors | |
| Current Ratio | Current Assets / Current Liabilities | |
| C+MS | Cash + Marketable Securities | Sum of liquid assets |
| Cash Ratio | (Cash + MS) / Current Liabilities | |
| Debt Ratio | Total Liabilities / Total Assets | |
| EBIT - Interest | EBIT - Interest | |
| DFL | EBIT / (EBIT - Interest) | Degree of Financial Leverage |
| LCOR | Labour Cost / Operating Expenses | Labour Cost to Operating Revenue |
| R&D Cost Ratio | R&D Cost / Net Sales | |
| Firm Size | ln(Total Assets) | Natural logarithm |
| ROE | Net Income / Equity | Return on Equity |
| ROS | Net Income / Net Sales | Return on Sales |

## Command-Line Options

### Required
- `--export <file.csv>` - Output CSV file path

### Optional Filters
- `--company <name>` - Filter by company name (exact match)
- `--year <YYYY>` - Filter by financial year
- `--db <path>` - Database path (default: `./extractions.db`)

### Optional Metadata
- `--sector <name>` - Sector name to include in all rows
- `--sector-short <code>` - Short sector code
- `--code <code>` - Code field for all rows

### Optional Features
- `--confidence` - Include 18 confidence rate columns

## Usage Examples

### Example 1: Export All Data

```bash
python pdf_extractor.py --export all_companies.csv
```

**Result:** CSV with all successful extractions

### Example 2: Export with Confidence Rates

```bash
python pdf_extractor.py --export results_with_confidence.csv --confidence
```

**Result:** CSV with 55 columns (37 base + 18 confidence)

### Example 3: Export Single Company

```bash
python pdf_extractor.py --export williamson_tea.csv \
  --company "WILLIAMSON TEA KENYA"
```

**Result:** CSV with only Williamson Tea Kenya records

### Example 4: Export Single Year Across Companies

```bash
python pdf_extractor.py --export year_2023.csv --year "2023"
```

**Result:** CSV with all companies for year 2023

### Example 5: Full Featured Export

```bash
python pdf_extractor.py --export manufacturing_sector.csv \
  --sector "Manufacturing" \
  --sector-short "MFG" \
  --code "MFG001" \
  --confidence \
  --db ./extractions.db
```

**Result:** CSV with sector metadata and confidence rates

## Workflow Example

Complete workflow from extraction to CSV export:

```bash
# Step 1: Extract data from PDFs (batch mode)
python pdf_extractor.py --folder ./reports --workers 4

# Step 2: Verify extractions in database
sqlite3 extractions.db "
  SELECT company_name, financial_year, success
  FROM extraction_attempts
  WHERE success = 1
  ORDER BY company_name, financial_year;
"

# Step 3: Export all to CSV with confidence
python pdf_extractor.py --export results.csv --confidence

# Step 4: Export by sector (if you have multiple sectors)
python pdf_extractor.py --export manufacturing.csv \
  --sector "Manufacturing" \
  --confidence

# Step 5: Export specific companies for analysis
python pdf_extractor.py --export top_companies.csv \
  --company "SGL Limited" \
  --confidence
```

## Standalone CSV Exporter

You can also use the CSV exporter as a standalone script:

```bash
python csv_exporter.py --output results.csv --confidence

# With filters
python csv_exporter.py --output results.csv \
  --company "SGL Limited" \
  --year "2023" \
  --confidence \
  --sector "Manufacturing"
```

## Data Quality Notes

### Confidence Rates
- Based on agreement rate from self-consistency sampling
- Higher confidence (>0.67) indicates more reliable data
- Lower confidence (<0.5) suggests manual review may be needed
- Use `--confidence` flag to identify low-quality extractions

### Missing Values (NA)
- `NA` indicates data was not found or extractable
- Some metrics (like R&D cost) are commonly `NA` if not disclosed
- Derived metrics are `NA` if source data is missing

### Negative Values
- Negative EBIT, Net Income, etc. are preserved as-is
- Indicates losses or negative performance
- Not converted to NA

## Troubleshooting

### "No successful extractions found"
- Check database has successful extractions: `sqlite3 extractions.db "SELECT COUNT(*) FROM extraction_attempts WHERE success = 1;"`
- Verify company name matches exactly (case-sensitive)
- Check if extractions were actually successful

### Missing derived metrics
- Check if source data exists (e.g., can't calculate ROE without equity)
- Review extraction results for NA values
- Some metrics require multiple fields (e.g., Cash Ratio needs cash, MS, and current liabilities)

### Confidence columns show NA
- Extraction may not have been run with self-consistency (num_samples=1)
- Database may be missing confidence data
- Re-run extractions with `--num-samples 3` or higher

## Database Schema

The export queries these tables:

```sql
-- Extraction attempts (metadata)
extraction_attempts:
  - company_name
  - financial_year
  - success
  - num_samples

-- Extraction results (values)
extraction_results:
  - metric_name
  - consensus_value
  - confidence
  - agreement_rate
```

Query example to check data:

```sql
SELECT
  ea.company_name,
  ea.financial_year,
  er.metric_name,
  er.consensus_value,
  er.agreement_rate
FROM extraction_attempts ea
JOIN extraction_results er ON ea.id = er.attempt_id
WHERE ea.success = 1
  AND ea.company_name = 'SGL Limited'
  AND er.metric_name = 'net_sales';
```

## Notes

- Currency values are in Kenyan Shillings (KES)
- All ratios and percentages are decimal format (not percentages)
- Natural log is used for Firm Size (not log10)
- CSV uses UTF-8 encoding
- Headers match the exact format from your example
- Empty fields in metadata (code, sector) are handled gracefully
