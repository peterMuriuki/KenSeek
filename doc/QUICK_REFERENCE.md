# Quick Reference Card

## PDF Financial Data Extractor - Common Commands

**Note:** All commands now use `cli.py` as the single entry point.

### Single File Extraction

```bash
# Basic extraction (uses OCR by default)
python cli.py report.pdf --company "SGL Limited" --year "2023"

# Save to JSON
python cli.py report.pdf --company "SGL Limited" --year "2023" \
  --output results.json

# Show reasoning and detailed confidence
python cli.py report.pdf --show-reasoning --show-confidence
```

### Batch Extraction

```bash
# Extract all companies (2 workers, safe default)
python cli.py --folder ./reports --workers 2

# Extract specific companies (4 workers for speed)
python cli.py --folder ./reports \
  --companies "SGL Limited" "WILLIAMSON TEA KENYA" \
  --workers 4

# Save individual results + summary
python cli.py --folder ./reports \
  --workers 3 \
  --output-dir ./results \
  --batch-summary ./summary.json
```

### CSV Export

```bash
# Export all to CSV
python cli.py --export results.csv

# Export with confidence rates (55 columns)
python cli.py --export results.csv --confidence

# Export specific company
python cli.py --export results.csv \
  --company "SGL Limited" \
  --confidence

# Export with metadata
python cli.py --export results.csv \
  --sector "Manufacturing" \
  --sector-short "MFG" \
  --code "001" \
  --confidence
```

### Complete Workflow

```bash
# 1. Batch extract all PDFs
python cli.py --folder ./reports --workers 4

# 2. Export to CSV with confidence
python cli.py --export all_companies.csv --confidence

# 3. Check results
head -5 all_companies.csv
```

### Database Queries

```bash
# View successful extractions
sqlite3 extractions.db "
  SELECT company_name, financial_year, started_at
  FROM extraction_attempts
  WHERE success = 1
  ORDER BY company_name, financial_year;
"

# View extraction results for a company
sqlite3 extractions.db "
  SELECT er.metric_name, er.consensus_value, er.agreement_rate
  FROM extraction_results er
  JOIN extraction_attempts ea ON er.attempt_id = ea.id
  WHERE ea.company_name = 'SGL Limited'
    AND ea.financial_year = '2023';
"

# View costs
sqlite3 extractions.db "
  SELECT company_name, SUM(uc.estimated_cost) as cost
  FROM usage_costs uc
  JOIN extraction_attempts ea ON uc.attempt_id = ea.id
  GROUP BY company_name;
"
```

## Directory Structure

```
reports/
├── SGL Limited/
│   ├── report-2021.pdf
│   ├── report-2022.pdf
│   └── report-2023.pdf
├── WILLIAMSON TEA KENYA/
│   └── annual-2023.pdf
└── UNGA GROUP/
    ├── report-2022.pdf
    └── report-2023.pdf
```

**Filename Format:** `<anything>-<YYYY>.pdf` (year extracted from filename)

## CSV Output Columns

**Base (37 columns):**
- Metadata: code, Sector, SectorS, Company, Cos, Year
- Financials (18): Credit_Sales, Net_Sales, Debtors, etc.
- Derived (13): Current_ratio, ROE, ROS, Debt_ratio, etc.

**With --confidence (+18 columns):**
- Agreement rates: Credit_Sales_Confidence, Net_Sales_Confidence, etc.

## Common Options

| Option | Description | Default |
|--------|-------------|---------|
| `--workers N` | Parallel workers (batch mode) | 2 |
| `--num-samples N` | Self-consistency samples | 3 |
| `--max-pages N` | Limit pages processed | All |
| `--confidence` | Include confidence in CSV | False |
| `--output FILE` | Save JSON result | - |
| `--output-dir DIR` | Save batch results | - |

## Troubleshooting

```bash
# Check database has data
sqlite3 extractions.db "SELECT COUNT(*) FROM extraction_attempts WHERE success = 1;"

# List all companies in database
sqlite3 extractions.db "SELECT DISTINCT company_name FROM extraction_attempts ORDER BY company_name;"

# Test extraction with limited pages (faster)
python cli.py report.pdf --max-pages 10 --company "Test" --year "2023"

# Verify CSV export works
python cli.py --export test.csv --company "SGL Limited"
```

## Performance Tips

- **Workers:** Start with 2, increase to 4 if RAM allows (need ~2-3GB per worker)
- **Samples:** Use 3 for production, 1 for testing (faster)
- **Max Pages:** Use `--max-pages 20` for testing, remove for production
- **OCR Cache:** Re-running is fast (OCR cached in database)

## Documentation Files

- **BATCH_EXTRACTION_GUIDE.md** - Batch processing guide
- **CSV_EXPORT_GUIDE.md** - CSV export detailed guide
- **DATABASE_GUIDE.md** - Database schema reference
- **EXPORT_SUMMARY.md** - Implementation details

## Environment Setup

```bash
# Required environment variables (.env file)
DEEPSEEK_API_KEY=sk-...
GEMINI_API_KEY=AIza...

# Install dependencies
pip install -r requirements.txt
```

## Help

```bash
# Show all options
python cli.py --help

# Standalone CSV exporter (alternative)
python csv_exporter.py --help
```
