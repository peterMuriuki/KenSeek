# PDF Financial Data Extractor

Extract 18 financial metrics from Kenyan company annual reports using AI-powered OCR and data extraction.

## Features

- ✅ **OCR Processing** - Extract text from PDF using Google Gemini Vision API
- ✅ **AI Extraction** - Extract 18 financial metrics using DeepSeek LLM
- ✅ **Self-Consistency** - Multiple extraction samples with consensus voting
- ✅ **Batch Processing** - Process entire folders with parallelization
- ✅ **CSV Export** - Export to CSV with derived financial ratios
- ✅ **Database Storage** - SQLite database with full extraction history

## Quick Start

### 1. Prerequisites

- Python 3.8+
- DeepSeek API key (get from https://platform.deepseek.com)
- Google Gemini API key (get from https://aistudio.google.com)

### 2. Installation

```bash
# Clone or download the repository
cd ken-seek

# Install dependencies
pip install -r requirements.txt

# Create environment file
cp .env.example .env

# Edit .env and add your API keys
nano .env
```

Add to `.env`:
```
DEEPSEEK_API_KEY=your_deepseek_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here
```

### 3. Run Extraction

#### Single File Extraction

```bash
python src/cli.py report.pdf --company "SGL Limited" --year "2023"
```

#### Batch Extraction (Multiple Companies)

```bash
# Organize PDFs in this structure:
# reports/
# ├── SGL Limited/
# │   ├── report-2021.pdf
# │   ├── report-2022.pdf
# │   └── report-2023.pdf
# └── Williamson Tea/
#     └── annual-2023.pdf

# Run batch extraction
python src/cli.py --folder ./reports --workers 4
```

#### Export to CSV

```bash
# Basic export
python src/cli.py --export results.csv

# Export with confidence rates
python src/cli.py --export results.csv --confidence
```

## Project Structure

```
ken-seek/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── .env                      # API keys (create from .env.example)
├── extractions.db           # SQLite database (auto-created)
│
├── src/                     # Source code
│   ├── cli.py              # Main CLI interface
│   ├── pdf_extractor.py    # Core extraction logic
│   ├── csv_exporter.py     # CSV export functionality
│   └── database.py         # Database operations
│
├── doc/                     # Documentation
│   ├── QUICK_REFERENCE.md  # Command reference
│   ├── BATCH_EXTRACTION_GUIDE.md
│   ├── CSV_EXPORT_GUIDE.md
│   └── ... (other guides)
│
└── output/                  # Generated files
    ├── ocr/                # OCR text files
    ├── results/            # Extraction results
    └── exports/            # CSV exports
```

## Common Commands

### Single File

```bash
# Extract from one PDF
python src/cli.py report.pdf --company "Company Name" --year "2023"

# Save results to JSON
python src/cli.py report.pdf --output results.json

# Show detailed confidence metrics
python src/cli.py report.pdf --show-confidence
```

### Batch Processing

```bash
# Process all companies (default: 2 workers)
python src/cli.py --folder ./reports

# Process with 4 parallel workers
python src/cli.py --folder ./reports --workers 4

# Process specific companies only
python src/cli.py --folder ./reports --companies "SGL Limited" "Williamson Tea"

# Save batch results
python src/cli.py --folder ./reports --output-dir ./output/results
```

### CSV Export

```bash
# Export all extractions
python src/cli.py --export output/exports/results.csv

# Export with confidence rates (55 columns instead of 37)
python src/cli.py --export output/exports/results.csv --confidence

# Export specific company
python src/cli.py --export output/exports/sgl.csv --company "SGL Limited"

# Export with sector metadata
python src/cli.py --export output/exports/manufacturing.csv \
  --sector "Manufacturing" --sector-short "MFG" --code "001"
```

## Extracted Metrics

### Base Metrics (18)
1. Credit Sales
2. Net Sales
3. Debtors
4. Impaired Debts
5. Current Assets
6. Current Liabilities
7. Cash
8. Marketable Securities
9. Total Liabilities
10. Total Assets
11. EBIT
12. Interest
13. Labour Cost
14. Operating Expenses
15. R&D Cost
16. Tax
17. Equity
18. Net Income

### Derived Metrics (13)
1. Trade Credit Exposure Rate
2. Default Rate
3. Current Ratio
4. Cash Ratio
5. Debt Ratio
6. EBIT - Interest
7. DFL (Degree of Financial Leverage)
8. LCOR (Labour Cost Operating Ratio)
9. R&D Cost Ratio
10. Firm Size (ln Assets)
11. ROE (Return on Equity)
12. ROS (Return on Sales)
13. C+MS (Cash + Marketable Securities)

## How It Works

1. **OCR Phase** - PDF pages are converted to images and processed by Google Gemini Vision API to extract text
2. **Extraction Phase** - Text is sent to DeepSeek LLM (3 samples) to extract financial metrics
3. **Consensus Phase** - Multiple samples are compared and consensus is computed via majority voting
4. **Storage Phase** - Results saved to SQLite database with confidence metrics
5. **Export Phase** - Data exported to CSV with calculated financial ratios

## Output Files

### OCR Text Files
- Location: `output/ocr/`
- Format: `{filename}_ocr.txt`
- Contains: Extracted text from all PDF pages

### Extraction Results
- Location: `extractions.db` (SQLite database)
- Contains: All extraction attempts, samples, and consensus results

### CSV Exports
- Location: `output/exports/` (or custom path)
- Format: 37 base columns + 18 optional confidence columns
- Currency values formatted with commas, ratios with 4 decimals

## Configuration

### Environment Variables

```bash
DEEPSEEK_API_KEY=sk-...        # Required: DeepSeek API key
GEMINI_API_KEY=AIza...         # Required: Gemini API key
```

### Command-Line Options

```bash
python src/cli.py --help        # Show all options

Common options:
  --model deepseek-chat         # Model to use (default)
  --num-samples 3               # Self-consistency samples (default: 3)
  --max-pages 20                # Limit pages (default: all)
  --workers 4                   # Parallel workers (batch mode)
  --confidence                  # Include confidence in CSV
  --show-reasoning              # Show model reasoning
```

## Troubleshooting

### Issue: "Missing required package"
```bash
pip install -r requirements.txt
```

### Issue: "API key not found"
Check your `.env` file has:
```
DEEPSEEK_API_KEY=your_key_here
GEMINI_API_KEY=your_key_here
```

### Issue: Database errors
The database is created automatically. If corrupted:
```bash
rm extractions.db  # Delete and it will be recreated
```

### Issue: Rate limits
Reduce parallel workers:
```bash
python src/cli.py --folder ./reports --workers 1
```

## Cost Estimates

Based on typical annual reports (50-100 pages):

- **OCR (Gemini):** ~$0.01-0.02 per PDF
- **Extraction (DeepSeek, 3 samples):** ~$0.05-0.10 per PDF
- **Total:** ~$0.06-0.12 per PDF

Batch processing 10 companies × 3 years = 30 PDFs ≈ $1.80-3.60

## Documentation

See the `doc/` folder for detailed guides:

- **[QUICK_REFERENCE.md](doc/QUICK_REFERENCE.md)** - Command cheat sheet
- **[BATCH_EXTRACTION_GUIDE.md](doc/BATCH_EXTRACTION_GUIDE.md)** - Batch processing details
- **[CSV_EXPORT_GUIDE.md](doc/CSV_EXPORT_GUIDE.md)** - CSV export format and usage
- **[DATABASE_GUIDE.md](doc/DATABASE_GUIDE.md)** - Database schema and queries

## Advanced Usage

### Query Database Directly

```bash
sqlite3 extractions.db "
  SELECT company_name, financial_year, success
  FROM extraction_attempts
  WHERE success = 1
  ORDER BY company_name, financial_year;
"
```

### Use as Python Library

```python
from src.pdf_extractor import PDFFinancialExtractor

extractor = PDFFinancialExtractor(api_key="your_key")
result = extractor.extract_with_ocr_preprocessing(
    pdf_path="report.pdf",
    company_name="SGL Limited",
    financial_year="2023"
)

print(result["extracted_data"])
```

## License

See LICENSE file for details.

## Support

For issues and questions:
1. Check documentation in `doc/` folder
2. Review error messages and troubleshooting section
3. Verify API keys are correct and have sufficient quota

## Version

Current version: 1.0.0

Last updated: December 2024
