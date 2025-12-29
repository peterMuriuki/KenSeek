# Project Structure

## Overview

This project is organized into clear folders for code, documentation, and output files.

## Directory Layout

```
ken-seek/
├── README.md                   # Quick start guide
├── requirements.txt            # Python dependencies
├── .env.example               # API key template
├── .env                       # Your API keys (gitignored)
├── extractions.db            # SQLite database
│
├── src/                      # Source code
│   ├── cli.py               # Command-line interface
│   ├── pdf_extractor.py     # PDF extraction engine
│   ├── csv_exporter.py      # CSV export module
│   └── database.py          # Database operations
│
├── doc/                      # Documentation
│   ├── QUICK_REFERENCE.md   # Command cheat sheet
│   ├── BATCH_EXTRACTION_GUIDE.md
│   ├── CSV_EXPORT_GUIDE.md
│   ├── DATABASE_GUIDE.md
│   └── ... (other guides)
│
└── output/                   # Generated files
    ├── ocr/                 # OCR text files (*.txt)
    ├── results/             # Extraction JSON files
    └── exports/             # CSV export files
```

## Source Code (src/)

### cli.py
Main entry point for all operations. Provides command-line interface with three modes:
- Single file extraction
- Batch folder extraction
- CSV export

**Usage:**
```bash
python src/cli.py --help
```

### pdf_extractor.py
Core extraction engine containing:
- PDF to image conversion
- OCR processing (Google Gemini)
- Financial data extraction (DeepSeek)
- Self-consistency voting
- Database storage

**Key Classes:**
- `PDFFinancialExtractor` - Main extractor class
- `ExtractionJob` - Batch processing job

### csv_exporter.py
CSV export functionality:
- Reads from database
- Calculates derived metrics
- Formats currency and ratios
- Exports to CSV with optional confidence columns

**Standalone Usage:**
```bash
python src/csv_exporter.py --output results.csv --confidence
```

### database.py
Database operations:
- SQLite database management
- Extraction attempt tracking
- Sample storage
- Results with confidence metrics

**Key Classes:**
- `ExtractionDatabase` - Database interface

## Documentation (doc/)

All guides and documentation files:

- **QUICK_REFERENCE.md** - Command cheat sheet
- **BATCH_EXTRACTION_GUIDE.md** - Batch processing details
- **CSV_EXPORT_GUIDE.md** - CSV format and export options
- **DATABASE_GUIDE.md** - Database schema and queries
- **CLI_REFACTORING_SUMMARY.md** - CLI refactoring notes
- **AGREEMENT_RATE_FIX.md** - Confidence rate implementation
- **EXPORT_SUMMARY.md** - Export feature details
- And more...

## Output Files (output/)

### output/ocr/
OCR text files extracted from PDFs.

**Format:** `{filename}_ocr.txt`

**Example:**
```
output/ocr/SGL-Annual-Report-2023_ocr.txt
```

### output/results/
Individual extraction result files (batch mode).

**Format:** `{company}_{year}.json`

**Example:**
```
output/results/SGL Limited_2023.json
```

### output/exports/
CSV export files.

**Example:**
```
output/exports/all_companies.csv
output/exports/manufacturing_sector.csv
```

## Data Files

### extractions.db
SQLite database containing:
- PDFs processed
- OCR cache
- Extraction attempts
- Extraction samples
- Consensus results
- Usage costs

**Location:** Root directory

**Query:**
```bash
sqlite3 extractions.db "SELECT * FROM extraction_attempts LIMIT 5;"
```

### .env
Environment variables for API keys.

**Create from template:**
```bash
cp .env.example .env
nano .env  # Add your keys
```

**Contents:**
```
DEEPSEEK_API_KEY=your_key_here
GEMINI_API_KEY=your_key_here
```

## File Naming Conventions

### Source Code
- Snake case: `pdf_extractor.py`, `csv_exporter.py`
- Clear, descriptive names
- Single responsibility per module

### Documentation
- ALL_CAPS with underscores: `QUICK_REFERENCE.md`
- Descriptive titles
- Grouped by topic

### Output Files
- OCR: `{pdf_name}_ocr.txt`
- Results: `{company}_{year}.json`
- Exports: Custom names (user-defined)

## Import Structure

Since all code is in `src/`, imports work as:

```python
# From within src/
from database import ExtractionDatabase
from pdf_extractor import PDFFinancialExtractor

# From project root
from src.database import ExtractionDatabase
from src.pdf_extractor import PDFFinancialExtractor
```

## Running from Different Locations

### From Project Root (Recommended)
```bash
python src/cli.py report.pdf
```

### From src/ Directory
```bash
cd src
python cli.py ../report.pdf
```

## Path References

All paths in the code are relative to project root:

- Database: `./extractions.db`
- OCR output: `output/ocr/`
- Results: `output/results/`
- Exports: `output/exports/`

## .gitignore

The following are typically ignored:
- `.env` (contains API keys)
- `*.pyc`, `__pycache__/` (Python cache)
- `extractions.db` (database)
- `output/` (generated files)
- `.claude/` (Claude Code workspace)

## Adding New Modules

To add new functionality:

1. Create module in `src/`
2. Import in `cli.py` if needed
3. Add documentation in `doc/`
4. Update README.md if user-facing

Example:
```bash
# Create new module
touch src/new_feature.py

# Add documentation
touch doc/NEW_FEATURE_GUIDE.md

# Update README if needed
nano README.md
```

## Best Practices

1. **Code:** Keep in `src/`
2. **Documentation:** Keep in `doc/`
3. **Output:** Keep in `output/`
4. **Database:** Keep at root level
5. **Configuration:** Use `.env` for secrets
6. **Imports:** Use relative imports within `src/`
7. **Paths:** Make relative to project root

## Quick Commands Reference

```bash
# Setup
pip install -r requirements.txt
cp .env.example .env

# Run
python src/cli.py report.pdf
python src/cli.py --folder ./reports
python src/cli.py --export output/exports/results.csv

# View Documentation
cat README.md
ls doc/

# Query Database
sqlite3 extractions.db "SELECT * FROM extraction_attempts;"

# Clean Output
rm -rf output/ocr/* output/results/* output/exports/*
```

## Version Control

Recommended .gitignore:
```
.env
*.pyc
__pycache__/
extractions.db
output/
.claude/
```

Always committed:
```
README.md
requirements.txt
.env.example
src/
doc/
```

## Summary

This structure provides:
- ✅ Clear separation of concerns
- ✅ Easy to navigate
- ✅ Simple to understand
- ✅ Scalable and maintainable
- ✅ Well-documented
