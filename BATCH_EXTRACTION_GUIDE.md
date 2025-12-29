# Batch Extraction Guide

This guide explains how to use the batch processing features for extracting financial data from multiple PDFs in parallel.

## Directory Structure

Organize your PDFs in the following structure:

```
reports/
├── SGL Limited/
│   ├── SGL-Annual-Report-2021.pdf
│   ├── SGL-Annual-Report-2022.pdf
│   └── SGL-Annual-Report-2023.pdf
├── Company B/
│   ├── report-2022.pdf
│   └── report-2023.pdf
└── Another Company/
    └── financial-2023.pdf
```

### Requirements:
1. **Root folder** contains subfolders for each company
2. **Company folders** are named after the company (e.g., "SGL Limited")
3. **PDF files** must follow the naming pattern: `<anything>-<YYYY>.pdf`
   - Examples: `report-2023.pdf`, `SGL-Annual-Report-2023.pdf`, `abc123-2022.pdf`
   - The year is extracted from the filename (4 digits after the last dash)

## Usage Examples

### 1. Extract All Companies and Years

Process all PDFs in the entire folder structure:

```bash
python pdf_extractor.py --folder ./reports --workers 4
```

- `--folder`: Path to root folder containing company subfolders
- `--workers`: Number of parallel workers (recommended: 2-4)

### 2. Extract Specific Companies

Process only selected companies:

```bash
python pdf_extractor.py --folder ./reports \
  --companies "SGL Limited" "Company B" \
  --workers 2
```

### 3. Save Individual Results

Save each extraction to a separate JSON file:

```bash
python pdf_extractor.py --folder ./reports \
  --output-dir ./results \
  --workers 3
```

This creates:
```
results/
├── SGL Limited_2021.json
├── SGL Limited_2022.json
├── SGL Limited_2023.json
├── Company B_2022.json
└── Company B_2023.json
```

### 4. Save Batch Summary

Get a summary of all extractions:

```bash
python pdf_extractor.py --folder ./reports \
  --batch-summary ./batch_summary.json \
  --workers 4
```

### 5. Limit Processing

Process only first few pages (faster, for testing):

```bash
python pdf_extractor.py --folder ./reports \
  --max-pages 10 \
  --workers 2
```

### 6. Adjust Extraction Quality

Control self-consistency samples (higher = more accurate but slower):

```bash
python pdf_extractor.py --folder ./reports \
  --num-samples 5 \
  --workers 2
```

## Single File Mode (Still Supported)

Extract a single PDF file (original functionality):

```bash
# Basic extraction
python pdf_extractor.py report.pdf \
  --company "SGL Limited" \
  --year "2023"

# With OCR (recommended)
python pdf_extractor.py report.pdf \
  --use-ocr \
  --company "SGL Limited" \
  --year "2023"

# Save results
python pdf_extractor.py report.pdf \
  --company "SGL Limited" \
  --year "2023" \
  --output results.json
```

## Command-Line Arguments

### Mode Selection
- `pdf_path`: Single PDF file path (single file mode)
- `--folder`: Root folder path (batch mode)

**Note:** Cannot use both simultaneously.

### Batch Mode Options
- `--companies`: Filter specific companies (space-separated list)
- `--workers`: Max parallel workers (default: 2, recommended: 2-4)
- `--output-dir`: Directory to save individual results
- `--batch-summary`: File path for batch summary JSON

### Extraction Options
- `--model`: DeepSeek model (default: "deepseek-chat")
- `--num-samples`: Self-consistency samples (default: 3)
- `--max-pages`: Limit pages per PDF (default: all)
- `--company`: Company name (single file mode only)
- `--year`: Financial year (single file mode only)

### Output Options
- `--output`: Output JSON file (single file mode)
- `--show-reasoning`: Show model's reasoning
- `--show-confidence`: Show detailed confidence metrics

## Performance Tuning

### Worker Configuration

The `--workers` parameter controls parallelization:

- **2 workers** (default): Safe for most systems, moderate speed
- **3-4 workers**: Faster, requires more RAM and API quota
- **1 worker**: Sequential processing, slowest but most reliable

**Recommendations:**
- Start with 2 workers
- Increase to 3-4 if you have sufficient RAM (8GB+ per worker)
- Monitor API rate limits - too many workers may trigger limits

### Self-Consistency Samples

The `--num-samples` parameter controls extraction quality:

- **1 sample**: Fastest, no consensus validation
- **3 samples** (default): Good balance of speed and accuracy
- **5 samples**: Higher accuracy, slower, recommended for critical data

### Memory Considerations

Each worker processes one PDF at a time:
- OCR stage: ~500MB-2GB per PDF (depends on pages/resolution)
- Extraction stage: ~200MB-500MB per PDF
- Total: Plan for 2-3GB RAM per worker

## Database Storage

All extractions are automatically stored in SQLite database (`extractions.db`):

### Query Examples

```sql
-- Get all extractions for a company
SELECT * FROM extraction_attempts
WHERE company_name = 'SGL Limited';

-- Get successful extractions by year
SELECT company_name, financial_year, started_at
FROM extraction_attempts
WHERE success = 1
ORDER BY financial_year DESC;

-- Get extraction results for a specific company/year
SELECT er.metric_name, er.consensus_value, er.confidence
FROM extraction_results er
JOIN extraction_attempts ea ON er.attempt_id = ea.id
WHERE ea.company_name = 'SGL Limited'
  AND ea.financial_year = '2023';

-- View total costs by company
SELECT ea.company_name, SUM(uc.estimated_cost) as total_cost
FROM usage_costs uc
JOIN extraction_attempts ea ON uc.attempt_id = ea.id
GROUP BY ea.company_name;
```

## Cost Estimation

Based on typical annual reports (50-100 pages):

**Per PDF:**
- OCR (Gemini): ~$0.01-0.02
- Extraction (DeepSeek, 3 samples): ~$0.05-0.10
- **Total: ~$0.06-0.12 per PDF**

**Batch Processing:**
- 10 companies × 3 years = 30 PDFs
- Estimated cost: $1.80-3.60
- Processing time: ~30-60 minutes (with 4 workers)

## Troubleshooting

### "No extraction jobs found"
- Check folder structure (subfolders = companies)
- Verify PDF filenames contain `-YYYY` pattern
- Use `--companies` to filter specific folders

### "Rate limit exceeded"
- Reduce `--workers` to 1 or 2
- Add delays between batches
- Check API quota on DeepSeek/Gemini dashboard

### Memory errors
- Reduce `--workers` to 1 or 2
- Use `--max-pages` to limit pages processed
- Process companies in smaller batches

### Database locked errors
- SQLite has limited concurrent write support
- Reduce `--workers` to 1-2
- Extractions are cached, so re-running is safe

## Example Workflow

```bash
# Step 1: Test with one company first
python pdf_extractor.py --folder ./reports \
  --companies "SGL Limited" \
  --workers 1 \
  --max-pages 10

# Step 2: Process all companies (dry run with page limit)
python pdf_extractor.py --folder ./reports \
  --workers 2 \
  --max-pages 20

# Step 3: Full processing with results saved
python pdf_extractor.py --folder ./reports \
  --workers 4 \
  --output-dir ./results \
  --batch-summary ./summary.json

# Step 4: Query database for results
sqlite3 extractions.db "
  SELECT company_name, financial_year, success
  FROM extraction_attempts
  ORDER BY company_name, financial_year;
"
```

## Notes

- **OCR is always used** in batch mode for best results
- **Company names** are automatically extracted from folder names
- **Financial years** are automatically extracted from filenames
- **All extractions are cached** in the database
- **Re-running is safe** - cached OCR will be reused
- **Results are saved** per extraction in the database
