# API-Based OCR Guide

## Overview

This project now uses **DeepSeek Vision API** for OCR preprocessing instead of local models. This provides the best balance of simplicity, reliability, and cost.

## How It Works

### Two-Stage Process

**Stage 1: OCR (Page-by-Page)**
- Each PDF page is sent individually to DeepSeek Vision API
- API extracts text from the image
- Text is collected for all pages

**Stage 2: Extraction (Text-Based)**
- All OCR text is combined into one document
- Single API call to extract financial metrics
- Much smaller payload than sending images

## Benefits

### ✅ **Handles Large PDFs**
- Process 50-100+ pages without 413 errors
- Each page processed individually (small payloads)
- No file size limits

### ✅ **Simple Setup**
- No local model installation
- No GPU required
- Works immediately with your existing DeepSeek API key

### ✅ **Reliable**
- Same vision model as direct extraction
- Proven to work with DeepSeek API
- Consistent results

### ✅ **Reasonable Cost**
For a 92-page PDF:
- OCR: 92 pages × ~500 tokens/page = ~$0.05
- Extraction: 1 call × ~40,000 tokens = ~$0.04
- **Total: ~$0.09 per PDF**

## Usage

### Command Line

```bash
# Process all 92 pages with OCR mode
python pdf_extractor.py report.pdf --use-ocr

# Process specific number of pages
python pdf_extractor.py report.pdf --use-ocr --max-pages 50

# With output file
python pdf_extractor.py report.pdf --use-ocr -o results.json
```

### Interactive CLI

```bash
python cli.py

# When prompted:
# - Select your PDF file
# - Choose "Yes" for "Use OCR preprocessing?"
# - Continue with other options
```

### Python Module

```python
from pdf_extractor import PDFFinancialExtractor

extractor = PDFFinancialExtractor()

# Extract with OCR preprocessing
result = extractor.extract_with_ocr_preprocessing(
    pdf_path="large_report.pdf",
    max_pages=None,  # Process all pages
    model="deepseek-chat",
    save_ocr=True  # Saves OCR text to file for review
)

print(f"Processed {result['ocr_metadata']['num_pages']} pages")
print(f"Extracted data: {result['extracted_data']}")
```

## Output

### Console Output

```
======================================================================
OCR PREPROCESSING (DeepSeek Vision API)
======================================================================
File: report.pdf
Method: Page-by-page OCR via DeepSeek API

[OCR] Processing all 92 pages

[OCR] Page 1/92
[OCR]   Image size: 245.3 KB
[OCR]   ✓ Extracted 1,245 characters
[OCR]   Preview: ANNUAL REPORT 2023 COMPANY NAME LIMITED...

[OCR] Page 2/92
...

[OCR] ✓ OCR Complete
[OCR] Successfully processed: 92/92 pages
[OCR] Total characters extracted: 95,432
[OCR] Average per page: 1,037 chars
======================================================================

[TEXT-EXTRACTION] Sending request to DeepSeek API...
...
```

### OCR Text File

The OCR text is automatically saved to `{pdf_name}_ocr.txt`:

```
=== PAGE 1 ===
ANNUAL REPORT 2023
COMPANY NAME LIMITED

Financial Year Ended 31 December 2023

=== PAGE BREAK ===

=== PAGE 2 ===
STATEMENT OF FINANCIAL POSITION
As at 31 December 2023

Assets                           2023        2022
                                KSh'000     KSh'000
Current Assets
Cash and cash equivalents       50,000      45,000
Trade receivables               35,000      32,000
...
```

### JSON Output

```json
{
  "success": true,
  "file": "report.pdf",
  "model": "deepseek-chat",
  "extraction_method": "ocr_text",
  "extracted_data": {
    "net_sales": "480000000",
    "ebit": "75000000",
    "total_assets": "1200000000",
    ...
  },
  "ocr_metadata": {
    "num_pages": 92,
    "total_characters": 95432,
    "ocr_saved_to": "report_ocr.txt"
  },
  "usage": {
    "prompt_tokens": 48234,
    "completion_tokens": 1256,
    "total_tokens": 49490
  }
}
```

## Performance

### Small PDF (10-15 pages)
- **Direct Mode**: 1 API call, ~2 MB, ~$0.02
- **OCR Mode**: 11 API calls (10 OCR + 1 extraction), ~$0.02
- **Recommendation**: Use either (similar cost)

### Medium PDF (25-30 pages)
- **Direct Mode**: 413 Error (payload too large)
- **OCR Mode**: 26 API calls, ~$0.05
- **Recommendation**: Use OCR mode

### Large PDF (92 pages)
- **Direct Mode**: 413 Error (impossible)
- **OCR Mode**: 93 API calls, ~$0.09
- **Recommendation**: Use OCR mode (only option)

## Comparison: Direct vs OCR Mode

| Feature | Direct Image Mode | OCR Mode (API-based) |
|---------|-------------------|----------------------|
| **Max Pages** | 10-15 pages | Unlimited |
| **Cost (92 pages)** | Impossible (413 error) | ~$0.09 |
| **Setup** | None | None |
| **API Calls** | 1 | 93 (92 OCR + 1 extract) |
| **Speed** | Fast (1 call) | Slower (93 calls) |
| **Reliability** | Fails on large PDFs | Works for any size |
| **Best For** | Small PDFs only | Medium-large PDFs |

## Tips

### 1. Review OCR Text
Always check the saved OCR file to ensure quality:

```bash
# Check first 50 lines
head -n 50 report_ocr.txt

# Search for specific content
grep -i "net sales" report_ocr.txt
```

### 2. Start Small, Then Scale
Test with fewer pages first:

```bash
# Test with 10 pages
python pdf_extractor.py report.pdf --use-ocr --max-pages 10

# If results look good, process all pages
python pdf_extractor.py report.pdf --use-ocr
```

### 3. Monitor Costs
Each 92-page PDF costs ~$0.09. For 100 PDFs:
- **Cost**: ~$9
- **Time**: ~15-20 minutes per PDF (API rate limits)

### 4. Batch Processing
Process multiple PDFs efficiently:

```python
from pathlib import Path
from pdf_extractor import PDFFinancialExtractor

extractor = PDFFinancialExtractor()

for pdf_file in Path("reports").glob("*.pdf"):
    print(f"Processing {pdf_file}...")

    result = extractor.extract_with_ocr_preprocessing(
        pdf_path=str(pdf_file),
        max_pages=None,
        save_ocr=True
    )

    # Save results
    output_file = f"{pdf_file.stem}_results.json"
    import json
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"Saved to {output_file}\n")
```

## Troubleshooting

### API Rate Limits
If you hit rate limits, add delays:

```python
import time

for page in pages:
    # Process page
    time.sleep(1)  # Wait 1 second between requests
```

### OCR Text Quality Issues
If text extraction is poor:
- Check PDF quality (scanned vs digital)
- Try increasing image DPI (edit `pdf_to_images()` in code)
- For scanned documents, consider pre-processing images

### High Costs
To reduce costs:
- Process only relevant pages (e.g., pages 15-40)
- Extract specific sections manually
- Use direct mode for small PDFs (< 15 pages)

## Summary

**API-based OCR is now the recommended approach for:**
- PDFs with 15+ pages
- Any PDF that causes 413 errors
- Batch processing scenarios
- When you need reliability over speed

**Use Direct Mode for:**
- PDFs with < 15 pages
- When speed is critical
- When you want to minimize API calls

The system automatically saves OCR text for review, making it easy to verify quality and debug issues.
