# Gemini OCR Setup Guide

## Quick Start (5 Minutes)

Get started with **FREE** Gemini OCR in 3 simple steps!

### Step 1: Get Your Gemini API Key (FREE)

1. Visit: https://aistudio.google.com/app/apikey
2. Sign in with your Google account
3. Click "Create API Key"
4. Copy your API key

**Free Tier Benefits:**
- ✅ **Unlimited tokens** for gemini-2.5-flash
- ✅ No credit card required
- ✅ Perfect for testing and development
- ✅ 1500 requests per day limit (more than enough!)

### Step 2: Add API Key to .env File

```bash
# Copy the example env file
cp .env.example .env

# Edit .env and add your keys
nano .env
```

Add both keys:
```bash
DEEPSEEK_API_KEY=your_deepseek_key_here
GEMINI_API_KEY=your_gemini_key_here
```

### Step 3: Run OCR Mode

```bash
# Process your PDF with OCR preprocessing
python pdf_extractor.py report.pdf --use-ocr

# Or use the interactive CLI
python cli.py
```

That's it! You're now using Gemini OCR + DeepSeek extraction.

---

## How It Works

### Two-Stage Process

**Stage 1: Gemini OCR** (Page-by-Page)
- Each PDF page → sent to Gemini
- Gemini extracts text from image
- Text is collected for all pages

**Stage 2: DeepSeek Extraction** (Text-Based)
- All OCR text → combined into one document
- Sent to DeepSeek for financial extraction
- Returns 18 financial metrics

### Why This Combination?

| Task | Service | Why? |
|------|---------|------|
| **OCR** | Google Gemini | ✅ Supports vision/images<br>✅ Extremely cheap ($0.02 for 92 pages)<br>✅ FREE tier available<br>✅ Excellent accuracy |
| **Extraction** | DeepSeek | ✅ Best for reasoning/analysis<br>✅ Cheap text processing<br>✅ High quality extraction<br>✅ Already configured |

---

## Cost Breakdown

### For 92-Page PDF

#### Using Gemini 2.5 Flash Lite (Cheapest)

```
OCR (Gemini):
- 92 images × ~250 tokens/image = 23,000 tokens
- Input: 23,000 tokens × $0.10/1M = $0.0023
- Output: ~5,000 chars/page × 92 = ~100,000 tokens
- Output: 100,000 tokens × $0.40/1M = $0.04
- Total OCR: ~$0.042

Extraction (DeepSeek):
- Input: ~100,000 tokens × $0.14/1M = $0.014
- Output: ~2,000 tokens × $0.28/1M = $0.0006
- Total Extraction: ~$0.015

TOTAL COST: ~$0.06 per 92-page PDF
```

#### Using Gemini 2.5 Flash (FREE!)

```
OCR (Gemini):
- Unlimited FREE tokens!
- $0.00

Extraction (DeepSeek):
- ~$0.015

TOTAL COST: ~$0.02 per 92-page PDF (basically free!)
```

### Cost Comparison

| Approach | 1 PDF (92p) | 10 PDFs | 100 PDFs |
|----------|-------------|---------|----------|
| **Gemini Flash Lite + DeepSeek** | $0.06 | $0.60 | $6.00 |
| **Gemini Flash (FREE) + DeepSeek** | $0.02 | $0.20 | $2.00 |
| OpenAI GPT-4V + DeepSeek | $0.96 | $9.60 | $96.00 |
| Tesseract + DeepSeek | $0.04 | $0.40 | $4.00 |

---

## Usage Examples

### Command Line

**Basic OCR mode:**
```bash
python pdf_extractor.py report.pdf --use-ocr
```

**Limit pages to save time/cost:**
```bash
python pdf_extractor.py report.pdf --use-ocr --max-pages 50
```

**With output file:**
```bash
python pdf_extractor.py report.pdf --use-ocr -o results.json
```

**Show reasoning:**
```bash
python pdf_extractor.py report.pdf --use-ocr --show-reasoning
```

### Interactive CLI

```bash
python cli.py

# Follow prompts:
# 1. Select your PDF file
# 2. Choose "Yes" for "Use OCR preprocessing?"
# 3. Continue with other options
```

### Python Script

```python
from pdf_extractor import PDFFinancialExtractor

# Initialize with both API keys
extractor = PDFFinancialExtractor(
    api_key="your_deepseek_key",
    gemini_api_key="your_gemini_key"
)

# Extract with OCR preprocessing
result = extractor.extract_with_ocr_preprocessing(
    pdf_path="large_report.pdf",
    max_pages=None,  # Process all pages
    model="deepseek-chat",
    save_ocr=True
)

# Results
print(f"Processed {result['ocr_metadata']['num_pages']} pages")
print(f"OCR text saved to: {result['ocr_metadata']['ocr_saved_to']}")
print(f"Extracted data: {result['extracted_data']}")
```

---

## Output

### Console Output

```
======================================================================
OCR PREPROCESSING (Google Gemini 2.5 Flash Lite)
======================================================================
File: report.pdf
Model: gemini-2.5-flash-lite
Cost: ~$0.10 per 1M tokens (images + text)

[STEP 1/6] Converting PDF to images (DPI: 150)...
[STEP 1/6] ✓ Converted 92 pages to images

[OCR] Processing all 92 pages
[OCR] Estimated cost for 92 pages: ~$0.0184

[OCR] Page 1/92
    Resizing from 1240x1754 to 1085x1536
[OCR]   ✓ Extracted 1,234 characters
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

Automatically saved to `{pdf_name}_ocr.txt`:

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

---

## Features

### ✅ **Handles Large PDFs**
- Process 50-100+ pages without errors
- No 413 "Request Too Large" errors
- Each page processed individually

### ✅ **Cost-Effective**
- Gemini Flash Lite: $0.06 per 92-page PDF
- Gemini Flash (FREE): $0.02 per 92-page PDF
- 10-15x cheaper than OpenAI GPT-4V

### ✅ **Free Tier**
- Unlimited tokens with gemini-2.5-flash
- 1500 requests per day
- No credit card required

### ✅ **High Quality**
- Excellent OCR accuracy
- Handles tables and formatting
- Preserves numbers and symbols

### ✅ **Simple Setup**
- No local installation required
- Cloud-based (works anywhere)
- 5-minute setup

---

## Tips & Best Practices

### 1. Start with Free Tier

Test with gemini-2.5-flash (FREE) first:

```python
extractor = PDFFinancialExtractor()
# By default uses gemini-2.5-flash-lite

# To use the free gemini-2.5-flash:
# Edit pdf_extractor.py line 148:
# self.gemini_model = genai.GenerativeModel('gemini-2.5-flash')
```

### 2. Review OCR Quality

Always check the saved OCR file:

```bash
# View first 50 lines
head -n 50 report_ocr.txt

# Search for specific metrics
grep -i "net sales" report_ocr.txt
grep -i "total assets" report_ocr.txt
```

### 3. Start Small, Scale Up

```bash
# Test with 5 pages first
python pdf_extractor.py report.pdf --use-ocr --max-pages 5

# If results look good, process all pages
python pdf_extractor.py report.pdf --use-ocr
```

### 4. Batch Processing

Process multiple PDFs efficiently:

```python
from pathlib import Path
from pdf_extractor import PDFFinancialExtractor
import json
import time

extractor = PDFFinancialExtractor()

for pdf_file in Path("reports").glob("*.pdf"):
    print(f"Processing {pdf_file.name}...")

    result = extractor.extract_with_ocr_preprocessing(
        pdf_path=str(pdf_file),
        max_pages=None,
        save_ocr=True
    )

    # Save results
    output_file = f"{pdf_file.stem}_results.json"
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"✓ Saved to {output_file}\n")

    # Small delay to respect rate limits
    time.sleep(1)
```

### 5. Monitor Costs

Track your usage:

```bash
# Visit: https://aistudio.google.com/app/apikey
# Check "API Usage" section
```

---

## Troubleshooting

### "Gemini API not configured" Error

**Problem:** GEMINI_API_KEY not found

**Solution:**
```bash
# Check your .env file
cat .env

# Ensure you have:
GEMINI_API_KEY=your_actual_key_here

# Verify the file is in the same directory as pdf_extractor.py
ls -la .env
```

### Rate Limit Errors

**Problem:** "Resource exhausted" or "Quota exceeded"

**Solution:**
```python
# Add delays between requests
import time

for page in pages:
    # Process page
    time.sleep(1)  # Wait 1 second between requests
```

Free tier limits:
- 1500 requests per day
- 1 million tokens per minute

### Poor OCR Quality

**Problem:** Text extraction is inaccurate

**Solutions:**
1. Check PDF quality (scanned vs digital)
2. Increase image resolution:
   ```python
   images = self.pdf_to_images(pdf_path, dpi=200)  # Default is 150
   ```
3. Review OCR text file to identify specific issues
4. For handwritten documents, OCR may struggle

### High Costs

**Problem:** Costs higher than expected

**Solutions:**
1. Use gemini-2.5-flash (FREE tier)
2. Process only relevant pages:
   ```bash
   # Only process financial statement pages (e.g., 15-40)
   python pdf_extractor.py report.pdf --use-ocr --max-pages 25
   ```
3. Check usage at https://aistudio.google.com/app/apikey

---

## Model Comparison

| Model | Input Cost | Output Cost | Free Tier | Best For |
|-------|-----------|-------------|-----------|----------|
| **gemini-2.5-flash** | FREE | FREE | ✅ Unlimited | Testing, development |
| **gemini-2.5-flash-lite** | $0.10/1M | $0.40/1M | ❌ | Production (cheapest) |
| gemini-2.5-flash (paid) | $0.30/1M | $2.50/1M | ❌ | Higher accuracy needs |
| gemini-2.0-flash | $0.10/1M | $0.40/1M | ❌ | Alternative option |

**Recommendation:** Start with **gemini-2.5-flash** (FREE), then switch to **gemini-2.5-flash-lite** for production.

---

## Summary

**Use Gemini OCR when:**
- ✅ PDF has 15+ pages
- ✅ Want cloud-based solution
- ✅ Need cost-effective OCR
- ✅ Want to use free tier

**Use Direct Mode (no OCR) when:**
- ✅ PDF has < 15 pages
- ✅ Don't want to set up Gemini
- ✅ Speed is critical (but limited to small PDFs)

**Commands:**
```bash
# OCR mode (Gemini + DeepSeek)
python pdf_extractor.py report.pdf --use-ocr

# Direct mode (DeepSeek only, small PDFs)
python pdf_extractor.py report.pdf
```

---

## Getting Help

1. Check this guide's Troubleshooting section
2. Review the OCR text file (`*_ocr.txt`) for quality issues
3. See `TROUBLESHOOTING.md` for general issues
4. Check Gemini API documentation: https://ai.google.dev/docs

**Happy extracting! 🚀**
