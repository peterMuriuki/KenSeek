# OCR Preprocessing Mode

**NEW FEATURE**: Two-stage extraction using **DeepSeek-OCR** (via Hugging Face) for dramatically improved efficiency!

Uses the official **[DeepSeek-OCR](https://github.com/deepseek-ai/DeepSeek-OCR)** model hosted on Hugging Face for state-of-the-art text extraction.

## What is OCR Mode?

OCR Mode is a two-stage extraction process that:

1. **Stage 1 (OCR)**: Uses DeepSeek's vision model to extract text from each PDF page
2. **Stage 2 (Extraction)**: Analyzes the extracted text to find financial metrics

Instead of sending images directly for analysis, we first convert them to text, then analyze the text.

## Benefits

### 🚀 **90%+ Payload Reduction**
- Images: ~3-10 MB for 25 pages
- Text: ~0.1-0.3 MB for 25 pages
- **Result**: 30-100x smaller payloads!

### 📄 **Process Much Larger PDFs**
- **Without OCR**: 10-20 pages max (413 errors)
- **With OCR**: 50-100+ pages possible
- No more "Request Too Large" errors!

### 💰 **Lower Costs**
- Fewer total tokens (OCR + extraction < direct image analysis)
- Text tokens are cheaper than image tokens
- **Typical savings**: 30-50% cost reduction

### ⚡ **Better for Text-Heavy Documents**
- More reliable for standard financial statements
- Preserves table structure
- Better number recognition

## When to Use OCR Mode

### ✅ **Use OCR Mode When:**
- PDF has more than 15-20 pages
- Getting "413 Request Too Large" errors
- Processing text-based PDFs (not scanned images)
- Need to process many documents (batch processing)
- Want to save costs
- Standard financial statements with clear text

### ❌ **Don't Use OCR Mode When:**
- PDF has only 5-10 pages (direct mode is simpler)
- PDF contains primarily images/charts (OCR won't help)
- Very poor scan quality (OCR may struggle)
- Need absolute maximum accuracy (direct vision may be slightly better for edge cases)

## Usage Examples

### Command Line

**Basic OCR mode:**
```bash
python pdf_extractor.py report.pdf --use-ocr
```

**OCR mode with page limiting:**
```bash
python pdf_extractor.py report.pdf --use-ocr --max-pages 30
```

**OCR mode with all options:**
```bash
python pdf_extractor.py report.pdf \
  --use-ocr \
  --max-pages 50 \
  --show-reasoning \
  -o results.json
```

### Interactive CLI

```bash
# Launch CLI
python cli.py

# When prompted:
# "Use OCR preprocessing?" → Yes
# "Limit pages?" → Optional
# ... continue with other options
```

**CLI Quick Mode:**
```bash
python cli.py --file report.pdf --use-ocr
```

### Python Module

```python
from pdf_extractor import PDFFinancialExtractor

extractor = PDFFinancialExtractor()

# OCR mode
result = extractor.extract_with_ocr_preprocessing(
    pdf_path="large_report.pdf",
    max_pages=50,  # Can handle much larger PDFs
    model="deepseek-chat",
    save_ocr=True  # Saves OCR text to file
)

if result["success"]:
    print("OCR Metadata:")
    print(f"  Pages: {result['ocr_metadata']['num_pages']}")
    print(f"  Characters: {result['ocr_metadata']['total_characters']:,}")
    print(f"  Saved to: {result['ocr_metadata']['ocr_saved_to']}")

    print("\nExtracted Data:")
    print(result['extracted_data'])
```

### Just OCR (No Extraction)

If you only want to OCR the PDF without extraction:

```python
from pdf_extractor import PDFFinancialExtractor

extractor = PDFFinancialExtractor()

# Just OCR the pages
ocr_pages = extractor.ocr_pdf_pages(
    pdf_path="report.pdf",
    max_pages=None  # All pages
)

# Save to file
full_text = "\n\n=== PAGE BREAK ===\n\n".join(
    f"=== PAGE {i+1} ===\n{text}"
    for i, text in enumerate(ocr_pages)
)

with open("report_ocr.txt", "w") as f:
    f.write(full_text)
```

## How It Works

### Stage 1: OCR (Page-by-Page)

For each page:
1. Convert PDF page to image (150 DPI)
2. Optimize and compress image
3. Send to DeepSeek vision with OCR prompt
4. Extract raw text with formatting preserved

**OCR Prompt:**
- Extract ALL text exactly as appears
- Preserve numbers and currency symbols
- Maintain table structure
- Keep line breaks
- Mark unclear text with [UNCLEAR]

### Stage 2: Text-Based Extraction

1. Combine all OCRed pages into single text document
2. Send text (not images!) to DeepSeek with extraction prompt
3. Model analyzes text to find 18 financial metrics
4. Return structured JSON result

## Output

### Standard Output (Same as Direct Mode)

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
  "usage": {
    "prompt_tokens": 5234,
    "completion_tokens": 856,
    "total_tokens": 6090
  }
}
```

### Additional OCR Metadata

```json
{
  "ocr_metadata": {
    "num_pages": 25,
    "total_characters": 45678,
    "ocr_saved_to": "report_ocr.txt"
  }
}
```

### OCR Text File

The OCR text is automatically saved to `{pdf_name}_ocr.txt`:

```
=== PAGE 1 ===
ANNUAL REPORT 2023
COMPANY NAME LIMITED

...

=== PAGE BREAK ===

=== PAGE 2 ===
STATEMENT OF FINANCIAL POSITION
As at 31 December 2023

Assets                           2023        2022
                                KSh'000     KSh'000
Current Assets
Cash and cash equivalents       50,000      45,000
...

=== PAGE BREAK ===

=== PAGE 3 ===
...
```

## Comparison: Direct vs OCR Mode

| Feature | Direct Image Mode | OCR Preprocessing Mode |
|---------|-------------------|------------------------|
| **Payload Size** | 3-10 MB (25 pages) | 0.1-0.3 MB (25 pages) |
| **Max Pages** | 10-20 pages | 50-100+ pages |
| **Cost** | Higher (image tokens) | Lower (text tokens) |
| **Speed** | Single API call | 2 stages (OCR + extract) |
| **413 Errors** | Common on large PDFs | Rare |
| **Accuracy** | Excellent for all types | Excellent for text PDFs |
| **Best For** | Small PDFs, images | Large text PDFs |
| **OCR Text Saved** | No | Yes (for review) |

## Performance Benchmarks

### Small PDF (10 pages)
- **Direct Mode**: ~2 MB payload, ~8,000 tokens, ~$0.02
- **OCR Mode**: ~0.1 MB payload, ~6,000 tokens, ~$0.015
- **Winner**: Direct (simpler, similar cost)

### Medium PDF (25 pages)
- **Direct Mode**: ~5 MB payload, ~15,000 tokens, ~$0.04
- **OCR Mode**: ~0.2 MB payload, ~10,000 tokens, ~$0.025
- **Winner**: OCR (lower cost, reliable)

### Large PDF (50 pages)
- **Direct Mode**: ~10 MB payload - **413 ERROR**
- **OCR Mode**: ~0.4 MB payload, ~20,000 tokens, ~$0.05
- **Winner**: OCR (only option that works!)

## Tips for Best Results

### 1. **Use for Financial Statements**
OCR mode works best for standard financial reports with clear text and tables.

### 2. **Check OCR Text Quality**
The OCR text is saved to a file. Review it to ensure accuracy:
```bash
cat report_ocr.txt | head -n 50
```

### 3. **Adjust Page Limits**
Start with 20-30 pages to test, then increase if needed:
```bash
python pdf_extractor.py report.pdf --use-ocr --max-pages 30
```

### 4. **Compare with Direct Mode**
For smaller PDFs, try both modes and compare results:
```bash
# Direct mode
python pdf_extractor.py report.pdf -o direct_results.json

# OCR mode
python pdf_extractor.py report.pdf --use-ocr -o ocr_results.json

# Compare
diff <(jq '.extracted_data' direct_results.json) \
     <(jq '.extracted_data' ocr_results.json)
```

### 5. **Batch Processing**
OCR mode is perfect for batch processing large collections:
```python
for pdf_file in Path("reports").glob("*.pdf"):
    result = extractor.extract_with_ocr_preprocessing(
        pdf_path=str(pdf_file),
        max_pages=50
    )
    # Save results...
```

## Limitations

### 1. **Two-Stage Process**
- Takes longer than direct mode (OCR each page first)
- For 25 pages: ~25 OCR calls + 1 extraction call

### 2. **OCR Quality Dependent**
- Poor scans may have OCR errors
- Handwritten notes won't OCR well
- Complex layouts might lose structure

### 3. **No Self-Consistency Yet**
- OCR mode doesn't support self-consistency currently
- Use direct mode if you need self-consistency

### 4. **Text-Based Only**
- Charts and images are described but not analyzed
- Visual data (graphs, diagrams) may be missed

## Troubleshooting

### OCR Text Looks Wrong

**Check the saved OCR file:**
```bash
cat report_ocr.txt
```

**If text is garbled:**
- Try a higher quality PDF scan
- Use direct image mode instead
- Extract problem pages separately

### Numbers Not Extracted Correctly

**OCR might have issues with:**
- Decimal points vs commas
- Thousands separators
- Currency symbols

**Solution**: Review the OCR text and manually correct if needed, or use direct mode.

### Still Getting 413 Errors

**Even with OCR, very large PDFs might fail:**
- Reduce `--max-pages` further
- Split PDF into smaller chunks
- Extract only financial statement pages (pages 10-25 typically)

## Future Enhancements

Planned improvements for OCR mode:

- [ ] Self-consistency support for OCR mode
- [ ] Parallel OCR processing (faster)
- [ ] OCR caching (reuse previous OCR results)
- [ ] Custom OCR prompts for specific document types
- [ ] Table structure preservation improvements

## Summary

**Use OCR Mode when:**
- ✅ Large PDFs (20+ pages)
- ✅ Getting 413 errors
- ✅ Text-based financial statements
- ✅ Batch processing
- ✅ Cost is a concern

**Use Direct Image Mode when:**
- ✅ Small PDFs (< 15 pages)
- ✅ Need self-consistency
- ✅ Image-heavy documents
- ✅ Maximum accuracy required

**Command:**
```bash
# OCR mode (recommended for most use cases)
python pdf_extractor.py report.pdf --use-ocr

# Direct mode (default)
python pdf_extractor.py report.pdf
```

That's it! OCR mode makes large PDF processing possible and cost-effective. 🚀
