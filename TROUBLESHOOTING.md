# Troubleshooting Guide

This guide helps resolve common issues when using the PDF Financial Data Extractor.

## Common Errors

### 413 Request Entity Too Large

**Error message:**
```
✗ Extraction Failed
Error: 413 Request Entity Too Large
```

**What it means:**
The PDF images are too large to send to the DeepSeek API in a single request.

**Solutions (in order of preference):**

1. **Limit pages to financial statements only:**
   ```bash
   # Process only first 10 pages
   python pdf_extractor.py report.pdf --max-pages 10

   # CLI interactive mode will ask about page limiting
   python cli.py
   ```

2. **Extract specific pages from your PDF:**
   - Use a PDF tool to extract only the financial statement pages (usually 3-10 pages)
   - Process this smaller PDF instead

3. **Split large PDFs:**
   ```bash
   # If you have a 100-page report, split it into smaller files
   # Process each section separately
   python pdf_extractor.py report_part1.pdf  # Pages 1-20
   python pdf_extractor.py report_part2.pdf  # Pages 21-40
   ```

**What we've already optimized:**
- DPI reduced from 200 to 150 (smaller images)
- JPEG compression at 85% quality (vs uncompressed PNG)
- Auto-resize images to max 2048px dimension
- All images optimized before encoding

### 401 Unauthorized

**Error message:**
```
[ERROR] 401 Unauthorized
```

**What it means:**
Your DeepSeek API key is missing, incorrect, or inactive.

**Solutions:**

1. **Check your .env file exists:**
   ```bash
   ls -la .env
   ```

2. **Verify the API key is set:**
   ```bash
   cat .env
   # Should show: DEEPSEEK_API_KEY=sk-...
   ```

3. **Get a new API key:**
   - Visit: https://platform.deepseek.com/api_keys
   - Create or copy your API key
   - Update .env file:
     ```
     DEEPSEEK_API_KEY=your_new_api_key_here
     ```

4. **Test with direct API key:**
   ```bash
   python pdf_extractor.py report.pdf --api-key sk-your-key-here
   ```

### 429 Rate Limit Exceeded

**Error message:**
```
[ERROR] 429 Rate limit exceeded
```

**What it means:**
You've made too many requests to the API in a short time.

**Solutions:**

1. **Wait and retry:**
   - Wait 1-5 minutes
   - Try again

2. **Check your API usage:**
   - Visit: https://platform.deepseek.com/usage
   - See if you've hit your quota

3. **For self-consistency mode:**
   ```bash
   # Reduce samples to minimize API calls
   python pdf_extractor.py report.pdf --self-consistency --num-samples 3

   # Instead of 5 or 7 samples
   ```

### JSON Parsing Errors

**Error message:**
```
[ERROR] Failed to parse JSON
```

**What it means:**
The model's response didn't contain valid JSON.

**Solutions:**

1. **Check the reasoning:**
   ```bash
   python pdf_extractor.py report.pdf --show-reasoning
   ```
   Look at what the model returned

2. **Try again:**
   - Sometimes this is a one-time error
   - Just re-run the command

3. **Use self-consistency:**
   ```bash
   python pdf_extractor.py report.pdf --self-consistency
   ```
   This runs multiple times and can overcome occasional parsing failures

### Poppler Not Found

**Error message:**
```
pdftoppm: command not found
```

**What it means:**
The poppler-utils package isn't installed.

**Solutions:**

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install poppler-utils
```

**macOS:**
```bash
brew install poppler
```

**Windows:**
- Download from: http://blog.alivate.com.au/poppler-windows/
- Add to system PATH

**Verify installation:**
```bash
pdftoppm -v
```

## Understanding the Logs

### Normal Extraction Flow

```
[STEP 1/6] Converting PDF to images (DPI: 150)...
[STEP 1/6] ✓ Converted 25 pages to images
[STEP 1/6] Total pixels: 150.5M pixels

[STEP 2/6] Processing all 25 pages

[STEP 3/6] Optimizing and encoding images...
[STEP 3/6] Processing page 1/25
[STEP 3/6]   Page 1: 125.3 KB
[STEP 3/6] Processing page 2/25
[STEP 3/6]   Page 2: 138.7 KB
...
[STEP 3/6] ✓ Total payload size: 3250.5 KB (3.17 MB)

[STEP 4/6] Sending request to DeepSeek API...
[STEP 4/6] Request details:
[STEP 4/6]   - Pages: 25
[STEP 4/6]   - Payload size: 3.17 MB
[STEP 4/6]   - Model: deepseek-chat
[STEP 4/6] ✓ Received response from DeepSeek
[STEP 4/6] Response length: 2543 characters

[STEP 5/6] Extracting response content...

[STEP 6/6] Parsing JSON from response...
[STEP 6/6] ✓ Successfully parsed 18 metrics

======================================================================
EXTRACTION COMPLETE
======================================================================
```

### What Each Step Means

- **STEP 1**: Converting PDF pages to images
  - Shows total pages and pixel count
  - Higher pixels = larger payload

- **STEP 2**: Page limiting (if applicable)
  - Shows how many pages will be processed

- **STEP 3**: Image optimization
  - Each page is resized, compressed, and encoded
  - Shows individual page sizes
  - **Total payload size is critical**: Should be < 10 MB

- **STEP 4**: API request
  - Sending data to DeepSeek
  - This is where 413 errors occur if payload is too large

- **STEP 5**: Response extraction
  - Getting the model's response

- **STEP 6**: JSON parsing
  - Extracting the 18 financial metrics from the response

### Warning Messages

**Large Payload Warning:**
```
[WARNING] Payload is large (12.5 MB). Consider using --max-pages to reduce size.
```
Action: Reduce pages with `--max-pages 10`

**Image Resize:**
```
    Resizing from 3508x2480 to 2048x1447
```
This is normal - images are automatically optimized

## Performance Tips

### Reduce Processing Time

1. **Limit pages:**
   ```bash
   # Only process financial statement pages
   python pdf_extractor.py report.pdf --max-pages 15
   ```

2. **Use appropriate model:**
   ```bash
   # deepseek-chat is faster and cheaper
   python pdf_extractor.py report.pdf --model deepseek-chat
   ```

### Reduce Cost

1. **Skip self-consistency for exploratory work:**
   ```bash
   # Single extraction (1x cost)
   python pdf_extractor.py report.pdf

   # vs self-consistency (3-5x cost)
   python pdf_extractor.py report.pdf --self-consistency
   ```

2. **Use fewer samples:**
   ```bash
   # 3 samples instead of 5 or 7
   python pdf_extractor.py report.pdf --self-consistency --num-samples 3
   ```

3. **Process only necessary pages:**
   - Extract specific pages from PDF first
   - Don't send cover pages, appendices, etc.

### Improve Accuracy

1. **Use self-consistency for critical reports:**
   ```bash
   python pdf_extractor.py report.pdf --self-consistency --num-samples 5
   ```

2. **Review low-confidence metrics:**
   ```bash
   python pdf_extractor.py report.pdf --self-consistency --show-confidence
   ```

3. **Check reasoning for unclear values:**
   ```bash
   python pdf_extractor.py report.pdf --show-reasoning
   ```

## Getting Help

### Collect Debug Information

When reporting issues, include:

1. **Full error output:**
   ```bash
   python pdf_extractor.py report.pdf 2>&1 | tee error_log.txt
   ```

2. **PDF details:**
   ```bash
   pdfinfo report.pdf
   # Shows number of pages, size, etc.
   ```

3. **Package versions:**
   ```bash
   pip list | grep -E "(openai|pdf2image|Pillow|rich|questionary)"
   ```

4. **Payload size from logs:**
   Look for the line:
   ```
   [STEP 3/6] ✓ Total payload size: X.XX MB
   ```

### Common Questions

**Q: How many pages can I process?**
A: It depends on the PDF quality. Generally:
- Simple text PDFs: 30-50 pages
- Scanned/image PDFs: 10-20 pages
- High-resolution scans: 5-10 pages

**Q: Why is my payload so large?**
A: Your PDF might have:
- High-resolution scans
- Color images
- Many pages
Try reducing with `--max-pages`

**Q: Can I increase image quality?**
A: Currently optimized for API limits. Contact if you need custom settings.

**Q: Is there a file size limit?**
A: The limit is on the encoded payload (images as base64), not the PDF file size. A 50 MB PDF might be fine if it's mostly text, but a 5 MB PDF with high-res scans might be too large.

## Still Having Issues?

1. Check the main [README.md](README.md) for general usage
2. Review [EXAMPLES.md](EXAMPLES.md) for usage examples
3. Try the interactive CLI: `python cli.py` (more user-friendly)
4. Open an issue with debug information collected above
