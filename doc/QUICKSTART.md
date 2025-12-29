# Quick Start Guide

Get up and running with the PDF Financial Data Extractor in 5 minutes!

## Step 1: Install Dependencies

```bash
# Install Python packages
pip install -r requirements.txt

# Install poppler (Ubuntu/Debian)
sudo apt-get update && sudo apt-get install -y poppler-utils

# Or on macOS
brew install poppler
```

## Step 2: Configure API Key

```bash
# Copy the example file
cp .env.example .env

# Edit .env and add your DeepSeek API key
nano .env  # or use any text editor
```

Add this line to `.env`:
```
DEEPSEEK_API_KEY=sk-your-actual-api-key-here
```

Get your API key from: https://platform.deepseek.com/api_keys

## Step 3: Run the Interactive CLI

```bash
python cli.py
```

That's it! The interactive interface will guide you through:
1. Selecting your PDF file
2. Configuring extraction options
3. Viewing beautifully formatted results

## Quick Examples

### Interactive Mode (Guided)
```bash
python cli.py
```

### Quick Mode (Direct)
```bash
python cli.py --file annual_report.pdf
```

### Advanced Quick Mode
```bash
python cli.py --file report.pdf --model deepseek-chat --max-pages 20 --show-reasoning
```

### Command Line Mode (For Scripts)
```bash
python pdf_extractor.py report.pdf -o results.json
```

## What You'll Get

The tool will extract these 18 financial metrics:
- Credit Sales, Net Sales, Debtors
- Current Assets, Current Liabilities
- Cash, Marketable Securities
- Total Assets, Total Liabilities, Equity
- EBIT, Interest, Tax, Net Income
- Labour Cost, Operating Expenses, R&D Cost
- Impaired Debts

All values in Kenyan Shillings (KES)!

## Output Location

Results are automatically saved to the `output/` directory with timestamps:
```
output/company_report_20250101_143022.json
```

## Tips

1. **Start with interactive mode** - It's the easiest way to learn
2. **Use --max-pages 10** for testing - Process just the first few pages
3. **Check --show-reasoning** - See how the AI made its decisions
4. **Batch process** - Select multiple PDFs in interactive mode
5. **Review N/A values** - They're normal when data isn't disclosed

## Troubleshooting

**PDF too large error (413)?**
- Use: `python pdf_extractor.py report.pdf --max-pages 10`
- Extract only financial statement pages from your PDF

**No API key?**
- Make sure you created `.env` file
- Check that it contains `DEEPSEEK_API_KEY=...`

**Poppler not found?**
- Run: `sudo apt-get install poppler-utils` (Ubuntu)
- Or: `brew install poppler` (macOS)

**Import errors?**
- Run: `pip install -r requirements.txt`

**For more help:**
- See `TROUBLESHOOTING.md` for detailed solutions
- Check the logs - they show exactly what's happening

## Next Steps

- See `README.md` for comprehensive documentation
- Check `examples/` for sample PDFs and results
- Read about the extraction prompts in `pdf_extractor.py`

Happy extracting!
