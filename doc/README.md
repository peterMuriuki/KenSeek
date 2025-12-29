# PDF Financial Data Extractor

A Python tool that uses DeepSeek's AI API to extract financial metrics from Kenyan company annual reports.

## Features

- **OCR Preprocessing Mode** - NEW! Extract text first for 90%+ smaller payloads - handle 50+ page PDFs
- **Interactive CLI** with beautiful formatted output and progress tracking
- **Self-Consistency Checking** - Extract with multiple samples and use consensus for higher accuracy
- Extracts 18 key financial metrics from PDF annual reports
- Uses Chain-of-Thought prompting for accurate extraction
- Specifically designed for Kenyan companies registered under the Kenya Association of Manufacturers
- Returns structured JSON output with confidence metrics
- Handles multi-page PDFs with image-based processing
- Batch processing support for multiple files
- Shows model reasoning process (optional)
- Visual tables, progress bars, and status indicators

## Extracted Metrics

The tool extracts the following 18 financial metrics:

1. **credit_sales** - Total credit sales for the period
2. **net_sales** - Net revenue/sales after returns and allowances
3. **debtors** - Accounts receivable / trade receivables
4. **impaired_debts** - Bad debts / doubtful debts / debt provisions
5. **current_assets** - Total current assets
6. **current_liabilities** - Total current liabilities
7. **cash** - Cash and cash equivalents
8. **marketable_securities** - Short-term investments
9. **total_liabilities** - Total liabilities (current + non-current)
10. **total_assets** - Total assets
11. **ebit** - Earnings Before Interest and Tax / Operating profit
12. **interest** - Interest expense
13. **labour_cost** - Employee costs / staff costs / wages and salaries
14. **operating_expenses** - Operating expenses / administrative expenses
15. **rd_cost** - Research and development costs
16. **tax** - Income tax expense
17. **equity** - Total equity / shareholders' equity
18. **net_income** - Net profit / profit after tax

All values are reported in Kenyan Shillings (KES). If a metric cannot be confidently determined, it will be marked as "N/A".

## Self-Consistency Checking

Self-consistency is an advanced feature that significantly improves extraction accuracy by running multiple independent extractions and using consensus voting.

### How It Works

1. **Multiple Samples**: The tool extracts data N times (typically 3-5) from the same PDF
2. **Independent Runs**: Each sample uses slightly higher temperature for diversity
3. **Majority Voting**: For each metric, the most common value across samples is selected
4. **Confidence Scoring**: Agreement percentage indicates reliability (100% = all samples agree)
5. **Warnings**: Low confidence metrics are flagged for manual review

### When to Use Self-Consistency

**Recommended for:**
- Critical financial reports where accuracy is paramount
- Complex or ambiguous financial statements
- Reports with non-standard formats
- When you need confidence metrics for downstream analysis

**May skip for:**
- Quick exploratory analysis
- Well-structured, clear financial statements
- Cost-sensitive scenarios (uses 3-5x more tokens)

### Confidence Levels

- **High (100%)**: All samples agreed - very reliable
- **Medium (60-99%)**: Most samples agreed - generally reliable, review recommended
- **Low (<60%)**: Significant disagreement - manual review required

### Example Output

```json
{
  "net_sales": {
    "final_value": "480000000",
    "confidence_score": 1.0,
    "confidence_level": "high",
    "agreement": "5/5",
    "samples": ["480000000", "480000000", "480000000", "480000000", "480000000"]
  },
  "ebit": {
    "final_value": "75000000",
    "confidence_score": 0.6,
    "confidence_level": "medium",
    "agreement": "3/5",
    "samples": ["75000000", "75000000", "75000000", "72000000", "78000000"],
    "warning": "Low confidence - review manually"
  }
}
```

### Usage Examples

**Interactive Mode:**
- The CLI will ask if you want to enable self-consistency
- Choose number of samples (3, 5, or 7)
- Optionally view detailed confidence metrics

**Command Line:**
```bash
# Enable self-consistency with 3 samples
python pdf_extractor.py report.pdf --self-consistency

# Use 5 samples for better accuracy
python pdf_extractor.py report.pdf --self-consistency --num-samples 5

# Show detailed confidence metrics
python pdf_extractor.py report.pdf --self-consistency --show-confidence

# CLI quick mode
python cli.py --file report.pdf --self-consistency --num-samples 5 --show-confidence
```

**Python Module:**
```python
from pdf_extractor import PDFFinancialExtractor

extractor = PDFFinancialExtractor()

result = extractor.extract_with_self_consistency(
    pdf_path="annual_report.pdf",
    num_samples=5,
    temperature=0.3
)

# Check confidence
for metric, details in result['confidence_metrics'].items():
    if details['confidence_level'] == 'low':
        print(f"Warning: {metric} has low confidence")
        print(f"  Samples: {details['samples']}")
```

### Cost Implications

Self-consistency uses N times the tokens of a single extraction:
- 3 samples: ~3x cost
- 5 samples: ~5x cost
- 7 samples: ~7x cost

For a typical 30-page report (~15,000 tokens per extraction):
- Single extraction: ~15,000 tokens
- 3-sample self-consistency: ~45,000 tokens
- 5-sample self-consistency: ~75,000 tokens

## Prerequisites

- Python 3.8 or higher
- poppler-utils (for PDF to image conversion)
- DeepSeek API key

### Install poppler-utils

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
Download from: http://blog.alivate.com.au/poppler-windows/

## Installation

1. Clone or download this repository

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your API key:
```bash
cp .env.example .env
```

4. Edit `.env` and add your DeepSeek API key:
```
DEEPSEEK_API_KEY=your_actual_api_key_here
```

Get your API key from: https://platform.deepseek.com/api_keys

## Usage

There are three ways to use this tool:

### 1. Interactive CLI (Recommended for Beginners)

Launch the interactive interface:

```bash
python cli.py
```

The interactive CLI provides:
- Step-by-step guided workflow
- Beautiful formatted output with tables and progress bars
- File browser for easy PDF selection
- Batch processing with checkboxes
- Automatic result saving to `output/` directory
- Visual progress tracking

**Quick mode (CLI with file specified):**
```bash
python cli.py --file report.pdf
python cli.py --file report.pdf --model deepseek-chat --max-pages 20 --show-reasoning

# Use OCR mode for large PDFs (recommended for 20+ pages)
python cli.py --file report.pdf --use-ocr
```

### 2. Command Line (Advanced Users)

Extract financial data from a PDF:

```bash
python pdf_extractor.py path/to/annual_report.pdf
```

**For large PDFs (20+ pages), use OCR mode:**
```bash
python pdf_extractor.py path/to/annual_report.pdf --use-ocr
```

See **[OCR_MODE.md](OCR_MODE.md)** for comprehensive OCR documentation.

### Advanced Options

**Save output to a JSON file:**
```bash
python pdf_extractor.py report.pdf -o output.json
```

**Show the model's reasoning process:**
```bash
python pdf_extractor.py report.pdf --show-reasoning
```

**Process only the first 10 pages:**
```bash
python pdf_extractor.py report.pdf --max-pages 10
```

**Use a different model:**
```bash
python pdf_extractor.py report.pdf -m deepseek-chat
```

**Override API key (without .env file):**
```bash
python pdf_extractor.py report.pdf --api-key sk-your-key-here
```

### Command Line Options

```
positional arguments:
  pdf_path              Path to the PDF file to process

optional arguments:
  -h, --help            Show help message
  -o OUTPUT, --output OUTPUT
                        Output JSON file path (default: print to stdout)
  -m MODEL, --model MODEL
                        DeepSeek model to use (default: deepseek-chat)
  --max-pages MAX_PAGES
                        Maximum number of pages to process
  --api-key API_KEY     DeepSeek API key (overrides .env)
  --show-reasoning      Display the model's reasoning process
  --self-consistency    Enable self-consistency checking
  --num-samples N       Number of samples for self-consistency (default: 3)
  --temperature T       Temperature for sampling (default: 0.3)
  --show-confidence     Show detailed confidence metrics
  --use-ocr             Use OCR preprocessing (recommended for large PDFs)
  --save-ocr            Save OCR text to file (default: true)
```

## Output Format

The tool outputs a JSON object with the following structure:

```json
{
  "success": true,
  "file": "path/to/report.pdf",
  "model": "deepseek-chat",
  "reasoning": "... (model's step-by-step reasoning) ...",
  "extracted_data": {
    "credit_sales": "500000000",
    "net_sales": "480000000",
    "debtors": "120000000",
    "impaired_debts": "5000000",
    "current_assets": "250000000",
    "current_liabilities": "150000000",
    "cash": "50000000",
    "marketable_securities": "N/A",
    "total_liabilities": "300000000",
    "total_assets": "600000000",
    "ebit": "75000000",
    "interest": "8000000",
    "labour_cost": "100000000",
    "operating_expenses": "200000000",
    "rd_cost": "N/A",
    "tax": "15000000",
    "equity": "300000000",
    "net_income": "52000000"
  },
  "usage": {
    "prompt_tokens": 12543,
    "completion_tokens": 1250,
    "total_tokens": 13793
  }
}
```

### 3. Python Module

You can also import and use the extractor in your own Python code:

```python
from pdf_extractor import PDFFinancialExtractor

# Initialize the extractor
extractor = PDFFinancialExtractor(api_key="your-api-key")

# Extract data
result = extractor.extract_financial_data(
    pdf_path="annual_report.pdf",
    model="deepseek-chat",
    max_pages=None  # Process all pages
)

if result["success"]:
    financial_data = result["extracted_data"]
    print(f"Net Sales: KES {financial_data['net_sales']}")
    print(f"Net Income: KES {financial_data['net_income']}")
else:
    print(f"Error: {result['error']}")
```

## Interactive CLI Features

The `cli.py` wrapper provides an enhanced user experience:

### File Selection
- **Manual entry**: Type or paste the file path
- **Browse current directory**: Select from a list of PDFs found
- **Batch processing**: Select multiple files using checkboxes

### Visual Feedback
- Beautiful formatted tables showing all 18 metrics
- Color-coded status indicators (✓ for found, ○ for N/A)
- Progress bars during PDF processing
- Token usage statistics
- Professional panel displays

### Configuration Options
- Model selection (deepseek-chat, deepseek-reasoner)
- Page limiting for faster testing
- Optional reasoning display
- Automatic output saving with timestamps

### Batch Processing
- Process multiple PDFs in one session
- Individual results for each file
- Summary statistics at the end
- Option to continue or stop between files

### CLI Command Line Options

```
python cli.py [options]

optional arguments:
  -h, --help            Show help message
  -f FILE, --file FILE  PDF file to process (enables quick mode)
  -m MODEL, --model MODEL
                        DeepSeek model to use (default: deepseek-chat)
  --max-pages MAX_PAGES
                        Maximum number of pages to process
  --show-reasoning      Display model reasoning process
  --no-save            Do not save results to file
```

## How It Works

1. **PDF Conversion**: The PDF is converted to high-resolution images (200 DPI)
2. **Image Encoding**: Images are encoded as base64 for API transmission
3. **AI Analysis**: Images are sent to DeepSeek's vision model with specialized prompts
4. **Chain-of-Thought**: The model reasons step-by-step through the financial statements
5. **Data Extraction**: The model extracts the 18 required metrics
6. **JSON Parsing**: The structured data is parsed and returned

## Tips for Best Results

- Use clear, high-quality PDF scans
- Ensure financial statements are visible and legible
- For large reports, consider using `--max-pages` to process only relevant sections
- Review the `--show-reasoning` output to understand the model's decision-making
- "N/A" values are expected and acceptable when data isn't available
- **Enable self-consistency for critical reports** - The 3-5x cost is worth it for high-stakes financial analysis
- **Review low-confidence metrics** - When self-consistency shows disagreement, check the original document
- **Start with 3 samples** - Balance between accuracy and cost before trying higher sample counts

## Cost Considerations

DeepSeek API charges based on token usage. Processing a typical 50-page annual report may use approximately 10,000-20,000 tokens. Check current pricing at: https://platform.deepseek.com/pricing

## Automatic Optimizations

The tool includes several automatic optimizations to handle large PDFs:

- **Image Compression**: JPEG encoding with 85% quality (vs uncompressed PNG)
- **Auto-Resize**: Images scaled to max 2048px dimension
- **Reduced DPI**: 150 DPI (down from 200) for faster processing
- **Format Conversion**: RGB conversion removes alpha channels

These optimizations reduce payload size by **60-80%** compared to unoptimized images.

## Troubleshooting

For detailed troubleshooting, see **[TROUBLESHOOTING.md](TROUBLESHOOTING.md)**

**Quick fixes for common errors:**

**"413 Request Entity Too Large":**
```bash
# BEST SOLUTION: Use OCR mode (90%+ smaller payload)
python pdf_extractor.py report.pdf --use-ocr

# OR: Limit to financial statement pages only
python pdf_extractor.py report.pdf --max-pages 10
```

**"poppler not found":**
```bash
# Ubuntu/Debian
sudo apt-get install poppler-utils

# macOS
brew install poppler
```

**"DEEPSEEK_API_KEY not found":**
```bash
# Create .env file
cp .env.example .env
# Edit .env and add your API key
```

**Large PDF processing:**
- Extract only the financial statement pages (typically 5-15 pages)
- The tool will show payload size - keep it under 10 MB
- Watch the logs for optimization details

## License

This tool is provided as-is for extracting financial data from Kenyan company annual reports.

## Support

For issues or questions, please refer to the DeepSeek API documentation: https://platform.deepseek.com/docs
