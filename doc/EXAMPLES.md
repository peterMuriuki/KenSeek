# Usage Examples

This guide provides practical examples for using the PDF Financial Data Extractor with and without self-consistency.

## Table of Contents

1. [Basic Usage](#basic-usage)
2. [Self-Consistency Examples](#self-consistency-examples)
3. [Batch Processing](#batch-processing)
4. [Python Module Usage](#python-module-usage)

## Basic Usage

### Simple Extraction

```bash
# Standard extraction
python pdf_extractor.py annual_report.pdf

# Save to file
python pdf_extractor.py annual_report.pdf -o results.json

# Process first 20 pages only
python pdf_extractor.py annual_report.pdf --max-pages 20

# Show model's reasoning
python pdf_extractor.py annual_report.pdf --show-reasoning
```

### Interactive CLI

```bash
# Launch interactive mode
python cli.py

# Quick mode with a specific file
python cli.py --file annual_report.pdf

# Quick mode with options
python cli.py --file report.pdf --model deepseek-chat --max-pages 20
```

## Self-Consistency Examples

### Command Line

**Basic self-consistency (3 samples):**
```bash
python pdf_extractor.py report.pdf --self-consistency
```

**Higher accuracy (5 samples):**
```bash
python pdf_extractor.py report.pdf --self-consistency --num-samples 5
```

**With confidence metrics:**
```bash
python pdf_extractor.py report.pdf \
  --self-consistency \
  --num-samples 5 \
  --show-confidence \
  --show-reasoning
```

**Save detailed results:**
```bash
python pdf_extractor.py report.pdf \
  --self-consistency \
  --num-samples 5 \
  --show-confidence \
  -o detailed_results.json
```

### Interactive CLI with Self-Consistency

```bash
# Launch CLI
python cli.py --file report.pdf --self-consistency --num-samples 5 --show-confidence

# The CLI will display:
# - Confidence summary panel
# - Table with confidence column
# - Detailed metrics for low-confidence fields
```

### Interpreting Self-Consistency Results

**Example output with high confidence:**
```json
{
  "net_sales": {
    "final_value": "480000000",
    "confidence_score": 1.0,
    "confidence_level": "high",
    "agreement": "5/5",
    "samples": ["480000000", "480000000", "480000000", "480000000", "480000000"]
  }
}
```
✓ All 5 samples agreed - use this value with confidence

**Example output with medium confidence:**
```json
{
  "labour_cost": {
    "final_value": "100000000",
    "confidence_score": 0.8,
    "confidence_level": "medium",
    "agreement": "4/5",
    "samples": ["100000000", "100000000", "100000000", "100000000", "95000000"],
    "vote_distribution": {"100000000": 4, "95000000": 1}
  }
}
```
⚠️ 4 out of 5 agreed - probably correct, but worth a quick check

**Example output with low confidence:**
```json
{
  "impaired_debts": {
    "final_value": "5000000",
    "confidence_score": 0.4,
    "confidence_level": "low",
    "agreement": "2/5",
    "samples": ["5000000", "5000000", "3000000", "7000000", "N/A"],
    "vote_distribution": {"5000000": 2, "3000000": 1, "7000000": 1, "N/A": 1},
    "warning": "Low confidence - review manually"
  }
}
```
🚨 Significant disagreement - manual review required! Check the original document.

## Batch Processing

### Using Interactive CLI

```bash
# Launch CLI
python cli.py

# Then:
# 1. Select "Batch process multiple files"
# 2. Use spacebar to select multiple PDFs
# 3. Press Enter
# 4. Configure extraction (including self-consistency)
# 5. Review results for each file
```

### Processing Multiple Files with Script

```bash
# Create a simple batch script
for file in reports/*.pdf; do
  echo "Processing $file..."
  python pdf_extractor.py "$file" \
    --self-consistency \
    --num-samples 3 \
    -o "output/$(basename "$file" .pdf)_results.json"
done
```

## Python Module Usage

### Basic Extraction

```python
from pdf_extractor import PDFFinancialExtractor

# Initialize
extractor = PDFFinancialExtractor(api_key="your-api-key")

# Extract
result = extractor.extract_financial_data(
    pdf_path="annual_report.pdf",
    model="deepseek-chat"
)

# Access data
if result["success"]:
    data = result["extracted_data"]
    print(f"Net Sales: KES {data['net_sales']:,}")
    print(f"Net Income: KES {data['net_income']:,}")
```

### Self-Consistency Extraction

```python
from pdf_extractor import PDFFinancialExtractor

# Initialize
extractor = PDFFinancialExtractor()

# Extract with self-consistency
result = extractor.extract_with_self_consistency(
    pdf_path="annual_report.pdf",
    num_samples=5,
    model="deepseek-chat",
    temperature=0.3
)

if result["success"]:
    # Access consensus data
    data = result["extracted_data"]

    # Check confidence metrics
    metrics = result["confidence_metrics"]

    # Identify low-confidence fields
    low_confidence = []
    for metric, details in metrics.items():
        if details["confidence_level"] == "low":
            low_confidence.append(metric)
            print(f"⚠️ {metric}: {details['samples']}")

    # Statistics
    stats = result["statistics"]
    print(f"\nConfidence Summary:")
    print(f"  High: {stats['high_confidence']}/{stats['total_metrics']}")
    print(f"  Medium: {stats['medium_confidence']}/{stats['total_metrics']}")
    print(f"  Low: {stats['low_confidence']}/{stats['total_metrics']}")

    # Save results
    import json
    with open("results_with_confidence.json", "w") as f:
        json.dump(result, f, indent=2)
```

### Advanced: Custom Confidence Threshold

```python
from pdf_extractor import PDFFinancialExtractor

extractor = PDFFinancialExtractor()

result = extractor.extract_with_self_consistency(
    pdf_path="annual_report.pdf",
    num_samples=7,  # More samples for critical data
    temperature=0.3
)

if result["success"]:
    # Filter by custom confidence threshold
    high_quality_data = {}
    needs_review = {}

    for metric, details in result["confidence_metrics"].items():
        if details["confidence_score"] >= 0.7:
            # High confidence - use directly
            high_quality_data[metric] = details["final_value"]
        else:
            # Low confidence - flag for review
            needs_review[metric] = {
                "value": details["final_value"],
                "samples": details["samples"],
                "agreement": details["agreement"]
            }

    print(f"High quality metrics: {len(high_quality_data)}")
    print(f"Needs review: {len(needs_review)}")

    if needs_review:
        print("\nMetrics requiring manual review:")
        for metric, info in needs_review.items():
            print(f"  {metric}: {info['samples']}")
```

### Batch Processing with Python

```python
from pathlib import Path
from pdf_extractor import PDFFinancialExtractor
import json

extractor = PDFFinancialExtractor()

pdf_files = Path("reports").glob("*.pdf")

results = []
for pdf_file in pdf_files:
    print(f"Processing {pdf_file.name}...")

    result = extractor.extract_with_self_consistency(
        pdf_path=str(pdf_file),
        num_samples=3
    )

    if result["success"]:
        # Save individual result
        output_file = Path("output") / f"{pdf_file.stem}_results.json"
        with open(output_file, "w") as f:
            json.dump(result, f, indent=2)

        results.append({
            "file": pdf_file.name,
            "success": True,
            "stats": result["statistics"]
        })
    else:
        results.append({
            "file": pdf_file.name,
            "success": False,
            "error": result.get("error")
        })

# Summary
print(f"\nProcessed {len(results)} files")
print(f"Successful: {sum(1 for r in results if r['success'])}")
print(f"Failed: {sum(1 for r in results if not r['success'])}")
```

## Tips and Best Practices

### When to Use Self-Consistency

✅ **Use self-consistency for:**
- Audited financial statements
- Regulatory filings
- Investment decisions
- Critical business analysis
- Reports with unclear formatting

❌ **Skip self-consistency for:**
- Quick exploratory analysis
- Internal drafts
- Well-structured standard reports
- Cost-sensitive scenarios

### Optimizing Sample Count

| Sample Count | Use Case | Accuracy | Cost |
|--------------|----------|----------|------|
| 3 | Standard production use | Good | 3x |
| 5 | Critical analysis | Better | 5x |
| 7 | Maximum accuracy needed | Best | 7x |

### Handling Disagreements

When self-consistency shows disagreement:

1. **Check confidence level**
   - High (100%): Trust the value
   - Medium (60-99%): Quick review recommended
   - Low (<60%): Manual verification required

2. **Review vote distribution**
   ```json
   "vote_distribution": {"75000000": 3, "72000000": 1, "78000000": 1}
   ```
   If values are close (75M vs 72M vs 78M), the variation might be from:
   - Different units (thousands vs actual amounts)
   - Multiple similar line items
   - Ambiguous labels

3. **Check the samples**
   ```python
   samples = metrics["ebit"]["samples"]
   print(f"Samples: {samples}")
   # If seeing: ["75000000", "75000", "75000000", "75000", "75000000"]
   # This indicates a unit scaling issue
   ```

4. **Review reasoning**
   - Use `--show-reasoning` to see the model's thought process
   - Check if the model identified the correct line item

## Conclusion

Self-consistency adds significant value for high-stakes financial data extraction. Start with 3 samples, review the confidence metrics, and increase to 5-7 samples for critical reports.

For questions or issues, check the main README or open an issue.
