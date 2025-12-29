#!/usr/bin/env python3
"""
PDF Financial Data Extractor using DeepSeek API
Extracts 18 financial metrics from Kenyan company annual reports.
"""

import os
import json
import base64
import re
from pathlib import Path
from typing import Dict, Optional, List, Tuple
from io import BytesIO
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import time

try:
    from openai import OpenAI
    from pdf2image import convert_from_path
    from PIL import Image
    from dotenv import load_dotenv
    import google.generativeai as genai
    from database import ExtractionDatabase
except ImportError as e:
    print(f"Missing required package: {e}")
    print("Please install requirements: pip install -r requirements.txt")
    exit(1)

# Load environment variables
load_dotenv()

# DeepSeek API configuration
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = "https://api.deepseek.com"

# Google Gemini API configuration (for OCR)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Chain-of-Thought prompt templates
SYSTEM_PROMPT = """You are an expert financial analyst and data extraction specialist with deep expertise in analyzing annual reports for Kenyan companies registered under the Kenya Association of Manufacturers.

Your task is to extract specific financial metrics from annual report text with high accuracy. You must think step-by-step and show your reasoning process before providing final values.

CRITICAL RULES:
1. Report ALL monetary values in Kenyan Shillings (KES/KSh)
2. If you cannot confidently determine a value, ALWAYS use "N/A" - this is perfectly acceptable
3. Using "N/A" is better than guessing or making assumptions
4. Only extract values you fully understand and can justify
5. Show your reasoning for each extracted value
6. Look for values in the most recent fiscal year in the document
7. Be aware of unit scaling (thousands, millions, billions)

IMPORTANT: "N/A" is a valid and expected response for fields that are:
- Not disclosed in the report
- Unclear or ambiguous
- Combined with other line items
- Reported in a non-standard format you're uncertain about
"""

EXTRACTION_PROMPT = """Analyze the attached annual report PDF and extract the 18 required financial metrics.

REQUIRED METRICS:
1. credit_sales: Total credit sales for the period
2. net_sales: Net revenue/sales after returns and allowances
3. debtors: Accounts receivable / borrowed moneys / trade receivables
4. impaired_debts: This are Bad debts / doubtful debts / debt provisions
5. current_assets: Total current assets
6. current_liabilities: Total current liabilities
7. cash: Cash and cash equivalents
8. marketable_securities: Short-term investments / marketable securities
9. total_liabilities: Total liabilities (current + non-current)
10. total_assets: Total assets
11. ebit: Earnings Before Interest and Tax / Operating profit
12. interest: Interest expense
13. labour_cost: Employee costs / staff costs / wages and salaries
14. operating_expenses: Operating expenses / administrative expenses
15. rd_cost: Research and development costs (if disclosed)
16. tax: Income tax expense
17. equity: Total equity / shareholders' equity
18. net_income: Net profit / profit after tax / profit for the year

EXTRACTION PROCESS:
For each metric, follow these steps:
1. Search: Locate where in the PDF this value appears (check financial statements, notes)
2. Identify: Find the specific value and note its context
3. Convert: Ensure the value is in Kenyan Shillings (check for KSh, KES, thousands, millions)
4. Verify: Confirm this is the correct value for the most recent period
5. Decide: Assign the final value or "N/A" if uncertain

RESPONSE FORMAT:
Think through each metric step-by-step, then provide your final answer as a JSON object with this exact structure:

{
  "credit_sales": "value_in_shillings_or_NA",
  "net_sales": "value_in_shillings_or_NA",
  "debtors": "value_in_shillings_or_NA",
  "impaired_debts": "value_in_shillings_or_NA",
  "current_assets": "value_in_shillings_or_NA",
  "current_liabilities": "value_in_shillings_or_NA",
  "cash": "value_in_shillings_or_NA",
  "marketable_securities": "value_in_shillings_or_NA",
  "total_liabilities": "value_in_shillings_or_NA",
  "total_assets": "value_in_shillings_or_NA",
  "ebit": "value_in_shillings_or_NA",
  "interest": "value_in_shillings_or_NA",
  "labour_cost": "value_in_shillings_or_NA",
  "operating_expenses": "value_in_shillings_or_NA",
  "rd_cost": "value_in_shillings_or_NA",
  "tax": "value_in_shillings_or_NA",
  "equity": "value_in_shillings_or_NA",
  "net_income": "value_in_shillings_or_NA"
}

IMPORTANT:
- Show your thinking process first, then provide the JSON
- Only include numeric values without currency symbols or commas in the JSON
- For "N/A" values, use exactly "N/A" (not null, not empty string, not 0)
- If a value is reported in thousands/millions, convert to actual amount
- It is completely acceptable to have multiple "N/A" values in your response
- Better to use "N/A" than to guess or estimate a value
"""


@dataclass
class ExtractionJob:
    """Represents a single extraction job."""
    pdf_path: str
    company_name: str
    financial_year: str

    def __repr__(self):
        return f"ExtractionJob(company={self.company_name}, year={self.financial_year})"


def parse_year_from_filename(filename: str) -> Optional[str]:
    """
    Extract year from filename following format: <random_chars>-<YYYY>.pdf

    Args:
        filename: Name of the PDF file

    Returns:
        Year as string (YYYY) or None if not found

    Examples:
        >>> parse_year_from_filename("abc123-2023.pdf")
        "2023"
        >>> parse_year_from_filename("report-2022.pdf")
        "2022"
        >>> parse_year_from_filename("SGL-Annual-Report-2023.pdf")
        "2023"
    """
    # Match pattern: anything followed by dash and 4 digits (year)
    match = re.search(r'-(\d{4})(?:\.pdf)?$', filename, re.IGNORECASE)
    if match:
        return match.group(1)
    return None


def scan_company_folders(root_folder: str, filter_companies: Optional[List[str]] = None) -> List[ExtractionJob]:
    """
    Scan root folder for company subfolders and their financial reports.

    Directory structure expected:
        root_folder/
        ├── Company_A/
        │   ├── report-2021.pdf
        │   ├── report-2022.pdf
        │   └── report-2023.pdf
        └── Company_B/
            └── annual-2023.pdf

    Args:
        root_folder: Path to root folder containing company subfolders
        filter_companies: Optional list of company names to process (None = all)

    Returns:
        List of ExtractionJob objects
    """
    root_path = Path(root_folder)

    if not root_path.exists():
        raise FileNotFoundError(f"Root folder not found: {root_folder}")

    if not root_path.is_dir():
        raise ValueError(f"Path is not a directory: {root_folder}")

    jobs = []

    # Iterate through company folders
    for company_folder in sorted(root_path.iterdir()):
        if not company_folder.is_dir():
            continue

        company_name = company_folder.name

        # Filter companies if specified
        if filter_companies and company_name not in filter_companies:
            continue

        # Scan for PDF files in company folder
        for pdf_file in sorted(company_folder.glob("*.pdf")):
            # Extract year from filename
            year = parse_year_from_filename(pdf_file.name)

            if year:
                jobs.append(ExtractionJob(
                    pdf_path=str(pdf_file),
                    company_name=company_name,
                    financial_year=year
                ))
            else:
                print(f"[WARN] Could not parse year from: {pdf_file.name} (skipping)")

    return jobs


def process_single_extraction(
    job: ExtractionJob,
    deepseek_api_key: str,
    gemini_api_key: str,
    db_path: str,
    model: str = "deepseek-chat",
    num_samples: int = 3,
    max_pages: Optional[int] = None
) -> Dict:
    """
    Process a single extraction job (designed for multiprocessing).

    This function is designed to be called by ProcessPoolExecutor.
    Each process gets its own extractor instance and database connection.

    Args:
        job: ExtractionJob to process
        deepseek_api_key: DeepSeek API key
        gemini_api_key: Gemini API key
        db_path: Path to SQLite database
        model: DeepSeek model to use
        num_samples: Number of samples for self-consistency
        max_pages: Maximum pages to process

    Returns:
        Result dictionary with success status and extracted data
    """
    try:
        # Create extractor instance (each process needs its own)
        extractor = PDFFinancialExtractor(
            api_key=deepseek_api_key,
            gemini_api_key=gemini_api_key,
            use_database=True,
            db_path=db_path
        )

        print(f"\n[{job.company_name} / {job.financial_year}] Starting extraction...")
        print(f"[{job.company_name} / {job.financial_year}] File: {job.pdf_path}")

        # Always use OCR preprocessing for batch mode
        result = extractor.extract_with_ocr_preprocessing(
            pdf_path=job.pdf_path,
            max_pages=max_pages,
            model=model,
            save_ocr=True,
            num_samples=num_samples,
            company_name=job.company_name,
            financial_year=job.financial_year
        )

        if result["success"]:
            print(f"[{job.company_name} / {job.financial_year}] ✓ Extraction completed")
        else:
            print(f"[{job.company_name} / {job.financial_year}] ✗ Extraction failed: {result.get('error', 'Unknown error')}")

        return {
            "job": job,
            "result": result
        }

    except Exception as e:
        print(f"[{job.company_name} / {job.financial_year}] ✗ Exception: {type(e).__name__}: {e}")
        return {
            "job": job,
            "result": {
                "success": False,
                "error": str(e),
                "file": job.pdf_path,
                "company_name": job.company_name,
                "financial_year": job.financial_year
            }
        }


class PDFFinancialExtractor:
    """Extract financial data from PDFs using Gemini (OCR) + DeepSeek (extraction)."""

    def __init__(self, api_key: Optional[str] = None, gemini_api_key: Optional[str] = None,
                 use_database: bool = True, db_path: str = "./extractions.db"):
        """
        Initialize the extractor with API credentials.

        Args:
            api_key: DeepSeek API key (for financial extraction)
            gemini_api_key: Google Gemini API key (for OCR preprocessing)
            use_database: Enable database caching and tracking
            db_path: Path to SQLite database file
        """
        # DeepSeek setup (for extraction)
        self.api_key = api_key or DEEPSEEK_API_KEY
        if not self.api_key:
            raise ValueError("DEEPSEEK_API_KEY not found. Set it in .env file or pass it directly.")

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=DEEPSEEK_BASE_URL
        )

        # Gemini setup (for OCR)
        self.gemini_api_key = gemini_api_key or GEMINI_API_KEY
        if self.gemini_api_key:
            genai.configure(api_key=self.gemini_api_key)
            # Use gemini-2.5-flash-lite for most cost-effective OCR
            self.gemini_model = genai.GenerativeModel('gemini-2.5-flash-lite')
            print("[INIT] ✓ Gemini OCR configured (gemini-2.5-flash-lite)")
        else:
            self.gemini_model = None
            print("[INIT] ⚠ Gemini API key not found. OCR mode will not be available.")

        # Database setup (for caching and tracking)
        self.use_database = use_database
        if use_database:
            self.db = ExtractionDatabase(db_path)
            print(f"[INIT] ✓ Database initialized ({db_path})")
        else:
            self.db = None

    def pdf_to_images(self, pdf_path: str, dpi: int = 150) -> List[Image.Image]:
        """Convert PDF pages to images."""
        print(f"[STEP 1/6] Converting PDF to images (DPI: {dpi})...")
        try:
            images = convert_from_path(pdf_path, dpi=dpi)
            print(f"[STEP 1/6] ✓ Converted {len(images)} pages to images")

            # Log image sizes
            total_size = sum(img.size[0] * img.size[1] for img in images)
            print(f"[STEP 1/6] Total pixels: {total_size:,} ({total_size / 1_000_000:.1f}M pixels)")

            return images
        except Exception as e:
            print(f"[STEP 1/6] ✗ Failed to convert PDF to images")
            raise RuntimeError(f"Failed to convert PDF to images: {e}")

    def optimize_image(self, image: Image.Image, max_dimension: int = 2048) -> Image.Image:
        """
        Optimize image for API transmission.

        Args:
            image: PIL Image to optimize
            max_dimension: Maximum width or height (default: 2048)

        Returns:
            Optimized PIL Image
        """
        width, height = image.size

        # Resize if image is too large
        if width > max_dimension or height > max_dimension:
            if width > height:
                new_width = max_dimension
                new_height = int(height * (max_dimension / width))
            else:
                new_height = max_dimension
                new_width = int(width * (max_dimension / height))

            print(f"    Resizing from {width}x{height} to {new_width}x{new_height}")
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

        # Convert to RGB if needed (remove alpha channel)
        if image.mode in ('RGBA', 'LA', 'P'):
            rgb_image = Image.new('RGB', image.size, (255, 255, 255))
            if image.mode == 'P':
                image = image.convert('RGBA')
            rgb_image.paste(image, mask=image.split()[-1] if image.mode in ('RGBA', 'LA') else None)
            image = rgb_image

        return image

    def image_to_base64(self, image: Image.Image, quality: int = 85) -> str:
        """
        Convert PIL Image to base64 string with compression.

        Args:
            image: PIL Image to convert
            quality: JPEG quality (1-100, default: 85)

        Returns:
            Base64 encoded string
        """
        buffered = BytesIO()

        # Save as JPEG with compression instead of PNG
        image.save(buffered, format="JPEG", quality=quality, optimize=True)

        size_kb = len(buffered.getvalue()) / 1024
        return base64.b64encode(buffered.getvalue()).decode('utf-8'), size_kb

    def extract_financial_data(
        self,
        pdf_path: str,
        model: str = "deepseek-chat",
        max_pages: Optional[int] = None
    ) -> Dict:
        """
        Extract financial metrics from a PDF using DeepSeek API.

        Args:
            pdf_path: Path to the PDF file
            model: DeepSeek model to use (default: deepseek-chat)
            max_pages: Maximum number of pages to process (None for all)

        Returns:
            Dictionary containing extracted financial metrics and metadata
        """
        # Validate PDF exists
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        print(f"\n{'='*70}")
        print(f"STARTING EXTRACTION")
        print(f"{'='*70}")
        print(f"File: {pdf_path}")
        print(f"Model: {model}")

        # Convert PDF to images
        images = self.pdf_to_images(pdf_path)

        if max_pages:
            images = images[:max_pages]
            print(f"[STEP 2/6] Limited to first {max_pages} pages")
        else:
            print(f"[STEP 2/6] Processing all {len(images)} pages")

        # Prepare content for API
        print(f"[STEP 3/6] Optimizing and encoding images...")
        content = [{"type": "text", "text": EXTRACTION_PROMPT}]

        # Add images to content
        total_size_kb = 0
        for idx, image in enumerate(images):
            print(f"[STEP 3/6] Processing page {idx + 1}/{len(images)}")

            # Optimize image
            optimized_image = self.optimize_image(image)

            # Convert to base64
            base64_image, size_kb = self.image_to_base64(optimized_image)
            total_size_kb += size_kb

            print(f"[STEP 3/6]   Page {idx + 1}: {size_kb:.1f} KB")

            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                }
            })

        print(f"[STEP 3/6] ✓ Total payload size: {total_size_kb:.1f} KB ({total_size_kb/1024:.2f} MB)")

        # Check if payload is too large
        if total_size_kb > 10000:  # 10 MB
            print(f"[WARNING] Payload is large ({total_size_kb/1024:.1f} MB). Consider using --max-pages to reduce size.")

        # Call DeepSeek API
        print(f"[STEP 4/6] Sending request to DeepSeek API...")
        print(f"[STEP 4/6] Request details:")
        print(f"[STEP 4/6]   - Pages: {len(images)}")
        print(f"[STEP 4/6]   - Payload size: {total_size_kb/1024:.2f} MB")
        print(f"[STEP 4/6]   - Model: {model}")

        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": content}
                ],
                temperature=0.1,  # Low temperature for consistent extraction
            )

            print(f"[STEP 4/6] ✓ Received response from DeepSeek")
            print(f"[STEP 4/6] Response length: {len(response.choices[0].message.content)} characters")

            # Extract the response content
            print(f"[STEP 5/6] Extracting response content...")
            raw_response = response.choices[0].message.content

            # Parse JSON from response
            print(f"[STEP 6/6] Parsing JSON from response...")
            financial_data = self._parse_json_from_response(raw_response)
            print(f"[STEP 6/6] ✓ Successfully parsed {len(financial_data)} metrics")

            print(f"\n{'='*70}")
            print(f"EXTRACTION COMPLETE")
            print(f"{'='*70}")

            return {
                "success": True,
                "file": pdf_path,
                "model": model,
                "reasoning": raw_response,
                "extracted_data": financial_data,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }
            }

        except Exception as e:
            print(f"\n{'='*70}")
            print(f"EXTRACTION FAILED")
            print(f"{'='*70}")
            print(f"[ERROR] {type(e).__name__}: {e}")

            # Provide helpful error messages
            error_str = str(e)
            if "413" in error_str or "Request Entity Too Large" in error_str:
                print(f"\n[SOLUTION] The PDF is too large. Try:")
                print(f"  1. Use --max-pages to process fewer pages (e.g., --max-pages 10)")
                print(f"  2. Extract only the financial statement pages from your PDF")
                print(f"  3. Use a lower DPI (contact support for custom DPI settings)")
            elif "401" in error_str or "Unauthorized" in error_str:
                print(f"\n[SOLUTION] API key issue. Check:")
                print(f"  1. Your API key is correct in .env file")
                print(f"  2. Your API key is active on DeepSeek platform")
            elif "429" in error_str or "rate limit" in error_str.lower():
                print(f"\n[SOLUTION] Rate limit exceeded. Try:")
                print(f"  1. Wait a few minutes and try again")
                print(f"  2. Check your API usage on DeepSeek platform")

            return {
                "success": False,
                "file": pdf_path,
                "error": str(e)
            }

    def _parse_json_from_response(self, response: str) -> Dict:
        """Extract and parse JSON from the model's response."""
        # Try to find JSON in the response
        start_idx = response.find('{')
        end_idx = response.rfind('}') + 1

        if start_idx == -1 or end_idx == 0:
            raise ValueError("No JSON object found in response")

        json_str = response[start_idx:end_idx]

        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON: {e}\nJSON string: {json_str}")

    def _init_local_ocr(self):
        """Initialize local DeepSeek-OCR model (lazy loading)."""
        if hasattr(self, 'ocr_model') and self.ocr_model is not None:
            return  # Already initialized

        print(f"[OCR] Loading DeepSeek-OCR model locally...")
        print(f"[OCR] This may take a few minutes on first run (downloading model)...")

        try:
            from transformers import AutoModelForVision2Seq, AutoTokenizer
            import torch

            model_name = "deepseek-ai/deepseek-vl-1.3b-chat"

            print(f"[OCR] Loading tokenizer...")
            self.ocr_tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

            print(f"[OCR] Loading model...")
            self.ocr_model = AutoModelForVision2Seq.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )

            # Move to GPU if available
            if torch.cuda.is_available():
                self.ocr_model = self.ocr_model.cuda()
                print(f"[OCR] ✓ Model loaded on GPU")
            else:
                print(f"[OCR] ✓ Model loaded on CPU (slower, consider using GPU)")

        except ImportError as e:
            print(f"[OCR] ✗ Missing dependencies: {e}")
            print(f"[OCR] Please install: pip install transformers torch")
            raise
        except Exception as e:
            print(f"[OCR] ✗ Failed to load model: {e}")
            raise

    def ocr_pdf_pages(
        self,
        pdf_path: str,
        max_pages: Optional[int] = None,
        use_gemini: bool = True,
        force_reocr: bool = False
    ) -> Tuple[List[str], Optional[int]]:
        """
        OCR all pages of a PDF using Google Gemini Vision API.

        Checks cache first. If cached OCR exists and file hash matches,
        returns cached text. Otherwise, performs OCR and caches results.

        Args:
            pdf_path: Path to the PDF file
            max_pages: Maximum number of pages to OCR (None for all)
            use_gemini: Use Gemini API for OCR (default: True)
            force_reocr: Force re-OCR even if cached (default: False)

        Returns:
            Tuple of (list of OCR text per page, PDF ID in database)
        """
        if not self.gemini_model:
            raise ValueError("Gemini API not configured. Set GEMINI_API_KEY in .env file.")

        print(f"\n{'='*70}")
        print(f"OCR PREPROCESSING (Google Gemini 2.5 Flash Lite)")
        print(f"{'='*70}")
        print(f"File: {pdf_path}")
        print(f"Model: gemini-2.5-flash-lite")

        # Check cache first
        pdf_id = None
        if self.use_database and not force_reocr:
            print(f"\n[CACHE] Checking for cached OCR...")
            file_hash = ExtractionDatabase.compute_file_hash(pdf_path)
            print(f"[CACHE] File hash: {file_hash[:16]}...")

            pdf_record = self.db.get_pdf_by_hash(file_hash)
            if pdf_record:
                pdf_id = pdf_record['id']
                cached_ocr = self.db.get_cached_ocr(pdf_id)

                if cached_ocr:
                    print(f"[CACHE] ✓ Found cached OCR!")
                    print(f"[CACHE] Pages: {len(cached_ocr)}")
                    print(f"[CACHE] OCR date: {pdf_record['ocr_date']}")
                    print(f"[CACHE] Total characters: {sum(len(text) for text in cached_ocr):,}")
                    print(f"[CACHE] Skipping OCR (using cache)")
                    print(f"{'='*70}\n")

                    # Apply max_pages limit to cached results
                    if max_pages:
                        return cached_ocr[:max_pages], pdf_id
                    return cached_ocr, pdf_id
                else:
                    print(f"[CACHE] PDF found but no OCR cached, will perform OCR")
            else:
                print(f"[CACHE] PDF not in cache, will perform OCR")
        elif force_reocr:
            print(f"[CACHE] Force re-OCR enabled, skipping cache")

        print(f"Cost: ~$0.10 per 1M tokens (images + text)")

        # Convert PDF to images
        images = self.pdf_to_images(pdf_path)

        if max_pages:
            images = images[:max_pages]
            print(f"[OCR] Limited to first {max_pages} pages")
        else:
            print(f"[OCR] Processing all {len(images)} pages")

        print(f"[OCR] Estimated cost for {len(images)} pages: ~${len(images) * 0.0002:.4f}")

        ocred_pages = []
        total_chars = 0
        failed_pages = 0

        for idx, image in enumerate(images):
            print(f"\n[OCR] Page {idx + 1}/{len(images)}")

            # Optimize image (Gemini handles large images well, but smaller is faster)
            optimized_image = self.optimize_image(image, max_dimension=2048)

            try:
                # Use Gemini Vision API for OCR
                page_text = self._ocr_with_gemini(optimized_image)

                ocred_pages.append(page_text)
                total_chars += len(page_text)

                print(f"[OCR]   ✓ Extracted {len(page_text)} characters")
                if len(page_text) > 0:
                    preview = page_text[:100].replace('\n', ' ')
                    print(f"[OCR]   Preview: {preview}...")

            except Exception as e:
                failed_pages += 1
                error_msg = str(e)
                print(f"[OCR]   ✗ Failed: {error_msg}")

                # Mark page as failed
                ocred_pages.append(f"[OCR FAILED FOR PAGE {idx + 1}]")
                print(f"[OCR]   Added placeholder for failed page")

        print(f"\n[OCR] ✓ OCR Complete")
        print(f"[OCR] Successfully processed: {len(images) - failed_pages}/{len(images)} pages")
        if failed_pages > 0:
            print(f"[OCR] Failed pages: {failed_pages}")
        print(f"[OCR] Total characters extracted: {total_chars:,}")
        if len(images) > 0:
            avg_chars = total_chars // len(images) if len(images) > 0 else 0
            print(f"[OCR] Average per page: {avg_chars:,} chars")

        # Cache OCR results in database
        if self.use_database:
            if pdf_id is None:
                # Create PDF record
                file_hash = ExtractionDatabase.compute_file_hash(pdf_path)
                pdf_id = self.db.get_or_create_pdf(pdf_path, file_hash, len(images))

            # Cache OCR text
            self.db.cache_ocr_text(pdf_id, ocred_pages, "gemini-2.5-flash-lite")

        print(f"{'='*70}\n")

        return ocred_pages, pdf_id

    def _ocr_local(self, image: Image.Image) -> str:
        """OCR using local DeepSeek-OCR model."""
        import torch

        # Prepare image
        # Convert PIL image to format expected by model
        # The model expects RGB images
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Create conversation format for the model
        conversation = [
            {
                "role": "User",
                "content": "<image>\nExtract all text from this image exactly as it appears. Preserve formatting, numbers, and table structure.",
                "images": [image]
            },
            {"role": "Assistant", "content": ""}
        ]

        # Prepare inputs
        inputs = self.ocr_tokenizer.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            return_dict=True
        )

        # Move to same device as model
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}

        # Generate
        with torch.no_grad():
            outputs = self.ocr_model.generate(
                **inputs,
                max_new_tokens=2048,
                do_sample=False,  # Deterministic for OCR
                temperature=None,
                use_cache=True
            )

        # Decode
        generated_text = self.ocr_tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )

        # Extract just the response (remove prompt)
        # The response is after "Assistant:"
        if "Assistant:" in generated_text:
            generated_text = generated_text.split("Assistant:")[-1].strip()

        return generated_text

    def _ocr_with_gemini(self, image: Image.Image) -> str:
        """
        OCR a single image using Google Gemini Vision API.

        Uses Gemini 2.5 Flash Lite for cost-effective and accurate text extraction.

        Args:
            image: PIL Image to OCR

        Returns:
            Extracted text from the image
        """
        if not self.gemini_model:
            raise ValueError("Gemini API not configured. Set GEMINI_API_KEY in .env file.")

        # Prepare OCR prompt
        ocr_prompt = """Extract ALL text from this document image EXACTLY as it appears.

CRITICAL REQUIREMENTS:
- Preserve ALL numbers, dates, and currency symbols EXACTLY
- Maintain table structure and column alignment
- Keep ALL line breaks and spacing
- Include headers, footers, and page numbers
- For unclear text, write [UNCLEAR: your_best_guess]
- Do NOT summarize or skip ANY text
- Do NOT add explanations, commentary, or analysis
- Do NOT interpret or modify the text

Output ONLY the extracted text exactly as shown in the image."""

        try:
            # Generate content with Gemini
            response = self.gemini_model.generate_content([ocr_prompt, image])

            # Extract text from response
            extracted_text = response.text

            return extracted_text

        except Exception as e:
            raise RuntimeError(f"Gemini OCR failed: {str(e)}")

    def extract_from_ocr_text(
        self,
        ocr_text: str,
        pdf_path: str = "OCR_TEXT",
        model: str = "deepseek-chat"
    ) -> Dict:
        """
        Extract financial data from OCRed text (no images).

        This is much more efficient than image-based extraction:
        - Smaller payload (text vs images)
        - Faster processing
        - Lower cost
        - Can handle much larger documents

        Args:
            ocr_text: Full OCRed text from the document
            pdf_path: Original PDF path (for metadata)
            model: DeepSeek model to use

        Returns:
            Dictionary containing extracted financial metrics and metadata
        """
        print(f"\n{'='*70}")
        print(f"TEXT-BASED EXTRACTION")
        print(f"{'='*70}")
        print(f"Source: {pdf_path}")
        print(f"Model: {model}")
        print(f"Text length: {len(ocr_text):,} characters")

        # Create text-based extraction prompt
        text_extraction_prompt = f"""You are analyzing OCRed text from a Kenyan company annual report.

{SYSTEM_PROMPT}

OCRed Document Text:
{ocr_text}

{EXTRACTION_PROMPT}"""

        print(f"\n[TEXT-EXTRACTION] Sending request to DeepSeek API...")
        print(f"[TEXT-EXTRACTION]   Prompt length: {len(text_extraction_prompt):,} characters")

        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": text_extraction_prompt}
                ],
                temperature=0.1,
            )

            print(f"[TEXT-EXTRACTION] ✓ Received response")
            print(f"[TEXT-EXTRACTION] Response length: {len(response.choices[0].message.content)} characters")

            # Parse response
            raw_response = response.choices[0].message.content
            financial_data = self._parse_json_from_response(raw_response)

            print(f"[TEXT-EXTRACTION] ✓ Successfully parsed {len(financial_data)} metrics")
            print(f"\n{'='*70}")
            print(f"EXTRACTION COMPLETE")
            print(f"{'='*70}")

            return {
                "success": True,
                "file": pdf_path,
                "model": model,
                "extraction_method": "ocr_text",
                "reasoning": raw_response,
                "extracted_data": financial_data,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }
            }

        except Exception as e:
            print(f"\n{'='*70}")
            print(f"EXTRACTION FAILED")
            print(f"{'='*70}")
            print(f"[ERROR] {type(e).__name__}: {e}")

            return {
                "success": False,
                "file": pdf_path,
                "extraction_method": "ocr_text",
                "error": str(e)
            }

    def extract_with_ocr_preprocessing(
        self,
        pdf_path: str,
        max_pages: Optional[int] = None,
        model: str = "deepseek-chat",
        save_ocr: bool = True,
        num_samples: int = 3,
        company_name: Optional[str] = None,
        financial_year: Optional[str] = None
    ) -> Dict:
        """
        Two-stage extraction with self-consistency: OCR then extract with consensus.

        Stage 1: OCR each page to extract text (uses cache if available)
        Stage 2: Extract financial data with self-consistency (multiple samples)
        Stage 3: Compute consensus via majority voting

        This approach:
        - Reduces payload size by 90%+ (OCR to text)
        - Handles much larger PDFs (50+ pages)
        - Lower cost (text tokens cheaper than images)
        - Higher accuracy (self-consistency with 3+ samples)
        - Caches OCR for reuse
        - Tracks all attempts in database

        Args:
            pdf_path: Path to the PDF file
            max_pages: Maximum number of pages to process
            model: DeepSeek model to use
            save_ocr: Save OCRed text to file
            num_samples: Number of samples for self-consistency (default: 3)
            company_name: Name of the company being analyzed
            financial_year: Financial year for the data being extracted

        Returns:
            Dictionary containing extracted financial metrics with confidence scores
        """
        # Stage 1: OCR (using Gemini Vision API, with caching)
        ocred_pages, pdf_id = self.ocr_pdf_pages(pdf_path, max_pages, use_gemini=True)

        # Combine all pages
        full_text = "\n\n=== PAGE BREAK ===\n\n".join(
            f"=== PAGE {i+1} ===\n{text}"
            for i, text in enumerate(ocred_pages)
        )

        # Save OCR text if requested
        ocr_file = None
        if save_ocr:
            ocr_file = Path(pdf_path).stem + "_ocr.txt"
            with open(ocr_file, 'w', encoding='utf-8') as f:
                f.write(full_text)
            print(f"[OCR] Saved OCR text to: {ocr_file}")

        # Create extraction attempt record
        attempt_id = None
        if self.use_database:
            # Ensure pdf_id exists
            if pdf_id is None:
                file_hash = ExtractionDatabase.compute_file_hash(pdf_path)
                pdf_id = self.db.get_or_create_pdf(pdf_path, file_hash, len(ocred_pages))

            attempt_id = self.db.create_extraction_attempt(
                pdf_id=pdf_id,
                model=model,
                num_samples=num_samples,
                extraction_method="ocr_text_with_consistency",
                company_name=company_name,
                financial_year=financial_year
            )
            print(f"[DB] ✓ Created extraction attempt #{attempt_id}")
            if company_name:
                print(f"[DB]   Company: {company_name}")
            if financial_year:
                print(f"[DB]   Financial Year: {financial_year}")

        # Stage 2 & 3: Extract with self-consistency
        try:
            result = self._extract_with_consistency_from_text(
                full_text=full_text,
                pdf_path=pdf_path,
                model=model,
                num_samples=num_samples,
                attempt_id=attempt_id
            )

            # Mark attempt as successful
            if self.use_database and attempt_id:
                self.db.complete_extraction_attempt(attempt_id, success=True)

        except Exception as e:
            # Mark attempt as failed
            if self.use_database and attempt_id:
                self.db.complete_extraction_attempt(attempt_id, success=False, error_message=str(e))
            raise

        # Add OCR metadata
        result["ocr_metadata"] = {
            "num_pages": len(ocred_pages),
            "total_characters": len(full_text),
            "ocr_saved_to": ocr_file,
            "pdf_id": pdf_id,
            "attempt_id": attempt_id
        }

        # Add company and financial year metadata
        if company_name or financial_year:
            result["company_metadata"] = {}
            if company_name:
                result["company_metadata"]["company_name"] = company_name
            if financial_year:
                result["company_metadata"]["financial_year"] = financial_year

        return result

    def _extract_with_consistency_from_text(
        self,
        full_text: str,
        pdf_path: str,
        model: str,
        num_samples: int,
        attempt_id: Optional[int] = None
    ) -> Dict:
        """
        Extract financial data from text with self-consistency.

        Runs multiple extraction samples, tracks each in database,
        and computes consensus via majority voting.

        Args:
            full_text: Full OCRed text
            pdf_path: Original PDF path
            model: DeepSeek model to use
            num_samples: Number of samples for self-consistency
            attempt_id: Database attempt ID for tracking

        Returns:
            Dictionary with consensus extraction results and confidence scores
        """
        print(f"\n{'='*70}")
        print(f"SELF-CONSISTENCY EXTRACTION ({num_samples} samples)")
        print(f"{'='*70}")
        print(f"Model: {model}")
        print(f"Text length: {len(full_text):,} characters")

        samples = []
        all_sample_data = []

        # Create text-based extraction prompt
        text_extraction_prompt = f"""You are analyzing OCRed text from a Kenyan company annual report.

{SYSTEM_PROMPT}

OCRed Document Text:
{full_text}

{EXTRACTION_PROMPT}"""

        # Run multiple samples
        for i in range(num_samples):
            print(f"\n[SAMPLE {i + 1}/{num_samples}] Running extraction...")

            try:
                response = self.client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "user", "content": text_extraction_prompt}
                    ],
                    temperature=0.1
                )

                raw_response = response.choices[0].message.content
                prompt_tokens = response.usage.prompt_tokens
                completion_tokens = response.usage.completion_tokens

                # Parse JSON from response
                financial_data = self._parse_json_from_response(raw_response)

                samples.append(financial_data)
                all_sample_data.append({
                    'prompt': text_extraction_prompt,
                    'response': raw_response,
                    'data': financial_data,
                    'prompt_tokens': prompt_tokens,
                    'completion_tokens': completion_tokens
                })

                print(f"[SAMPLE {i + 1}/{num_samples}] ✓ Extracted {len(financial_data)} metrics")
                print(f"[SAMPLE {i + 1}/{num_samples}] Tokens: {prompt_tokens} + {completion_tokens} = {prompt_tokens + completion_tokens}")

                # Save sample to database
                if self.use_database and attempt_id:
                    self.db.save_extraction_sample(
                        attempt_id=attempt_id,
                        sample_number=i + 1,
                        prompt_content=text_extraction_prompt,
                        response_content=raw_response,
                        extracted_data=financial_data,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens
                    )

            except Exception as e:
                print(f"[SAMPLE {i + 1}/{num_samples}] ✗ Failed: {e}")
                # Continue with other samples even if one fails
                continue

        if len(samples) == 0:
            raise RuntimeError("All extraction samples failed")

        print(f"\n[CONSENSUS] Computing consensus from {len(samples)} samples...")

        # Compute consensus
        consensus_data, confidence_metrics = self._compute_consensus(samples)

        # Save consensus results to database
        if self.use_database and attempt_id:
            self.db.save_extraction_results(attempt_id, consensus_data, confidence_metrics)
            print(f"[DB] ✓ Saved consensus results")

        # Calculate total usage
        total_prompt_tokens = sum(s['prompt_tokens'] for s in all_sample_data)
        total_completion_tokens = sum(s['completion_tokens'] for s in all_sample_data)

        # Track costs in database
        if self.use_database and attempt_id:
            # DeepSeek pricing: ~$0.14 per 1M input tokens, ~$0.28 per 1M output tokens
            estimated_cost = (total_prompt_tokens * 0.14 / 1_000_000) + \
                           (total_completion_tokens * 0.28 / 1_000_000)

            self.db.track_usage_cost(
                attempt_id=attempt_id,
                service="deepseek_extraction",
                model=model,
                prompt_tokens=total_prompt_tokens,
                completion_tokens=total_completion_tokens,
                estimated_cost=estimated_cost
            )

        print(f"{'='*70}\n")

        # Return result in standard format
        return {
            "success": True,
            "file": pdf_path,
            "model": model,
            "extraction_method": "ocr_text_with_consistency",
            "num_samples": len(samples),
            "extracted_data": consensus_data,
            "confidence_metrics": confidence_metrics,
            "usage": {
                "prompt_tokens": total_prompt_tokens,
                "completion_tokens": total_completion_tokens,
                "total_tokens": total_prompt_tokens + total_completion_tokens
            }
        }

    def _compute_consensus(self, samples: List[Dict]) -> Tuple[Dict, Dict]:
        """
        Compute consensus from multiple extraction samples using majority voting.

        Args:
            samples: List of extracted data dictionaries

        Returns:
            Tuple of (consensus_data, confidence_metrics)
        """
        if not samples:
            raise ValueError("No samples provided for consensus")

        # Get all metric keys from first sample
        metrics = list(samples[0].keys())

        consensus_data = {}
        confidence_metrics = {}

        for metric in metrics:
            # Collect all values for this metric
            values = [sample.get(metric, "N/A") for sample in samples]

            # Count occurrences
            value_counts = Counter(values)
            most_common_value, count = value_counts.most_common(1)[0]

            # Calculate confidence (agreement percentage)
            confidence = count / len(samples)

            # Determine confidence level
            if confidence == 1.0:
                confidence_level = "high"
            elif confidence >= 0.6:
                confidence_level = "medium"
            else:
                confidence_level = "low"

            consensus_data[metric] = most_common_value

            confidence_metrics[metric] = {
                "final_value": most_common_value,
                "confidence_score": round(confidence, 2),
                "confidence_level": confidence_level,
                "agreement": f"{count}/{len(samples)}",
                "samples": values,
                "vote_distribution": dict(value_counts)
            }

            # Add warning for low confidence
            if confidence < 0.6:
                confidence_metrics[metric]["warning"] = "Low confidence - review manually"

        return consensus_data, confidence_metrics

    def extract_with_self_consistency(
        self,
        pdf_path: str,
        num_samples: int = 3,
        model: str = "deepseek-chat",
        max_pages: Optional[int] = None,
        temperature: float = 0.3,
        delay_between_samples: float = 1.0
    ) -> Dict:
        """
        Extract financial data with self-consistency checking.

        Runs the extraction multiple times and uses majority voting to determine
        the final values, along with confidence scores.

        Args:
            pdf_path: Path to the PDF file
            num_samples: Number of independent extraction runs (default: 3)
            model: DeepSeek model to use
            max_pages: Maximum number of pages to process
            temperature: Sampling temperature (higher = more diversity, default: 0.3)
            delay_between_samples: Delay in seconds between API calls (default: 1.0)

        Returns:
            Dictionary containing consensus results with confidence metrics
        """
        if num_samples < 2:
            raise ValueError("num_samples must be at least 2 for self-consistency")

        if num_samples > 10:
            print(f"Warning: {num_samples} samples will be expensive. Consider using 3-5 samples.")

        print(f"\n{'='*70}")
        print(f"SELF-CONSISTENCY EXTRACTION: {num_samples} samples")
        print(f"{'='*70}")

        # Validate PDF exists
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        # Convert PDF to images once (reuse for all samples)
        print(f"\nFile: {pdf_path}")
        print(f"Model: {model}")
        images = self.pdf_to_images(pdf_path)

        if max_pages:
            images = images[:max_pages]
            print(f"[STEP 2/6] Limited to first {max_pages} pages")
        else:
            print(f"[STEP 2/6] Processing all {len(images)} pages")

        # Prepare content for API (reuse for all samples)
        print(f"[STEP 3/6] Optimizing and encoding images (will reuse for all {num_samples} samples)...")
        content = [{"type": "text", "text": EXTRACTION_PROMPT}]

        total_size_kb = 0
        for idx, image in enumerate(images):
            print(f"[STEP 3/6] Processing page {idx + 1}/{len(images)}")

            # Optimize image
            optimized_image = self.optimize_image(image)

            # Convert to base64
            base64_image, size_kb = self.image_to_base64(optimized_image)
            total_size_kb += size_kb

            print(f"[STEP 3/6]   Page {idx + 1}: {size_kb:.1f} KB")

            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                }
            })

        print(f"[STEP 3/6] ✓ Total payload size: {total_size_kb:.1f} KB ({total_size_kb/1024:.2f} MB)")

        # Check if payload is too large
        if total_size_kb > 10000:  # 10 MB
            print(f"[WARNING] Payload is large ({total_size_kb/1024:.1f} MB). Consider using --max-pages to reduce size.")

        # Run multiple extraction samples
        samples = []
        all_reasoning = []
        total_usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0
        }

        for i in range(num_samples):
            print(f"\n[STEP 4/6] Running sample {i+1}/{num_samples}")
            print(f"[STEP 4/6] Sending request to DeepSeek API...")
            print(f"[STEP 4/6]   - Temperature: {temperature}")
            print(f"[STEP 4/6]   - Payload size: {total_size_kb/1024:.2f} MB")

            try:
                response = self.client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": content}
                    ],
                    temperature=temperature,
                )

                print(f"[STEP 4/6] ✓ Sample {i+1}/{num_samples} received")
                print(f"[STEP 4/6]   Response length: {len(response.choices[0].message.content)} characters")

                # Extract and parse response
                print(f"[STEP 5/6] Parsing sample {i+1}/{num_samples}...")
                raw_response = response.choices[0].message.content
                financial_data = self._parse_json_from_response(raw_response)
                print(f"[STEP 5/6] ✓ Sample {i+1}/{num_samples} parsed successfully")

                samples.append(financial_data)
                all_reasoning.append(raw_response)

                # Accumulate usage
                total_usage["prompt_tokens"] += response.usage.prompt_tokens
                total_usage["completion_tokens"] += response.usage.completion_tokens
                total_usage["total_tokens"] += response.usage.total_tokens

                # Delay between samples (except after last one)
                if i < num_samples - 1:
                    print(f"[STEP 4/6] Waiting {delay_between_samples}s before next sample...")
                    time.sleep(delay_between_samples)

            except Exception as e:
                print(f"\n[ERROR] Sample {i+1}/{num_samples} failed: {type(e).__name__}: {e}")

                # Provide helpful error messages
                error_str = str(e)
                if "413" in error_str or "Request Entity Too Large" in error_str:
                    print(f"\n[SOLUTION] The PDF is too large. Try:")
                    print(f"  1. Use --max-pages to process fewer pages (e.g., --max-pages 10)")
                    print(f"  2. Extract only the financial statement pages from your PDF")
                elif "401" in error_str or "Unauthorized" in error_str:
                    print(f"\n[SOLUTION] API key issue. Check your API key in .env file")
                elif "429" in error_str or "rate limit" in error_str.lower():
                    print(f"\n[SOLUTION] Rate limit exceeded. Wait a few minutes and try again")

                print(f"\n[WARNING] Continuing with {len(samples)} successful samples...")

        if not samples:
            print(f"\n{'='*70}")
            print(f"EXTRACTION FAILED")
            print(f"{'='*70}")
            print(f"[ERROR] All {num_samples} samples failed")
            return {
                "success": False,
                "file": pdf_path,
                "error": "All extraction samples failed"
            }

        # Compute consensus
        print(f"\n[STEP 6/6] Computing consensus from {len(samples)} samples...")
        consensus_data, confidence_metrics = self._compute_consensus(samples)

        # Calculate overall statistics
        high_confidence_count = sum(1 for m in confidence_metrics.values() if m["confidence_level"] == "high")
        medium_confidence_count = sum(1 for m in confidence_metrics.values() if m["confidence_level"] == "medium")
        low_confidence_count = sum(1 for m in confidence_metrics.values() if m["confidence_level"] == "low")

        print(f"[STEP 6/6] ✓ Consensus computed")
        print(f"[STEP 6/6] Confidence breakdown:")
        print(f"[STEP 6/6]   - High: {high_confidence_count}/{len(confidence_metrics)} metrics")
        print(f"[STEP 6/6]   - Medium: {medium_confidence_count}/{len(confidence_metrics)} metrics")
        print(f"[STEP 6/6]   - Low: {low_confidence_count}/{len(confidence_metrics)} metrics")

        print(f"\n{'='*70}")
        print(f"SELF-CONSISTENCY EXTRACTION COMPLETE")
        print(f"{'='*70}")

        return {
            "success": True,
            "file": pdf_path,
            "model": model,
            "self_consistency": {
                "enabled": True,
                "num_samples": len(samples),
                "requested_samples": num_samples,
                "temperature": temperature
            },
            "extracted_data": consensus_data,
            "confidence_metrics": confidence_metrics,
            "statistics": {
                "total_metrics": len(confidence_metrics),
                "high_confidence": high_confidence_count,
                "medium_confidence": medium_confidence_count,
                "low_confidence": low_confidence_count,
                "metrics_with_na": sum(1 for m in consensus_data.values() if m == "N/A")
            },
            "samples": {
                "all_extractions": samples,
                "all_reasoning": all_reasoning
            },
            "usage": total_usage
        }

    @staticmethod
    def batch_extract_from_folder(
        root_folder: str,
        filter_companies: Optional[List[str]] = None,
        max_workers: int = 2,
        deepseek_api_key: Optional[str] = None,
        gemini_api_key: Optional[str] = None,
        db_path: str = "./extractions.db",
        model: str = "deepseek-chat",
        num_samples: int = 3,
        max_pages: Optional[int] = None,
        output_dir: Optional[str] = None
    ) -> Dict:
        """
        Batch extract financial data from multiple PDFs in a folder structure.

        Scans root folder for company subfolders, identifies PDFs with years,
        and processes them in parallel.

        Directory structure expected:
            root_folder/
            ├── Company_A/
            │   ├── report-2021.pdf
            │   ├── report-2022.pdf
            │   └── report-2023.pdf
            └── Company_B/
                └── annual-2023.pdf

        Args:
            root_folder: Path to root folder containing company subfolders
            filter_companies: Optional list of company names to filter (None = all)
            max_workers: Maximum parallel workers (default: 2, recommended: 2-4)
            deepseek_api_key: DeepSeek API key
            gemini_api_key: Gemini API key
            db_path: Path to SQLite database
            model: DeepSeek model to use
            num_samples: Number of samples for self-consistency
            max_pages: Maximum pages per PDF
            output_dir: Optional directory to save individual results JSON files

        Returns:
            Dictionary with batch processing results and statistics
        """
        # Get API keys
        deepseek_key = deepseek_api_key or DEEPSEEK_API_KEY
        gemini_key = gemini_api_key or GEMINI_API_KEY

        if not deepseek_key:
            raise ValueError("DEEPSEEK_API_KEY not found")
        if not gemini_key:
            raise ValueError("GEMINI_API_KEY not found")

        print(f"\n{'='*70}")
        print(f"BATCH EXTRACTION MODE")
        print(f"{'='*70}")
        print(f"Root folder: {root_folder}")
        print(f"Max workers: {max_workers}")
        print(f"Model: {model}")
        print(f"Samples per extraction: {num_samples}")

        # Scan for extraction jobs
        print(f"\n[SCAN] Scanning folder structure...")
        jobs = scan_company_folders(root_folder, filter_companies)

        if not jobs:
            print(f"[SCAN] ✗ No extraction jobs found")
            if filter_companies:
                print(f"[SCAN] Filtered companies: {', '.join(filter_companies)}")
            return {
                "success": False,
                "error": "No extraction jobs found",
                "total_jobs": 0
            }

        print(f"[SCAN] ✓ Found {len(jobs)} extraction jobs")

        # Group jobs by company
        companies_found = {}
        for job in jobs:
            if job.company_name not in companies_found:
                companies_found[job.company_name] = []
            companies_found[job.company_name].append(job.financial_year)

        print(f"[SCAN] Companies: {len(companies_found)}")
        for company, years in sorted(companies_found.items()):
            print(f"[SCAN]   - {company}: {len(years)} reports ({', '.join(sorted(years))})")

        if filter_companies:
            print(f"[SCAN] Filtering: {', '.join(filter_companies)}")

        # Create output directory if specified
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            print(f"[OUTPUT] Results will be saved to: {output_dir}")

        # Process jobs in parallel
        print(f"\n{'='*70}")
        print(f"PROCESSING {len(jobs)} JOBS (max {max_workers} parallel)")
        print(f"{'='*70}\n")

        results = []
        completed = 0
        failed = 0
        start_time = time.time()

        # Use ProcessPoolExecutor for parallel processing
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all jobs
            future_to_job = {
                executor.submit(
                    process_single_extraction,
                    job,
                    deepseek_key,
                    gemini_key,
                    db_path,
                    model,
                    num_samples,
                    max_pages
                ): job
                for job in jobs
            }

            # Process results as they complete
            for future in as_completed(future_to_job):
                job = future_to_job[future]
                try:
                    result_data = future.result()
                    results.append(result_data)

                    if result_data["result"]["success"]:
                        completed += 1
                        print(f"\n[PROGRESS] {completed}/{len(jobs)} completed, {failed} failed")

                        # Save individual result if output_dir specified
                        if output_dir:
                            output_file = Path(output_dir) / f"{job.company_name}_{job.financial_year}.json"
                            with open(output_file, 'w') as f:
                                json.dump(result_data["result"], f, indent=2)
                            print(f"[OUTPUT] Saved: {output_file.name}")
                    else:
                        failed += 1
                        print(f"\n[PROGRESS] {completed}/{len(jobs)} completed, {failed} failed")

                except Exception as e:
                    failed += 1
                    print(f"\n[ERROR] Job failed with exception: {e}")
                    print(f"[PROGRESS] {completed}/{len(jobs)} completed, {failed} failed")
                    results.append({
                        "job": job,
                        "result": {
                            "success": False,
                            "error": str(e),
                            "file": job.pdf_path
                        }
                    })

        elapsed_time = time.time() - start_time

        # Print summary
        print(f"\n{'='*70}")
        print(f"BATCH EXTRACTION COMPLETE")
        print(f"{'='*70}")
        print(f"Total jobs: {len(jobs)}")
        print(f"Successful: {completed}")
        print(f"Failed: {failed}")
        print(f"Time elapsed: {elapsed_time:.1f}s ({elapsed_time/60:.1f} minutes)")
        if completed > 0:
            print(f"Average time per job: {elapsed_time/len(jobs):.1f}s")

        # Group results by company
        print(f"\nRESULTS BY COMPANY:")
        company_results = {}
        for res in results:
            job = res["job"]
            company = job.company_name
            if company not in company_results:
                company_results[company] = {"success": 0, "failed": 0}

            if res["result"]["success"]:
                company_results[company]["success"] += 1
            else:
                company_results[company]["failed"] += 1

        for company, stats in sorted(company_results.items()):
            total = stats["success"] + stats["failed"]
            print(f"  {company}: {stats['success']}/{total} successful")

        return {
            "success": True,
            "total_jobs": len(jobs),
            "successful": completed,
            "failed": failed,
            "elapsed_time": elapsed_time,
            "results": results,
            "company_summary": company_results
        }


def main():
    """Main entry point for the script."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract financial data from PDF annual reports using DeepSeek API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single file extraction
  python pdf_extractor.py report.pdf --company "SGL Limited" --year "2023"

  # Batch extraction from folder (all companies)
  python pdf_extractor.py --folder ./reports --workers 4

  # Batch extraction for specific companies
  python pdf_extractor.py --folder ./reports --companies "SGL Limited" "Company B" --workers 2

  # Batch extraction with output directory
  python pdf_extractor.py --folder ./reports --output-dir ./results --workers 3

  # Export results to CSV
  python pdf_extractor.py --export results.csv --confidence

  # Export specific company/year
  python pdf_extractor.py --export results.csv --company "SGL Limited" --year "2023"

Folder Structure:
  reports/
  ├── SGL Limited/
  │   ├── report-2021.pdf
  │   ├── report-2022.pdf
  │   └── report-2023.pdf
  └── Another Company/
      └── annual-2023.pdf
        """
    )

    # Mode selection (either single file or folder batch)
    parser.add_argument(
        "pdf_path",
        nargs="?",
        help="Path to the PDF file to process (single file mode)"
    )
    parser.add_argument(
        "--folder",
        help="Path to root folder containing company subfolders (batch mode)",
        default=None
    )
    parser.add_argument(
        "--export",
        help="Export database results to CSV file (export mode)",
        default=None
    )
    parser.add_argument(
        "-o", "--output",
        help="Output JSON file path (default: print to stdout)",
        default=None
    )
    parser.add_argument(
        "-m", "--model",
        help="DeepSeek model to use (default: deepseek-chat)",
        default="deepseek-chat"
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        help="Maximum number of pages to process",
        default=None
    )
    parser.add_argument(
        "--api-key",
        help="DeepSeek API key (overrides .env)",
        default=None
    )
    parser.add_argument(
        "--show-reasoning",
        action="store_true",
        help="Display the model's reasoning process"
    )
    parser.add_argument(
        "--self-consistency",
        action="store_true",
        help="Enable self-consistency checking with multiple samples"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        help="Number of samples for self-consistency (default: 3)",
        default=3
    )
    parser.add_argument(
        "--temperature",
        type=float,
        help="Temperature for sampling (default: 0.3)",
        default=0.3
    )
    parser.add_argument(
        "--show-confidence",
        action="store_true",
        help="Show detailed confidence metrics (only with --self-consistency)"
    )
    parser.add_argument(
        "--use-ocr",
        action="store_true",
        help="Use OCR preprocessing (extract text first, then analyze) - Recommended for large PDFs"
    )
    parser.add_argument(
        "--save-ocr",
        action="store_true",
        help="Save OCR text to file (only with --use-ocr)",
        default=True
    )
    parser.add_argument(
        "--company",
        help="Company name for the financial data being extracted",
        default=None
    )
    parser.add_argument(
        "--year",
        help="Financial year for the data being extracted (e.g., '2023', 'FY2022-23')",
        default=None
    )

    # Batch mode arguments
    parser.add_argument(
        "--companies",
        nargs="+",
        help="Filter specific companies to process (batch mode only)",
        default=None
    )
    parser.add_argument(
        "--workers",
        type=int,
        help="Maximum parallel workers for batch processing (default: 2, recommended: 2-4)",
        default=2
    )
    parser.add_argument(
        "--output-dir",
        help="Directory to save individual extraction results (batch mode only)",
        default=None
    )
    parser.add_argument(
        "--batch-summary",
        help="Save batch summary to JSON file (batch mode only)",
        default=None
    )

    # Export mode arguments
    parser.add_argument(
        "--confidence",
        action="store_true",
        help="Include confidence rate columns in CSV export (export mode only)"
    )
    parser.add_argument(
        "--sector",
        help="Sector name for CSV export (export mode only)",
        default=""
    )
    parser.add_argument(
        "--sector-short",
        help="Short sector name for CSV export (export mode only)",
        default=""
    )
    parser.add_argument(
        "--code",
        help="Code field for CSV export (export mode only)",
        default=""
    )
    parser.add_argument(
        "--db",
        help="Database path (default: ./extractions.db)",
        default="./extractions.db"
    )

    args = parser.parse_args()

    # Validate arguments
    if args.export:
        # Export mode - doesn't need pdf_path or folder
        pass
    elif not args.folder and not args.pdf_path:
        parser.error("Either provide a PDF file path, use --folder for batch mode, or use --export for export mode")

    if args.folder and args.pdf_path:
        parser.error("Cannot use both single file mode and --folder batch mode simultaneously")

    if args.export and (args.folder or args.pdf_path):
        parser.error("Cannot use --export with extraction modes (--folder or pdf_path)")

    try:
        # EXPORT MODE: Export database results to CSV
        if args.export:
            from csv_exporter import export_to_csv

            print(f"\n{'='*70}")
            print(f"CSV EXPORT MODE")
            print(f"{'='*70}")
            print(f"Database: {args.db}")
            print(f"Output CSV: {args.export}")
            if args.company:
                print(f"Filter: Company = {args.company}")
            if args.year:
                print(f"Filter: Year = {args.year}")
            if args.confidence:
                print(f"Include confidence rates: Yes")

            count = export_to_csv(
                db_path=args.db,
                output_file=args.export,
                include_confidence=args.confidence,
                company_filter=args.company,
                year_filter=args.year,
                sector=args.sector,
                sector_short=args.sector_short,
                code=args.code
            )

            print(f"\n{'='*70}")
            print(f"EXPORT COMPLETE")
            print(f"{'='*70}")
            print(f"Records exported: {count}")
            print(f"Output file: {args.export}")

            return 0

        # BATCH MODE: Process entire folder
        elif args.folder:
            result = PDFFinancialExtractor.batch_extract_from_folder(
                root_folder=args.folder,
                filter_companies=args.companies,
                max_workers=args.workers,
                deepseek_api_key=args.api_key,
                gemini_api_key=None,  # Will use env var
                model=args.model,
                num_samples=args.num_samples,
                max_pages=args.max_pages,
                output_dir=args.output_dir
            )

            # Save batch summary if requested
            if args.batch_summary and result["success"]:
                with open(args.batch_summary, 'w') as f:
                    json.dump(result, f, indent=2)
                print(f"\n[OUTPUT] Batch summary saved to: {args.batch_summary}")

            return 0 if result["success"] and result["failed"] == 0 else 1

        # SINGLE FILE MODE: Process one PDF
        else:
            # Initialize extractor
            extractor = PDFFinancialExtractor(api_key=args.api_key)

            # In single file mode, always use OCR if not explicitly using other methods
            if args.use_ocr or (not args.self_consistency):
                # OCR preprocessing method (default and recommended)
                result = extractor.extract_with_ocr_preprocessing(
                    pdf_path=args.pdf_path,
                    max_pages=args.max_pages,
                    model=args.model,
                    save_ocr=args.save_ocr,
                    num_samples=args.num_samples,
                    company_name=args.company,
                    financial_year=args.year
                )
            elif args.self_consistency:
                result = extractor.extract_with_self_consistency(
                    pdf_path=args.pdf_path,
                    num_samples=args.num_samples,
                    model=args.model,
                    max_pages=args.max_pages,
                    temperature=args.temperature
                )
            else:
                result = extractor.extract_financial_data(
                    pdf_path=args.pdf_path,
                    model=args.model,
                    max_pages=args.max_pages
                )

            # Display results (single file mode only)
            if result["success"]:
                print("\n" + "="*60)
                if result.get("self_consistency", {}).get("enabled"):
                    print("SELF-CONSISTENCY EXTRACTION SUCCESSFUL")
                else:
                    print("EXTRACTION SUCCESSFUL")
                print("="*60)

                # Show company metadata if available
                if result.get("company_metadata"):
                    print("\nCOMPANY INFORMATION:")
                    if result["company_metadata"].get("company_name"):
                        print(f"  Company: {result['company_metadata']['company_name']}")
                    if result["company_metadata"].get("financial_year"):
                        print(f"  Financial Year: {result['company_metadata']['financial_year']}")

                # Show self-consistency statistics
                if result.get("self_consistency", {}).get("enabled"):
                    sc = result["self_consistency"]
                    stats = result.get("statistics", {})
                    print(f"\nSELF-CONSISTENCY MODE:")
                    print(f"  Samples: {sc['num_samples']}/{sc['requested_samples']}")
                    print(f"  Temperature: {sc['temperature']}")
                    print(f"\nCONFIDENCE SUMMARY:")
                    print(f"  High confidence: {stats.get('high_confidence', 0)}/{stats.get('total_metrics', 0)} metrics")
                    print(f"  Medium confidence: {stats.get('medium_confidence', 0)}/{stats.get('total_metrics', 0)} metrics")
                    print(f"  Low confidence: {stats.get('low_confidence', 0)}/{stats.get('total_metrics', 0)} metrics")

                if args.show_reasoning:
                    if result.get("self_consistency", {}).get("enabled"):
                        print("\nMODEL REASONING (Sample 1):")
                        print("-"*60)
                        print(result["samples"]["all_reasoning"][0])
                        print("-"*60)
                    else:
                        print("\nMODEL REASONING:")
                        print("-"*60)
                        print(result["reasoning"])
                        print("-"*60)

                print("\nEXTRACTED FINANCIAL DATA:")
                print("-"*60)
                print(json.dumps(result["extracted_data"], indent=2))

                # Show detailed confidence metrics if requested
                if args.show_confidence and result.get("confidence_metrics"):
                    print("\nDETAILED CONFIDENCE METRICS:")
                    print("-"*60)
                    for metric, details in result["confidence_metrics"].items():
                        print(f"\n{metric}:")
                        print(f"  Value: {details['final_value']}")
                        print(f"  Confidence: {details['confidence_score']} ({details['confidence_level']})")
                        print(f"  Agreement: {details['agreement']}")
                        if len(details.get('vote_distribution', {})) > 1:
                            print(f"  Vote distribution: {details['vote_distribution']}")
                        if 'warning' in details:
                            print(f"  ⚠️  {details['warning']}")

                print("\nAPI USAGE:")
                print(f"  Prompt tokens: {result['usage']['prompt_tokens']:,}")
                print(f"  Completion tokens: {result['usage']['completion_tokens']:,}")
                print(f"  Total tokens: {result['usage']['total_tokens']:,}")

                # Save to file if specified
                if args.output:
                    with open(args.output, 'w') as f:
                        json.dump(result, f, indent=2)
                    print(f"\nFull results saved to: {args.output}")
            else:
                print("\n" + "="*60)
                print("EXTRACTION FAILED")
                print("="*60)
                print(f"Error: {result['error']}")
                return 1

            return 0

    except Exception as e:
        print(f"\nError: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
