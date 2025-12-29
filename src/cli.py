#!/usr/bin/env python3
"""
Unified CLI for PDF Financial Data Extractor
Supports both interactive mode and command-line arguments.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Optional, List, Dict

# PDF extractor will be imported lazily when needed
PDFFinancialExtractor = None


def run_single_file_extraction(
    pdf_path: str,
    company_name: Optional[str] = None,
    financial_year: Optional[str] = None,
    model: str = "deepseek-chat",
    num_samples: int = 3,
    max_pages: Optional[int] = None,
    output_file: Optional[str] = None,
    show_reasoning: bool = False,
    show_confidence: bool = False,
    api_key: Optional[str] = None
) -> int:
    """
    Run single file extraction.

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    # Lazy import
    from pdf_extractor import PDFFinancialExtractor

    try:
        # Initialize extractor
        extractor = PDFFinancialExtractor(api_key=api_key)

        # Always use OCR preprocessing for best results
        result = extractor.extract_with_ocr_preprocessing(
            pdf_path=pdf_path,
            max_pages=max_pages,
            model=model,
            save_ocr=True,
            num_samples=num_samples,
            company_name=company_name,
            financial_year=financial_year
        )

        # Display results
        if result["success"]:
            print("\n" + "="*60)
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
                print(f"\nCONFIDENCE SUMMARY:")
                print(f"  High confidence: {stats.get('high_confidence', 0)}/{stats.get('total_metrics', 0)} metrics")
                print(f"  Medium confidence: {stats.get('medium_confidence', 0)}/{stats.get('total_metrics', 0)} metrics")
                print(f"  Low confidence: {stats.get('low_confidence', 0)}/{stats.get('total_metrics', 0)} metrics")

            if show_reasoning:
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
            if show_confidence and result.get("confidence_metrics"):
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
            if output_file:
                with open(output_file, 'w') as f:
                    json.dump(result, f, indent=2)
                print(f"\nFull results saved to: {output_file}")

            return 0
        else:
            print("\n" + "="*60)
            print("EXTRACTION FAILED")
            print("="*60)
            print(f"Error: {result['error']}")
            return 1

    except Exception as e:
        print(f"\nError: {e}")
        return 1


def run_batch_extraction(
    root_folder: str,
    filter_companies: Optional[List[str]] = None,
    max_workers: int = 2,
    model: str = "deepseek-chat",
    num_samples: int = 3,
    max_pages: Optional[int] = None,
    output_dir: Optional[str] = None,
    batch_summary: Optional[str] = None,
    api_key: Optional[str] = None
) -> int:
    """
    Run batch extraction from folder.

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    # Lazy import
    from pdf_extractor import PDFFinancialExtractor

    result = PDFFinancialExtractor.batch_extract_from_folder(
        root_folder=root_folder,
        filter_companies=filter_companies,
        max_workers=max_workers,
        deepseek_api_key=api_key,
        gemini_api_key=None,  # Will use env var
        model=model,
        num_samples=num_samples,
        max_pages=max_pages,
        output_dir=output_dir
    )

    # Save batch summary if requested
    if batch_summary and result["success"]:
        with open(batch_summary, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\n[OUTPUT] Batch summary saved to: {batch_summary}")

    return 0 if result["success"] and result["failed"] == 0 else 1


def run_csv_export(
    output_file: str,
    include_confidence: bool = False,
    company_filter: Optional[str] = None,
    year_filter: Optional[str] = None,
    sector: str = "",
    sector_short: str = "",
    code: str = "",
    db_path: str = "./extractions.db"
) -> int:
    """
    Run CSV export.

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    from csv_exporter import export_to_csv

    print(f"\n{'='*70}")
    print(f"CSV EXPORT MODE")
    print(f"{'='*70}")
    print(f"Database: {db_path}")
    print(f"Output CSV: {output_file}")
    if company_filter:
        print(f"Filter: Company = {company_filter}")
    if year_filter:
        print(f"Filter: Year = {year_filter}")
    if include_confidence:
        print(f"Include confidence rates: Yes")

    count = export_to_csv(
        db_path=db_path,
        output_file=output_file,
        include_confidence=include_confidence,
        company_filter=company_filter,
        year_filter=year_filter,
        sector=sector,
        sector_short=sector_short,
        code=code
    )

    print(f"\n{'='*70}")
    print(f"EXPORT COMPLETE")
    print(f"{'='*70}")
    print(f"Records exported: {count}")
    print(f"Output file: {output_file}")

    return 0


def run_interactive_mode():
    """
    Run interactive mode using Rich/questionary.
    """
    try:
        from rich.console import Console
        from rich.panel import Panel
        from rich.table import Table
        from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
        from rich.prompt import Prompt, Confirm
        from rich.syntax import Syntax
        from rich.tree import Tree
        from rich import box
        import questionary
        from questionary import Style
    except ImportError as e:
        print(f"Interactive mode requires additional packages: {e}")
        print("Please install: pip install rich questionary")
        return 1

    # Import the interactive CLI class from backup
    # For now, print a message
    console = Console()
    console.print("\n[bold yellow]Interactive Mode[/bold yellow]")
    console.print("Interactive mode is currently being refactored.")
    console.print("Please use command-line mode for now.\n")
    console.print("Examples:")
    console.print("  python cli.py report.pdf --company 'SGL Limited' --year 2023")
    console.print("  python cli.py --folder ./reports --workers 4")
    console.print("  python cli.py --export results.csv --confidence")
    console.print("\nFor help: python cli.py --help\n")
    return 0


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="PDF Financial Data Extractor - Extract financial metrics from annual reports",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES:

  Single File Extraction:
    python cli.py report.pdf --company "SGL Limited" --year "2023"
    python cli.py report.pdf --output results.json

  Batch Extraction:
    python cli.py --folder ./reports --workers 4
    python cli.py --folder ./reports --companies "SGL Limited" "Company B"
    python cli.py --folder ./reports --output-dir ./results

  CSV Export:
    python cli.py --export results.csv
    python cli.py --export results.csv --confidence
    python cli.py --export results.csv --company "SGL Limited" --year "2023"

  Interactive Mode:
    python cli.py --interactive

FOLDER STRUCTURE (for batch mode):
  reports/
  ├── SGL Limited/
  │   ├── report-2021.pdf
  │   ├── report-2022.pdf
  │   └── report-2023.pdf
  └── Another Company/
      └── annual-2023.pdf
        """
    )

    # Mode selection
    parser.add_argument(
        "pdf_path",
        nargs="?",
        help="Path to PDF file (single file mode)"
    )
    parser.add_argument(
        "--folder",
        help="Root folder for batch extraction",
        default=None
    )
    parser.add_argument(
        "--export",
        help="Export to CSV file",
        default=None
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode"
    )

    # Common arguments
    parser.add_argument(
        "--api-key",
        help="DeepSeek API key (or set DEEPSEEK_API_KEY env var)",
        default=os.getenv("DEEPSEEK_API_KEY")
    )
    parser.add_argument(
        "--model",
        help="DeepSeek model to use",
        default="deepseek-chat"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        help="Number of samples for self-consistency",
        default=3
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        help="Maximum pages to process",
        default=None
    )

    # Single file arguments
    parser.add_argument(
        "--company",
        help="Company name",
        default=None
    )
    parser.add_argument(
        "--year",
        help="Financial year",
        default=None
    )
    parser.add_argument(
        "-o", "--output",
        help="Output JSON file (single file mode)",
        default=None
    )
    parser.add_argument(
        "--show-reasoning",
        action="store_true",
        help="Show model reasoning"
    )
    parser.add_argument(
        "--show-confidence",
        action="store_true",
        help="Show detailed confidence metrics"
    )

    # Batch mode arguments
    parser.add_argument(
        "--companies",
        nargs="+",
        help="Filter specific companies (batch mode)",
        default=None
    )
    parser.add_argument(
        "--workers",
        type=int,
        help="Max parallel workers (batch mode)",
        default=2
    )
    parser.add_argument(
        "--output-dir",
        help="Directory for batch results (batch mode)",
        default=None
    )
    parser.add_argument(
        "--batch-summary",
        help="Batch summary JSON file (batch mode)",
        default=None
    )

    # Export mode arguments
    parser.add_argument(
        "--confidence",
        action="store_true",
        help="Include confidence columns (export mode)"
    )
    parser.add_argument(
        "--sector",
        help="Sector name (export mode)",
        default=""
    )
    parser.add_argument(
        "--sector-short",
        help="Short sector code (export mode)",
        default=""
    )
    parser.add_argument(
        "--code",
        help="Code field (export mode)",
        default=""
    )
    parser.add_argument(
        "--db",
        help="Database path",
        default="./extractions.db"
    )

    args = parser.parse_args()

    # Determine mode and run
    try:
        # Interactive mode
        if args.interactive:
            return run_interactive_mode()

        # Export mode
        elif args.export:
            return run_csv_export(
                output_file=args.export,
                include_confidence=args.confidence,
                company_filter=args.company,
                year_filter=args.year,
                sector=args.sector,
                sector_short=args.sector_short,
                code=args.code,
                db_path=args.db
            )

        # Batch mode
        elif args.folder:
            return run_batch_extraction(
                root_folder=args.folder,
                filter_companies=args.companies,
                max_workers=args.workers,
                model=args.model,
                num_samples=args.num_samples,
                max_pages=args.max_pages,
                output_dir=args.output_dir,
                batch_summary=args.batch_summary,
                api_key=args.api_key
            )

        # Single file mode
        elif args.pdf_path:
            return run_single_file_extraction(
                pdf_path=args.pdf_path,
                company_name=args.company,
                financial_year=args.year,
                model=args.model,
                num_samples=args.num_samples,
                max_pages=args.max_pages,
                output_file=args.output,
                show_reasoning=args.show_reasoning,
                show_confidence=args.show_confidence,
                api_key=args.api_key
            )

        # No mode specified
        else:
            parser.print_help()
            return 0

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
