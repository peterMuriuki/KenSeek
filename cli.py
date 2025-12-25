#!/usr/bin/env python3
"""
Interactive CLI wrapper for PDF Financial Data Extractor
Provides a user-friendly interface for extracting financial data from PDFs.
"""

import os
import sys
import json
from pathlib import Path
from typing import Optional, List, Dict
from datetime import datetime

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
    print(f"Missing required package: {e}")
    print("Please install CLI requirements: pip install -r requirements.txt")
    sys.exit(1)

from pdf_extractor import PDFFinancialExtractor

# Initialize Rich console
console = Console()

# Custom style for questionary prompts
custom_style = Style([
    ('qmark', 'fg:#673ab7 bold'),
    ('question', 'bold'),
    ('answer', 'fg:#2196f3 bold'),
    ('pointer', 'fg:#673ab7 bold'),
    ('highlighted', 'fg:#673ab7 bold'),
    ('selected', 'fg:#4caf50'),
    ('separator', 'fg:#cc5454'),
    ('instruction', 'fg:#858585'),
])


class FinancialExtractorCLI:
    """Interactive CLI wrapper for PDF Financial Extractor."""

    def __init__(self):
        self.console = console
        self.extractor = None
        self.api_key = os.getenv("DEEPSEEK_API_KEY")
        self.output_dir = Path("output")
        self.output_dir.mkdir(exist_ok=True)

    def show_banner(self):
        """Display welcome banner."""
        banner = """
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║      PDF Financial Data Extractor - DeepSeek Edition         ║
║                                                               ║
║   Extract 18 financial metrics from Kenyan annual reports    ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝
        """
        self.console.print(banner, style="bold cyan")

    def check_api_key(self) -> bool:
        """Check if API key is configured."""
        if self.api_key:
            self.console.print("✓ API key found", style="green")
            return True
        else:
            self.console.print("✗ API key not found", style="red")
            self.console.print("\nPlease configure your DeepSeek API key:", style="yellow")
            self.console.print("1. Create a .env file in the current directory")
            self.console.print("2. Add: DEEPSEEK_API_KEY=your_api_key_here")
            self.console.print("\nOr enter it now (will not be saved):")

            api_key = Prompt.ask("DeepSeek API Key", password=True)
            if api_key:
                self.api_key = api_key
                return True
            return False

    def initialize_extractor(self):
        """Initialize the PDF extractor."""
        try:
            self.extractor = PDFFinancialExtractor(api_key=self.api_key)
            self.console.print("✓ Extractor initialized", style="green")
            return True
        except Exception as e:
            self.console.print(f"✗ Failed to initialize: {e}", style="red")
            return False

    def select_pdf_files(self) -> List[str]:
        """Interactive PDF file selection."""
        self.console.print("\n[bold]PDF File Selection[/bold]")

        choices = [
            "Enter file path manually",
            "Select from current directory",
            "Batch process multiple files"
        ]

        selection = questionary.select(
            "How would you like to select PDF files?",
            choices=choices,
            style=custom_style
        ).ask()

        if selection == "Enter file path manually":
            path = questionary.path(
                "Enter PDF file path:",
                only_directories=False,
                style=custom_style
            ).ask()
            return [path] if path else []

        elif selection == "Select from current directory":
            pdf_files = list(Path.cwd().glob("**/*.pdf"))
            if not pdf_files:
                self.console.print("No PDF files found in current directory", style="yellow")
                return []

            file_choices = [str(f.relative_to(Path.cwd())) for f in pdf_files]
            selected = questionary.select(
                "Select a PDF file:",
                choices=file_choices,
                style=custom_style
            ).ask()
            return [selected] if selected else []

        elif selection == "Batch process multiple files":
            pdf_files = list(Path.cwd().glob("**/*.pdf"))
            if not pdf_files:
                self.console.print("No PDF files found in current directory", style="yellow")
                return []

            file_choices = [str(f.relative_to(Path.cwd())) for f in pdf_files]
            selected = questionary.checkbox(
                "Select PDF files (use space to select, enter to confirm):",
                choices=file_choices,
                style=custom_style
            ).ask()
            return selected if selected else []

        return []

    def configure_extraction(self) -> Dict:
        """Configure extraction parameters."""
        self.console.print("\n[bold]Extraction Configuration[/bold]")

        config = {}

        # Model selection
        config['model'] = questionary.select(
            "Select DeepSeek model:",
            choices=[
                "deepseek-chat (Recommended)",
                "deepseek-reasoner"
            ],
            style=custom_style
        ).ask().split(" ")[0]

        # Extraction method
        config['use_ocr'] = questionary.confirm(
            "Use OCR preprocessing? (Recommended for large PDFs - extracts text first, 90%+ smaller payload)",
            default=False,
            style=custom_style
        ).ask()

        # Max pages
        limit_pages = questionary.confirm(
            "Limit number of pages to process?",
            default=False,
            style=custom_style
        ).ask()

        if limit_pages:
            config['max_pages'] = questionary.text(
                "Maximum pages to process:",
                default="20",
                validate=lambda x: x.isdigit() and int(x) > 0,
                style=custom_style
            ).ask()
            config['max_pages'] = int(config['max_pages'])
        else:
            config['max_pages'] = None

        # Self-consistency (not compatible with OCR mode currently)
        if not config['use_ocr']:
            config['self_consistency'] = questionary.confirm(
                "Enable self-consistency checking? (More accurate but slower and more expensive)",
                default=False,
                style=custom_style
            ).ask()
        else:
            # OCR mode doesn't support self-consistency yet
            config['self_consistency'] = False

        if config['self_consistency']:
            config['num_samples'] = questionary.select(
                "Number of samples for consensus:",
                choices=["3 (Recommended)", "5 (Better accuracy)", "7 (Maximum accuracy)"],
                style=custom_style
            ).ask().split(" ")[0]
            config['num_samples'] = int(config['num_samples'])

            config['show_confidence'] = questionary.confirm(
                "Show detailed confidence metrics?",
                default=True,
                style=custom_style
            ).ask()
        else:
            config['num_samples'] = 3
            config['show_confidence'] = False

        # Show reasoning
        config['show_reasoning'] = questionary.confirm(
            "Show model's reasoning process?",
            default=True,
            style=custom_style
        ).ask()

        # Save output
        config['save_output'] = questionary.confirm(
            "Save results to file?",
            default=True,
            style=custom_style
        ).ask()

        return config

    def extract_from_pdf(self, pdf_path: str, config: Dict) -> Optional[Dict]:
        """Extract financial data from a single PDF with progress tracking."""
        self.console.print(f"\n[bold cyan]Processing:[/bold cyan] {pdf_path}")

        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                console=self.console
            ) as progress:

                if config.get('use_ocr'):
                    # OCR preprocessing mode
                    task1 = progress.add_task("[cyan]OCR preprocessing (Stage 1/2)...", total=None)

                    # Extract using OCR
                    result = self.extractor.extract_with_ocr_preprocessing(
                        pdf_path=pdf_path,
                        max_pages=config.get('max_pages'),
                        model=config['model'],
                        save_ocr=True
                    )

                    progress.update(task1, completed=True)

                elif config.get('self_consistency'):
                    # Self-consistency mode - multiple samples
                    task_desc = f"[cyan]Running {config['num_samples']} extraction samples..."
                    task1 = progress.add_task(task_desc, total=None)

                    # Extract data with self-consistency
                    result = self.extractor.extract_with_self_consistency(
                        pdf_path=pdf_path,
                        num_samples=config['num_samples'],
                        model=config['model'],
                        max_pages=config.get('max_pages')
                    )

                    progress.update(task1, completed=True)
                else:
                    # Standard single extraction
                    task1 = progress.add_task("[cyan]Converting PDF to images...", total=None)
                    task2 = progress.add_task("[cyan]Encoding images...", total=None, start=False)
                    task3 = progress.add_task("[cyan]Calling DeepSeek API...", total=None, start=False)
                    task4 = progress.add_task("[cyan]Parsing results...", total=None, start=False)

                    # Extract data
                    result = self.extractor.extract_financial_data(
                        pdf_path=pdf_path,
                        model=config['model'],
                        max_pages=config.get('max_pages')
                    )

                    # Update progress
                    progress.update(task1, completed=True)
                    progress.start_task(task2)
                    progress.update(task2, completed=True)
                    progress.start_task(task3)
                    progress.update(task3, completed=True)
                    progress.start_task(task4)
                    progress.update(task4, completed=True)

            return result

        except Exception as e:
            self.console.print(f"[red]Error processing {pdf_path}: {e}[/red]")
            return None

    def display_results(self, result: Dict, config: Dict):
        """Display extraction results in a formatted way."""
        if not result or not result.get('success'):
            self.console.print("\n[red]✗ Extraction Failed[/red]")
            if result:
                self.console.print(f"Error: {result.get('error', 'Unknown error')}")
            return

        self.console.print("\n[green]✓ Extraction Successful[/green]")

        # Show self-consistency statistics if enabled
        if result.get("self_consistency", {}).get("enabled"):
            sc = result["self_consistency"]
            stats = result.get("statistics", {})

            sc_panel = Panel(
                f"[cyan]Samples:[/cyan] {sc['num_samples']}/{sc['requested_samples']}\n"
                f"[cyan]Temperature:[/cyan] {sc['temperature']}\n\n"
                f"[yellow]Confidence Summary:[/yellow]\n"
                f"  High: {stats.get('high_confidence', 0)}/{stats.get('total_metrics', 0)} metrics\n"
                f"  Medium: {stats.get('medium_confidence', 0)}/{stats.get('total_metrics', 0)} metrics\n"
                f"  Low: {stats.get('low_confidence', 0)}/{stats.get('total_metrics', 0)} metrics",
                title="Self-Consistency Mode",
                border_style="magenta"
            )
            self.console.print("\n", sc_panel)

        # Show OCR metadata if OCR was used
        if result.get("extraction_method") == "ocr_text":
            ocr_meta = result.get("ocr_metadata", {})

            ocr_panel = Panel(
                f"[cyan]Pages OCRed:[/cyan] {ocr_meta.get('num_pages', 'N/A')}\n"
                f"[cyan]Total Characters:[/cyan] {ocr_meta.get('total_characters', 0):,}\n"
                f"[cyan]OCR Text Saved:[/cyan] {ocr_meta.get('ocr_saved_to', 'No')}\n\n"
                f"[yellow]Benefits:[/yellow]\n"
                f"  ✓ 90%+ smaller payload vs images\n"
                f"  ✓ Can process 50+ page documents\n"
                f"  ✓ Lower API costs",
                title="OCR Preprocessing Mode",
                border_style="green"
            )
            self.console.print("\n", ocr_panel)

        # Display extracted data in a table
        data = result['extracted_data']
        confidence_metrics = result.get('confidence_metrics', {})

        table = Table(
            title="Extracted Financial Metrics (KES)",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold magenta"
        )

        table.add_column("Metric", style="cyan", no_wrap=True)
        table.add_column("Value", style="green", justify="right")

        # Add confidence column if self-consistency is enabled
        if confidence_metrics:
            table.add_column("Confidence", style="yellow", justify="center")
            table.add_column("Status", style="yellow", justify="center")
        else:
            table.add_column("Status", style="yellow", justify="center")

        # Financial metrics with descriptions
        metrics = {
            'credit_sales': 'Credit Sales',
            'net_sales': 'Net Sales',
            'debtors': 'Debtors/Receivables',
            'impaired_debts': 'Impaired Debts',
            'current_assets': 'Current Assets',
            'current_liabilities': 'Current Liabilities',
            'cash': 'Cash & Equivalents',
            'marketable_securities': 'Marketable Securities',
            'total_liabilities': 'Total Liabilities',
            'total_assets': 'Total Assets',
            'ebit': 'EBIT',
            'interest': 'Interest Expense',
            'labour_cost': 'Labour Cost',
            'operating_expenses': 'Operating Expenses',
            'rd_cost': 'R&D Cost',
            'tax': 'Tax Expense',
            'equity': 'Equity',
            'net_income': 'Net Income'
        }

        for key, label in metrics.items():
            value = data.get(key, 'N/A')
            status = "✓" if value != "N/A" else "○"

            # Format value with commas if it's a number
            if value != "N/A":
                try:
                    formatted_value = f"{int(value):,}"
                except:
                    formatted_value = value
            else:
                formatted_value = "N/A"

            # Add row with or without confidence info
            if confidence_metrics and key in confidence_metrics:
                conf = confidence_metrics[key]
                conf_level = conf['confidence_level']
                conf_display = f"{conf['confidence_score']:.0%} ({conf_level[0].upper()})"

                # Add warning emoji for low confidence
                if conf_level == "low":
                    conf_display = f"⚠️ {conf_display}"

                table.add_row(label, formatted_value, conf_display, status)
            else:
                table.add_row(label, formatted_value, status)

        self.console.print("\n", table)

        # Display API usage
        usage = result.get('usage', {})
        usage_panel = Panel(
            f"[cyan]Prompt Tokens:[/cyan] {usage.get('prompt_tokens', 0):,}\n"
            f"[cyan]Completion Tokens:[/cyan] {usage.get('completion_tokens', 0):,}\n"
            f"[cyan]Total Tokens:[/cyan] {usage.get('total_tokens', 0):,}",
            title="API Usage",
            border_style="blue"
        )
        self.console.print("\n", usage_panel)

        # Show detailed confidence metrics if requested
        if config.get('show_confidence') and confidence_metrics:
            self.console.print("\n[bold magenta]Detailed Confidence Metrics:[/bold magenta]")

            for metric_key, label in list(metrics.items())[:5]:  # Show first 5 as examples
                if metric_key in confidence_metrics:
                    conf = confidence_metrics[metric_key]
                    conf_text = (
                        f"[cyan]{label}:[/cyan]\n"
                        f"  Final Value: {conf['final_value']}\n"
                        f"  Confidence: {conf['confidence_score']:.0%} ({conf['confidence_level']})\n"
                        f"  Agreement: {conf['agreement']}\n"
                    )

                    if len(conf.get('vote_distribution', {})) > 1:
                        conf_text += f"  Votes: {conf['vote_distribution']}\n"

                    if 'warning' in conf:
                        conf_text += f"  ⚠️ {conf['warning']}"

                    self.console.print(Panel(conf_text, border_style="dim"))

            if len(metrics) > 5:
                self.console.print(f"\n[dim]... and {len(metrics) - 5} more metrics (see full output file for details)[/dim]")

        # Show reasoning if requested
        if config.get('show_reasoning'):
            if result.get("self_consistency", {}).get("enabled"):
                # Show reasoning from first sample
                if result.get("samples", {}).get("all_reasoning"):
                    self.console.print("\n[bold]Model Reasoning (Sample 1):[/bold]")
                    reasoning_text = result["samples"]["all_reasoning"][0][:2000]
                    if len(result["samples"]["all_reasoning"][0]) > 2000:
                        reasoning_text += "\n... (truncated)"
                    self.console.print(Panel(reasoning_text, border_style="dim"))
            elif result.get('reasoning'):
                self.console.print("\n[bold]Model Reasoning:[/bold]")
                reasoning_text = result['reasoning'][:2000]  # Limit display
                if len(result['reasoning']) > 2000:
                    reasoning_text += "\n... (truncated)"
                self.console.print(Panel(reasoning_text, border_style="dim"))

        # Save to file if requested
        if config.get('save_output'):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            pdf_name = Path(result['file']).stem
            output_file = self.output_dir / f"{pdf_name}_{timestamp}.json"

            with open(output_file, 'w') as f:
                json.dump(result, f, indent=2)

            self.console.print(f"\n[green]✓ Results saved to:[/green] {output_file}")

    def batch_process(self, pdf_files: List[str], config: Dict):
        """Process multiple PDF files."""
        self.console.print(f"\n[bold]Batch Processing {len(pdf_files)} files[/bold]")

        results = []
        successful = 0
        failed = 0

        for idx, pdf_file in enumerate(pdf_files, 1):
            self.console.print(f"\n[bold]File {idx}/{len(pdf_files)}[/bold]")
            result = self.extract_from_pdf(pdf_file, config)

            if result and result.get('success'):
                successful += 1
                results.append(result)
                self.display_results(result, config)
            else:
                failed += 1

            # Pause between files
            if idx < len(pdf_files):
                if not Confirm.ask("\nContinue to next file?", default=True):
                    break

        # Summary
        summary = Table(title="Batch Processing Summary", box=box.DOUBLE)
        summary.add_column("Metric", style="cyan")
        summary.add_column("Count", justify="right", style="green")
        summary.add_row("Total Files", str(len(pdf_files)))
        summary.add_row("Successful", str(successful))
        summary.add_row("Failed", str(failed))

        self.console.print("\n", summary)

    def run_interactive(self):
        """Run the CLI in interactive mode."""
        self.show_banner()

        # Check API key
        if not self.check_api_key():
            return

        # Initialize extractor
        if not self.initialize_extractor():
            return

        while True:
            # Select PDF files
            pdf_files = self.select_pdf_files()

            if not pdf_files:
                self.console.print("[yellow]No files selected[/yellow]")
                if not Confirm.ask("\nTry again?", default=True):
                    break
                continue

            # Configure extraction
            config = self.configure_extraction()

            # Process files
            if len(pdf_files) == 1:
                result = self.extract_from_pdf(pdf_files[0], config)
                self.display_results(result, config)
            else:
                self.batch_process(pdf_files, config)

            # Continue?
            if not Confirm.ask("\n\nProcess more files?", default=False):
                break

        self.console.print("\n[bold cyan]Thank you for using PDF Financial Extractor![/bold cyan]")

    def run_quick(self, pdf_path: str, **kwargs):
        """Run extraction in quick mode (non-interactive)."""
        # Check API key
        if not self.check_api_key():
            return False

        # Initialize extractor
        if not self.initialize_extractor():
            return False

        config = {
            'model': kwargs.get('model', 'deepseek-chat'),
            'max_pages': kwargs.get('max_pages'),
            'show_reasoning': kwargs.get('show_reasoning', False),
            'save_output': kwargs.get('save_output', True),
            'use_ocr': kwargs.get('use_ocr', False),
            'self_consistency': kwargs.get('self_consistency', False),
            'num_samples': kwargs.get('num_samples', 3),
            'show_confidence': kwargs.get('show_confidence', False)
        }

        result = self.extract_from_pdf(pdf_path, config)
        self.display_results(result, config)

        return result and result.get('success')


def main():
    """Main entry point for CLI."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Interactive CLI for PDF Financial Data Extraction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode (recommended)
  python cli.py

  # Quick mode with specific file
  python cli.py --file report.pdf

  # Quick mode with options
  python cli.py --file report.pdf --model deepseek-chat --max-pages 20 --show-reasoning
        """
    )

    parser.add_argument(
        '-f', '--file',
        help='PDF file to process (enables quick mode)',
        default=None
    )
    parser.add_argument(
        '-m', '--model',
        help='DeepSeek model to use (default: deepseek-chat)',
        default='deepseek-chat'
    )
    parser.add_argument(
        '--max-pages',
        type=int,
        help='Maximum number of pages to process',
        default=None
    )
    parser.add_argument(
        '--show-reasoning',
        action='store_true',
        help='Display model reasoning process'
    )
    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Do not save results to file'
    )
    parser.add_argument(
        '--self-consistency',
        action='store_true',
        help='Enable self-consistency checking with multiple samples'
    )
    parser.add_argument(
        '--num-samples',
        type=int,
        help='Number of samples for self-consistency (default: 3)',
        default=3
    )
    parser.add_argument(
        '--show-confidence',
        action='store_true',
        help='Show detailed confidence metrics (only with --self-consistency)'
    )
    parser.add_argument(
        '--use-ocr',
        action='store_true',
        help='Use OCR preprocessing (90%+ smaller payload, recommended for large PDFs)'
    )

    args = parser.parse_args()

    # Create CLI instance
    cli = FinancialExtractorCLI()

    # Run in appropriate mode
    if args.file:
        # Quick mode
        success = cli.run_quick(
            pdf_path=args.file,
            model=args.model,
            max_pages=args.max_pages,
            show_reasoning=args.show_reasoning,
            save_output=not args.no_save,
            use_ocr=args.use_ocr,
            self_consistency=args.self_consistency,
            num_samples=args.num_samples,
            show_confidence=args.show_confidence
        )
        sys.exit(0 if success else 1)
    else:
        # Interactive mode
        try:
            cli.run_interactive()
        except KeyboardInterrupt:
            console.print("\n\n[yellow]Interrupted by user[/yellow]")
            sys.exit(0)


if __name__ == "__main__":
    main()
