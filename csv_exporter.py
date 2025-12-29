#!/usr/bin/env python3
"""
CSV Exporter for Financial Data
Exports extraction results to CSV format with derived metrics and optional confidence rates.
"""

import csv
import math
from typing import Dict, List, Optional
from database import ExtractionDatabase


def safe_float(value: str) -> Optional[float]:
    """
    Safely convert string to float, handling N/A and empty values.

    Args:
        value: String value to convert

    Returns:
        Float value or None if conversion fails
    """
    if not value or value == "N/A" or value == "NA":
        return None

    try:
        # Remove commas and whitespace
        cleaned = str(value).replace(",", "").strip()
        return float(cleaned)
    except (ValueError, AttributeError):
        return None


def calculate_derived_metrics(data: Dict[str, str]) -> Dict[str, Optional[float]]:
    """
    Calculate derived financial metrics from extracted data.

    Args:
        data: Dictionary of extracted financial data

    Returns:
        Dictionary of derived metrics
    """
    # Convert all values to floats
    credit_sales = safe_float(data.get('credit_sales'))
    net_sales = safe_float(data.get('net_sales'))
    debtors = safe_float(data.get('debtors'))
    impaired_debts = safe_float(data.get('impaired_debts'))
    current_assets = safe_float(data.get('current_assets'))
    current_liabilities = safe_float(data.get('current_liabilities'))
    cash = safe_float(data.get('cash'))
    marketable_securities = safe_float(data.get('marketable_securities'))
    total_liabilities = safe_float(data.get('total_liabilities'))
    total_assets = safe_float(data.get('total_assets'))
    ebit = safe_float(data.get('ebit'))
    interest = safe_float(data.get('interest'))
    labour_cost = safe_float(data.get('labour_cost'))
    operating_expenses = safe_float(data.get('operating_expenses'))
    rd_cost = safe_float(data.get('rd_cost'))
    tax = safe_float(data.get('tax'))
    equity = safe_float(data.get('equity'))
    net_income = safe_float(data.get('net_income'))

    derived = {}

    # Trade credit exposure rate (Debtors / Net Sales)
    if debtors is not None and net_sales is not None and net_sales != 0:
        derived['trade_credit_exposure_rate'] = debtors / net_sales
    else:
        derived['trade_credit_exposure_rate'] = None

    # Default Rate (Impaired Debts / Debtors)
    if impaired_debts is not None and debtors is not None and debtors != 0:
        derived['default_rate'] = impaired_debts / debtors
    else:
        derived['default_rate'] = None

    # Current Ratio (Current Assets / Current Liabilities)
    if current_assets is not None and current_liabilities is not None and current_liabilities != 0:
        derived['current_ratio'] = current_assets / current_liabilities
    else:
        derived['current_ratio'] = None

    # C+MS (Cash + Marketable Securities)
    c_ms = 0
    if cash is not None:
        c_ms += cash
    if marketable_securities is not None:
        c_ms += marketable_securities
    derived['c_ms'] = c_ms if (cash is not None or marketable_securities is not None) else None

    # Cash Ratio ((Cash + Marketable Securities) / Current Liabilities)
    if derived['c_ms'] is not None and current_liabilities is not None and current_liabilities != 0:
        derived['cash_ratio'] = derived['c_ms'] / current_liabilities
    else:
        derived['cash_ratio'] = None

    # Debt Ratio (Total Liabilities / Total Assets)
    if total_liabilities is not None and total_assets is not None and total_assets != 0:
        derived['debt_ratio'] = total_liabilities / total_assets
    else:
        derived['debt_ratio'] = None

    # EBIT - Interest
    if ebit is not None and interest is not None:
        derived['ebit_minus_interest'] = ebit - interest
    else:
        derived['ebit_minus_interest'] = None

    # DFL (Degree of Financial Leverage) = EBIT / (EBIT - Interest)
    if ebit is not None and derived['ebit_minus_interest'] is not None and derived['ebit_minus_interest'] != 0:
        derived['dfl'] = ebit / derived['ebit_minus_interest']
    else:
        derived['dfl'] = None

    # LCOR (Labour Cost to Operating Revenue Ratio) = Labour Cost / Operating Expenses
    if labour_cost is not None and operating_expenses is not None and operating_expenses != 0:
        derived['lcor'] = labour_cost / operating_expenses
    else:
        derived['lcor'] = None

    # R&D Cost Ratio (R&D Cost / Net Sales)
    if rd_cost is not None and net_sales is not None and net_sales != 0:
        derived['rd_cost_ratio'] = rd_cost / net_sales
    else:
        derived['rd_cost_ratio'] = None

    # Firm Size (Ln of Total Assets)
    if total_assets is not None and total_assets > 0:
        derived['firm_size'] = math.log(total_assets)
    else:
        derived['firm_size'] = None

    # ROE (Return on Equity) = Net Income / Equity
    if net_income is not None and equity is not None and equity != 0:
        derived['roe'] = net_income / equity
    else:
        derived['roe'] = None

    # ROS (Return on Sales) = Net Income / Net Sales
    if net_income is not None and net_sales is not None and net_sales != 0:
        derived['ros'] = net_income / net_sales
    else:
        derived['ros'] = None

    return derived


def format_value(value: Optional[float], is_currency: bool = False) -> str:
    """
    Format a numeric value for CSV output.

    Args:
        value: Numeric value to format
        is_currency: Whether to format as currency (with commas)

    Returns:
        Formatted string
    """
    if value is None:
        return "NA"

    if is_currency:
        # Format with thousand separators, no decimals for currency
        return f"{int(value):,}"
    else:
        # Format ratios/percentages with decimals
        return f"{value:.4f}"


def export_to_csv(
    db_path: str,
    output_file: str,
    include_confidence: bool = False,
    company_filter: Optional[str] = None,
    year_filter: Optional[str] = None,
    sector: str = "",
    sector_short: str = "",
    code: str = ""
) -> int:
    """
    Export extraction results to CSV file.

    Args:
        db_path: Path to SQLite database
        output_file: Output CSV file path
        include_confidence: Include confidence rate columns
        company_filter: Filter by company name
        year_filter: Filter by financial year
        sector: Sector name for all records
        sector_short: Short sector name
        code: Code field for records

    Returns:
        Number of records exported
    """
    # Connect to database
    db = ExtractionDatabase(db_path)

    # Get all successful extractions
    extractions = db.get_all_successful_extractions(
        company_name=company_filter,
        financial_year=year_filter
    )

    if not extractions:
        print(f"No successful extractions found")
        return 0

    # Define CSV columns (base columns)
    base_columns = [
        'code', 'Sector', 'SectorS', 'Company', 'Cos', 'Year',
        'Credit_Sales (kes)', 'Net_Sales (Kes)', 'Trade_credit_exposure_rate',
        'Debtors (Kes)', 'Impaired_Debts (Kes)', 'Default_Rate',
        'Current_Assets (Kes)', 'Current_Liabilities (Kes)', 'Current_ratio',
        'Cash (Kes)', 'Marketable_Securities (Kes)', 'C+MS (Kes)', 'Cash_Ratio',
        'Total_liabilities (Kes)', 'Total_assets (Kes)', 'Debt_ratio',
        'EBIT (Kes)', 'Interest (Kes)', 'EBIT - Interest (Kes)', 'DFL',
        'Labour cost (Kes)', 'Operating expenses (Kes)', 'LCOR (Kes)',
        'R&D_cost  (Kes)', 'R&D_cost_ratio',
        'Firm_Size (Ln Assets)', 'Tax (Kes)', 'Equity (Kes)', 'Netincome (Kes)',
        'ROE', 'ROS'
    ]

    # Add confidence columns if requested
    if include_confidence:
        confidence_columns = [
            'Credit_Sales_Confidence', 'Net_Sales_Confidence', 'Debtors_Confidence',
            'Impaired_Debts_Confidence', 'Current_Assets_Confidence', 'Current_Liabilities_Confidence',
            'Cash_Confidence', 'Marketable_Securities_Confidence', 'Total_liabilities_Confidence',
            'Total_assets_Confidence', 'EBIT_Confidence', 'Interest_Confidence',
            'Labour_cost_Confidence', 'Operating_expenses_Confidence', 'R&D_cost_Confidence',
            'Tax_Confidence', 'Equity_Confidence', 'Net_income_Confidence'
        ]
        columns = base_columns + confidence_columns
    else:
        columns = base_columns

    # Write CSV
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()

        for extraction in extractions:
            data = extraction.get('extracted_data', {})
            confidence = extraction.get('confidence_metrics', {})

            # Calculate derived metrics
            derived = calculate_derived_metrics(data)

            # Build row
            row = {
                'code': code,
                'Sector': sector,
                'SectorS': sector_short,
                'Company': extraction.get('company_name', ''),
                'Cos': '',  # Empty as in example
                'Year': extraction.get('financial_year', ''),

                # Base financial data (raw values)
                'Credit_Sales (kes)': format_value(safe_float(data.get('credit_sales')), is_currency=True),
                'Net_Sales (Kes)': format_value(safe_float(data.get('net_sales')), is_currency=True),
                'Debtors (Kes)': format_value(safe_float(data.get('debtors')), is_currency=True),
                'Impaired_Debts (Kes)': format_value(safe_float(data.get('impaired_debts')), is_currency=True),
                'Current_Assets (Kes)': format_value(safe_float(data.get('current_assets')), is_currency=True),
                'Current_Liabilities (Kes)': format_value(safe_float(data.get('current_liabilities')), is_currency=True),
                'Cash (Kes)': format_value(safe_float(data.get('cash')), is_currency=True),
                'Marketable_Securities (Kes)': format_value(safe_float(data.get('marketable_securities')), is_currency=True),
                'Total_liabilities (Kes)': format_value(safe_float(data.get('total_liabilities')), is_currency=True),
                'Total_assets (Kes)': format_value(safe_float(data.get('total_assets')), is_currency=True),
                'EBIT (Kes)': format_value(safe_float(data.get('ebit')), is_currency=True),
                'Interest (Kes)': format_value(safe_float(data.get('interest')), is_currency=True),
                'Labour cost (Kes)': format_value(safe_float(data.get('labour_cost')), is_currency=True),
                'Operating expenses (Kes)': format_value(safe_float(data.get('operating_expenses')), is_currency=True),
                'R&D_cost  (Kes)': format_value(safe_float(data.get('rd_cost')), is_currency=True),
                'Tax (Kes)': format_value(safe_float(data.get('tax')), is_currency=True),
                'Equity (Kes)': format_value(safe_float(data.get('equity')), is_currency=True),
                'Netincome (Kes)': format_value(safe_float(data.get('net_income')), is_currency=True),

                # Derived metrics (ratios)
                'Trade_credit_exposure_rate': format_value(derived['trade_credit_exposure_rate']),
                'Default_Rate': format_value(derived['default_rate']),
                'Current_ratio': format_value(derived['current_ratio']),
                'C+MS (Kes)': format_value(derived['c_ms'], is_currency=True),
                'Cash_Ratio': format_value(derived['cash_ratio']),
                'Debt_ratio': format_value(derived['debt_ratio']),
                'EBIT - Interest (Kes)': format_value(derived['ebit_minus_interest'], is_currency=True),
                'DFL': format_value(derived['dfl']),
                'LCOR (Kes)': format_value(derived['lcor']),
                'R&D_cost_ratio': format_value(derived['rd_cost_ratio']),
                'Firm_Size (Ln Assets)': format_value(derived['firm_size']),
                'ROE': format_value(derived['roe']),
                'ROS': format_value(derived['ros'])
            }

            # Add confidence columns if requested
            if include_confidence:
                metric_mapping = {
                    'credit_sales': 'Credit_Sales_Confidence',
                    'net_sales': 'Net_Sales_Confidence',
                    'debtors': 'Debtors_Confidence',
                    'impaired_debts': 'Impaired_Debts_Confidence',
                    'current_assets': 'Current_Assets_Confidence',
                    'current_liabilities': 'Current_Liabilities_Confidence',
                    'cash': 'Cash_Confidence',
                    'marketable_securities': 'Marketable_Securities_Confidence',
                    'total_liabilities': 'Total_liabilities_Confidence',
                    'total_assets': 'Total_assets_Confidence',
                    'ebit': 'EBIT_Confidence',
                    'interest': 'Interest_Confidence',
                    'labour_cost': 'Labour_cost_Confidence',
                    'operating_expenses': 'Operating_expenses_Confidence',
                    'rd_cost': 'R&D_cost_Confidence',
                    'tax': 'Tax_Confidence',
                    'equity': 'Equity_Confidence',
                    'net_income': 'Net_income_Confidence'
                }

                for metric_key, column_name in metric_mapping.items():
                    conf_data = confidence.get(metric_key, {})
                    agreement_rate = conf_data.get('agreement_rate')

                    if agreement_rate is not None:
                        # Handle both decimal (0.67) and fraction string ("2/3") formats
                        if isinstance(agreement_rate, str):
                            # Parse fraction format like "2/3" or "3/3"
                            if '/' in agreement_rate:
                                try:
                                    parts = agreement_rate.split('/')
                                    numerator = float(parts[0])
                                    denominator = float(parts[1])
                                    agreement_rate = numerator / denominator if denominator > 0 else 0.0
                                except (ValueError, IndexError):
                                    agreement_rate = 0.0
                            else:
                                # Try to convert string to float
                                try:
                                    agreement_rate = float(agreement_rate)
                                except ValueError:
                                    agreement_rate = 0.0

                        row[column_name] = f"{agreement_rate:.2f}"
                    else:
                        row[column_name] = "NA"

            writer.writerow(row)

    db.close()

    print(f"[EXPORT] ✓ Exported {len(extractions)} records to {output_file}")
    if include_confidence:
        print(f"[EXPORT] Included confidence rate columns")

    return len(extractions)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Export extraction results to CSV")
    parser.add_argument("--db", default="./extractions.db", help="Database path")
    parser.add_argument("-o", "--output", required=True, help="Output CSV file")
    parser.add_argument("--confidence", action="store_true", help="Include confidence rate columns")
    parser.add_argument("--company", help="Filter by company name")
    parser.add_argument("--year", help="Filter by financial year")
    parser.add_argument("--sector", default="", help="Sector name for all records")
    parser.add_argument("--sector-short", default="", help="Short sector name")
    parser.add_argument("--code", default="", help="Code field for records")

    args = parser.parse_args()

    count = export_to_csv(
        db_path=args.db,
        output_file=args.output,
        include_confidence=args.confidence,
        company_filter=args.company,
        year_filter=args.year,
        sector=args.sector,
        sector_short=args.sector_short,
        code=args.code
    )

    print(f"\n✓ Export complete: {count} records written to {args.output}")
