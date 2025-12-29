#!/usr/bin/env python3
"""
Create test data in the database for testing CSV export.
"""

from database import ExtractionDatabase
from datetime import datetime


def create_test_data():
    """Create sample extraction data for testing."""
    print("Creating test extraction data...")

    db = ExtractionDatabase("./extractions.db")

    # Create test PDF records
    companies_data = [
        {
            "company": "SGL Limited",
            "year": "2023",
            "data": {
                "credit_sales": "5000000",
                "net_sales": "12500000",
                "debtors": "3200000",
                "impaired_debts": "150000",
                "current_assets": "8500000",
                "current_liabilities": "2500000",
                "cash": "1200000",
                "marketable_securities": "500000",
                "total_liabilities": "6500000",
                "total_assets": "18000000",
                "ebit": "2500000",
                "interest": "250000",
                "labour_cost": "1800000",
                "operating_expenses": "3500000",
                "rd_cost": "200000",
                "tax": "550000",
                "equity": "11500000",
                "net_income": "1700000"
            },
            "confidence": {
                "credit_sales": 1.00,
                "net_sales": 1.00,
                "debtors": 0.67,
                "impaired_debts": 0.67,
                "current_assets": 1.00,
                "current_liabilities": 1.00,
                "cash": 1.00,
                "marketable_securities": 0.67,
                "total_liabilities": 1.00,
                "total_assets": 1.00,
                "ebit": 0.67,
                "interest": 1.00,
                "labour_cost": 1.00,
                "operating_expenses": 1.00,
                "rd_cost": 0.33,
                "tax": 1.00,
                "equity": 1.00,
                "net_income": 1.00
            }
        },
        {
            "company": "WILLIAMSON TEA KENYA",
            "year": "2023",
            "data": {
                "credit_sales": "N/A",
                "net_sales": "4200000",
                "debtors": "950000",
                "impaired_debts": "50000",
                "current_assets": "3100000",
                "current_liabilities": "850000",
                "cash": "420000",
                "marketable_securities": "N/A",
                "total_liabilities": "2100000",
                "total_assets": "9200000",
                "ebit": "580000",
                "interest": "45000",
                "labour_cost": "890000",
                "operating_expenses": "1200000",
                "rd_cost": "N/A",
                "tax": "125000",
                "equity": "7100000",
                "net_income": "410000"
            },
            "confidence": {
                "credit_sales": 0.00,
                "net_sales": 1.00,
                "debtors": 1.00,
                "impaired_debts": 0.67,
                "current_assets": 1.00,
                "current_liabilities": 1.00,
                "cash": 1.00,
                "marketable_securities": 0.00,
                "total_liabilities": 1.00,
                "total_assets": 1.00,
                "ebit": 0.67,
                "interest": 1.00,
                "labour_cost": 1.00,
                "operating_expenses": 1.00,
                "rd_cost": 0.00,
                "tax": 1.00,
                "equity": 1.00,
                "net_income": 0.67
            }
        },
        {
            "company": "UNGA GROUP",
            "year": "2023",
            "data": {
                "credit_sales": "N/A",
                "net_sales": "17895670",
                "debtors": "3017093",
                "impaired_debts": "N/A",
                "current_assets": "6676636",
                "current_liabilities": "3413608",
                "cash": "841338",
                "marketable_securities": "N/A",
                "total_liabilities": "4590656",
                "total_assets": "10646066",
                "ebit": "718960",
                "interest": "166748",
                "labour_cost": "1286745",
                "operating_expenses": "1813229",
                "rd_cost": "N/A",
                "tax": "70388",
                "equity": "6055410",
                "net_income": "544814"
            },
            "confidence": {
                "credit_sales": 0.00,
                "net_sales": 1.00,
                "debtors": 1.00,
                "impaired_debts": 0.00,
                "current_assets": 1.00,
                "current_liabilities": 1.00,
                "cash": 1.00,
                "marketable_securities": 0.00,
                "total_liabilities": 1.00,
                "total_assets": 1.00,
                "ebit": 1.00,
                "interest": 1.00,
                "labour_cost": 1.00,
                "operating_expenses": 1.00,
                "rd_cost": 0.00,
                "tax": 1.00,
                "equity": 1.00,
                "net_income": 1.00
            }
        }
    ]

    for company_data in companies_data:
        company_name = company_data["company"]
        year = company_data["year"]
        data = company_data["data"]
        confidence_scores = company_data["confidence"]

        print(f"\nCreating test data for {company_name} ({year})...")

        # Create PDF record
        file_hash = f"test_hash_{company_name.replace(' ', '_')}_{year}"
        cursor = db.conn.cursor()
        cursor.execute("""
            INSERT INTO pdfs (file_path, file_hash, file_size, num_pages)
            VALUES (?, ?, ?, ?)
        """, (f"/test/{company_name}/{year}.pdf", file_hash, 1024000, 50))
        db.conn.commit()
        pdf_id = cursor.lastrowid

        # Create extraction attempt
        attempt_id = db.create_extraction_attempt(
            pdf_id=pdf_id,
            model="deepseek-chat",
            num_samples=3,
            extraction_method="ocr_text_with_consistency",
            company_name=company_name,
            financial_year=year
        )

        # Create confidence metrics in the format expected by save_extraction_results
        confidence_metrics = {}
        for metric_name, value in data.items():
            score = confidence_scores[metric_name]
            if score == 1.00:
                level = "high"
            elif score >= 0.6:
                level = "medium"
            elif score > 0:
                level = "low"
            else:
                level = "unknown"

            confidence_metrics[metric_name] = {
                "final_value": value,
                "confidence_score": score,
                "confidence_level": level,
                "agreement": f"3/3" if score == 1.00 else f"2/3" if score >= 0.6 else f"1/3",
                "vote_distribution": {value: 3} if score == 1.00 else {value: 2, "other": 1}
            }

        # Save extraction results
        db.save_extraction_results(attempt_id, data, confidence_metrics)

        # Mark attempt as successful
        db.complete_extraction_attempt(attempt_id, success=True)

        print(f"  ✓ Created extraction for {company_name} ({year})")

    db.close()

    print(f"\n{'='*70}")
    print("TEST DATA CREATION COMPLETE")
    print(f"{'='*70}")
    print(f"Created {len(companies_data)} test extractions")
    print("Database: ./extractions.db")
    print("\nYou can now test CSV export with:")
    print("  python pdf_extractor.py --export test_results.csv")
    print("  python pdf_extractor.py --export test_results.csv --confidence")


if __name__ == "__main__":
    create_test_data()
