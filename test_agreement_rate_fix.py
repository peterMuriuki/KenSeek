#!/usr/bin/env python3
"""
Test script to verify agreement_rate is correctly stored and retrieved.
"""

import json
from database import ExtractionDatabase


def test_agreement_rate_storage():
    """Test that agreement_rate is correctly stored as decimal."""
    print("="*70)
    print("AGREEMENT RATE STORAGE TEST")
    print("="*70)

    # Create test database
    db = ExtractionDatabase(":memory:")  # Use in-memory database for testing

    # Create a test PDF record directly in database (bypass file check)
    cursor = db.conn.cursor()
    cursor.execute("""
        INSERT INTO pdfs (file_path, file_hash, file_size, num_pages)
        VALUES (?, ?, ?, ?)
    """, ("/test/report.pdf", "test_hash_123", 1024, 10))
    db.conn.commit()
    pdf_id = cursor.lastrowid

    # Create a test extraction attempt
    attempt_id = db.create_extraction_attempt(
        pdf_id=pdf_id,
        model="deepseek-chat",
        num_samples=3,
        extraction_method="ocr_text_with_consistency",
        company_name="Test Company",
        financial_year="2023"
    )

    # Create test consensus data and confidence metrics
    # This mimics the structure from _compute_consensus
    consensus_data = {
        'net_sales': '1000000',
        'credit_sales': '500000',
        'debtors': '200000'
    }

    confidence_metrics = {
        'net_sales': {
            'final_value': '1000000',
            'confidence_score': 1.00,  # All 3 samples agreed
            'confidence_level': 'high',
            'agreement': '3/3',
            'vote_distribution': {'1000000': 3}
        },
        'credit_sales': {
            'final_value': '500000',
            'confidence_score': 0.67,  # 2 out of 3 samples agreed
            'confidence_level': 'medium',
            'agreement': '2/3',
            'vote_distribution': {'500000': 2, '450000': 1}
        },
        'debtors': {
            'final_value': '200000',
            'confidence_score': 0.33,  # Only 1 out of 3 samples agreed
            'confidence_level': 'low',
            'agreement': '1/3',
            'vote_distribution': {'200000': 1, '180000': 1, '210000': 1}
        }
    }

    # Save extraction results
    db.save_extraction_results(attempt_id, consensus_data, confidence_metrics)

    # Mark attempt as successful
    db.complete_extraction_attempt(attempt_id, success=True)

    print("\nTest data saved to database")
    print("\nExpected agreement rates:")
    print("  net_sales: 1.00 (3/3 agreement)")
    print("  credit_sales: 0.67 (2/3 agreement)")
    print("  debtors: 0.33 (1/3 agreement)")

    # Retrieve and verify
    print("\nRetrieving from database...")
    cursor = db.conn.cursor()
    cursor.execute("""
        SELECT metric_name, consensus_value, confidence, agreement_rate
        FROM extraction_results
        WHERE attempt_id = ?
        ORDER BY metric_name
    """, (attempt_id,))

    results = cursor.fetchall()

    print("\nActual values from database:")
    all_correct = True
    for row in results:
        metric_name = row['metric_name']
        agreement_rate = row['agreement_rate']
        confidence_level = row['confidence']

        expected_rate = confidence_metrics[metric_name]['confidence_score']
        expected_level = confidence_metrics[metric_name]['confidence_level']

        status = "✓" if agreement_rate == expected_rate else "✗"
        level_status = "✓" if confidence_level == expected_level else "✗"

        print(f"  {status} {metric_name}: {agreement_rate} (expected: {expected_rate})")
        print(f"    {level_status} confidence_level: {confidence_level} (expected: {expected_level})")

        if agreement_rate != expected_rate or confidence_level != expected_level:
            all_correct = False

    # Test get_all_successful_extractions
    print("\nTesting get_all_successful_extractions()...")
    extractions = db.get_all_successful_extractions()

    if len(extractions) == 1:
        print("  ✓ Retrieved 1 extraction")
        extraction = extractions[0]

        if 'confidence_metrics' in extraction:
            print("  ✓ Confidence metrics present")
            conf = extraction['confidence_metrics']

            for metric_name in ['net_sales', 'credit_sales', 'debtors']:
                if metric_name in conf:
                    retrieved_rate = conf[metric_name]['agreement_rate']
                    expected_rate = confidence_metrics[metric_name]['confidence_score']

                    status = "✓" if retrieved_rate == expected_rate else "✗"
                    print(f"    {status} {metric_name} agreement_rate: {retrieved_rate} (expected: {expected_rate})")

                    if retrieved_rate != expected_rate:
                        all_correct = False
                else:
                    print(f"    ✗ {metric_name} not found in confidence metrics")
                    all_correct = False
        else:
            print("  ✗ Confidence metrics missing")
            all_correct = False
    else:
        print(f"  ✗ Expected 1 extraction, got {len(extractions)}")
        all_correct = False

    db.close()

    print("\n" + "="*70)
    if all_correct:
        print("ALL TESTS PASSED ✓")
        print("Agreement rates are correctly stored as decimals!")
    else:
        print("TESTS FAILED ✗")
        print("Agreement rates are NOT being stored correctly")
    print("="*70)

    return all_correct


if __name__ == "__main__":
    try:
        success = test_agreement_rate_storage()
        exit(0 if success else 1)
    except Exception as e:
        print(f"\n✗ ERROR: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
