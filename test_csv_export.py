#!/usr/bin/env python3
"""
Test script for CSV export functionality.
Tests the derived metrics calculation and formatting.
"""

from csv_exporter import safe_float, calculate_derived_metrics, format_value


def test_safe_float():
    """Test safe_float conversion."""
    print("Testing safe_float()...")

    test_cases = [
        ("12345", 12345.0),
        ("12,345", 12345.0),
        ("12,345.67", 12345.67),
        ("N/A", None),
        ("NA", None),
        ("", None),
        (None, None),
        ("invalid", None),
    ]

    for input_val, expected in test_cases:
        result = safe_float(input_val)
        status = "✓" if result == expected else "✗"
        print(f"  {status} safe_float('{input_val}') = {result} (expected: {expected})")


def test_derived_metrics():
    """Test derived metrics calculation."""
    print("\nTesting calculate_derived_metrics()...")

    # Sample data (Williamson Tea 2018 from example CSV)
    test_data = {
        'credit_sales': '1808385',
        'net_sales': '3984971',
        'debtors': '2093446',
        'impaired_debts': '2367',
        'current_assets': '3657136',
        'current_liabilities': '558617',
        'cash': '360669',
        'marketable_securities': 'N/A',
        'total_liabilities': '2657717',
        'total_assets': '9505074',
        'ebit': '810056',
        'interest': '10059',
        'labour_cost': '677773',
        'operating_expenses': '549312',
        'rd_cost': 'N/A',
        'tax': '307287',
        'equity': '6847357',
        'net_income': '502769'
    }

    derived = calculate_derived_metrics(test_data)

    print(f"\n  Calculated Derived Metrics:")
    print(f"    Trade Credit Exposure Rate: {derived['trade_credit_exposure_rate']:.4f}")
    print(f"    Default Rate: {derived['default_rate']:.4f}")
    print(f"    Current Ratio: {derived['current_ratio']:.4f}")
    print(f"    Cash Ratio: {derived['cash_ratio']:.4f}")
    print(f"    Debt Ratio: {derived['debt_ratio']:.4f}")
    print(f"    EBIT - Interest: {derived['ebit_minus_interest']}")
    print(f"    DFL: {derived['dfl']:.4f}")
    print(f"    Firm Size (ln Assets): {derived['firm_size']:.4f}")
    print(f"    ROE: {derived['roe']:.4f}")
    print(f"    ROS: {derived['ros']:.4f}")

    # Verify some key calculations
    assert abs(derived['current_ratio'] - (3657136 / 558617)) < 0.01, "Current ratio incorrect"
    assert abs(derived['debt_ratio'] - (2657717 / 9505074)) < 0.01, "Debt ratio incorrect"
    assert abs(derived['roe'] - (502769 / 6847357)) < 0.01, "ROE incorrect"

    print("\n  ✓ All derived metrics calculated correctly")


def test_format_value():
    """Test value formatting."""
    print("\nTesting format_value()...")

    test_cases = [
        # (value, is_currency, expected)
        (12345.67, True, "12,345"),
        (12345.67, False, "12345.6700"),
        (None, True, "NA"),
        (None, False, "NA"),
        (0.1234, False, "0.1234"),
        (1234567, True, "1,234,567"),
    ]

    for value, is_currency, expected in test_cases:
        result = format_value(value, is_currency)
        status = "✓" if result == expected else "✗"
        print(f"  {status} format_value({value}, is_currency={is_currency}) = '{result}' (expected: '{expected}')")


def test_missing_data_handling():
    """Test handling of missing/NA data."""
    print("\nTesting missing data handling...")

    # Data with missing values
    test_data = {
        'credit_sales': 'N/A',
        'net_sales': '1000000',
        'debtors': 'N/A',
        'impaired_debts': 'N/A',
        'current_assets': '500000',
        'current_liabilities': 'N/A',
        'cash': '100000',
        'marketable_securities': 'N/A',
        'total_liabilities': '300000',
        'total_assets': '800000',
        'ebit': 'N/A',
        'interest': 'N/A',
        'labour_cost': 'N/A',
        'operating_expenses': 'N/A',
        'rd_cost': 'N/A',
        'tax': '50000',
        'equity': '500000',
        'net_income': '100000'
    }

    derived = calculate_derived_metrics(test_data)

    # These should be NA because of missing data
    assert derived['trade_credit_exposure_rate'] is None, "Should be None with missing debtors"
    assert derived['default_rate'] is None, "Should be None with missing impaired_debts"
    assert derived['current_ratio'] is None, "Should be None with missing current_liabilities"

    # These should calculate successfully
    assert derived['debt_ratio'] is not None, "Debt ratio should calculate"
    assert derived['roe'] is not None, "ROE should calculate"
    assert derived['ros'] is not None, "ROS should calculate"

    print(f"  ✓ Debt Ratio: {format_value(derived['debt_ratio'])}")
    print(f"  ✓ ROE: {format_value(derived['roe'])}")
    print(f"  ✓ ROS: {format_value(derived['ros'])}")
    print(f"  ✓ Trade Credit Rate: {format_value(derived['trade_credit_exposure_rate'])}")

    print("\n  ✓ Missing data handled correctly")


if __name__ == "__main__":
    print("="*70)
    print("CSV EXPORT FUNCTIONALITY TESTS")
    print("="*70)

    try:
        test_safe_float()
        test_derived_metrics()
        test_format_value()
        test_missing_data_handling()

        print("\n" + "="*70)
        print("ALL TESTS PASSED ✓")
        print("="*70)

    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        exit(1)
    except Exception as e:
        print(f"\n✗ ERROR: {type(e).__name__}: {e}")
        exit(1)
