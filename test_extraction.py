#!/usr/bin/env python3
"""
Validation Tests for FNFTA Financial Data Extraction

This module contains tests and data quality checks for the extracted financial data.
"""

import json
import os
import sqlite3
import unittest
from datetime import datetime
from typing import List, Dict, Any
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import extraction module
try:
    from extract_financials import (
        parse_currency,
        FinancialStatement,
        RemunerationRecord,
        FinancialStatementExtractor,
        RemunerationExtractor,
        FNFTADatabase,
        load_metadata
    )
except ImportError:
    logger.error("Could not import extract_financials module")
    raise


class TestCurrencyParsing(unittest.TestCase):
    """Tests for currency parsing function"""

    def test_basic_number(self):
        self.assertEqual(parse_currency("1234.56"), 1234.56)

    def test_with_dollar_sign(self):
        self.assertEqual(parse_currency("$1,234.56"), 1234.56)

    def test_with_commas(self):
        self.assertEqual(parse_currency("1,234,567"), 1234567.0)

    def test_negative_parentheses(self):
        self.assertEqual(parse_currency("(1,234)"), -1234.0)

    def test_negative_with_dollar(self):
        self.assertEqual(parse_currency("$(5,000.00)"), -5000.0)

    def test_nil_values(self):
        self.assertIsNone(parse_currency("-"))
        self.assertIsNone(parse_currency("—"))
        self.assertIsNone(parse_currency("nil"))
        self.assertIsNone(parse_currency(""))
        self.assertIsNone(parse_currency(None))

    def test_large_numbers(self):
        self.assertEqual(parse_currency("$25,902,376"), 25902376.0)

    def test_spaces(self):
        self.assertEqual(parse_currency("$ 1,234.56"), 1234.56)


class TestFinancialStatementDataclass(unittest.TestCase):
    """Tests for FinancialStatement dataclass"""

    def test_creation(self):
        fs = FinancialStatement(
            band_number=1,
            band_name="Test Band",
            fiscal_year="2023-2024"
        )
        self.assertEqual(fs.band_number, 1)
        self.assertEqual(fs.band_name, "Test Band")
        self.assertIsNone(fs.total_revenue)

    def test_all_fields_nullable(self):
        fs = FinancialStatement(
            band_number=1,
            band_name="Test",
            fiscal_year="2023-2024"
        )
        # All financial fields should be None by default
        self.assertIsNone(fs.total_revenue)
        self.assertIsNone(fs.total_expenditures)
        self.assertIsNone(fs.net_revenue)
        self.assertIsNone(fs.total_liabilities)


class TestRemunerationDataclass(unittest.TestCase):
    """Tests for RemunerationRecord dataclass"""

    def test_creation(self):
        record = RemunerationRecord(
            band_number=1,
            band_name="Test Band",
            fiscal_year="2023-2024",
            position="Chief",
            name="John Doe",
            salary=100000.0
        )
        self.assertEqual(record.position, "Chief")
        self.assertEqual(record.salary, 100000.0)


class TestDatabaseOperations(unittest.TestCase):
    """Tests for database operations"""

    @classmethod
    def setUpClass(cls):
        """Set up test database"""
        cls.db_path = '/tmp/test_fnfta.db'
        cls.db = FNFTADatabase(cls.db_path)
        cls.db.create_tables()

    @classmethod
    def tearDownClass(cls):
        """Clean up test database"""
        if os.path.exists(cls.db_path):
            os.remove(cls.db_path)

    def test_insert_financial_statement(self):
        fs = FinancialStatement(
            band_number=999,
            band_name="Test Band",
            fiscal_year="2023-2024",
            total_revenue=1000000.0,
            total_expenditures=900000.0,
            net_revenue=100000.0
        )
        self.db.insert_financial_statement(fs)

        # Verify insertion
        self.db.connect()
        cursor = self.db.conn.cursor()
        cursor.execute('SELECT * FROM financial_statements WHERE band_number = 999')
        row = cursor.fetchone()
        self.db.close()

        self.assertIsNotNone(row)
        self.assertEqual(row['total_revenue'], 1000000.0)

    def test_insert_remuneration(self):
        record = RemunerationRecord(
            band_number=999,
            band_name="Test Band",
            fiscal_year="2023-2024",
            position="Chief",
            name="Test Chief",
            salary=100000.0,
            total_remuneration=150000.0
        )
        self.db.insert_remuneration(record)

        # Verify insertion
        self.db.connect()
        cursor = self.db.conn.cursor()
        cursor.execute('SELECT * FROM remuneration WHERE band_number = 999')
        row = cursor.fetchone()
        self.db.close()

        self.assertIsNotNone(row)
        self.assertEqual(row['salary'], 100000.0)

    def test_upsert_behavior(self):
        """Test that duplicate records are updated not duplicated"""
        fs = FinancialStatement(
            band_number=998,
            band_name="Upsert Test",
            fiscal_year="2023-2024",
            total_revenue=500000.0
        )
        self.db.insert_financial_statement(fs)

        # Update with new value
        fs.total_revenue = 600000.0
        self.db.insert_financial_statement(fs)

        # Should only have one record
        self.db.connect()
        cursor = self.db.conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM financial_statements WHERE band_number = 998')
        count = cursor.fetchone()[0]
        cursor.execute('SELECT total_revenue FROM financial_statements WHERE band_number = 998')
        revenue = cursor.fetchone()[0]
        self.db.close()

        self.assertEqual(count, 1)
        self.assertEqual(revenue, 600000.0)


class DataQualityChecks:
    """Data quality validation checks for the extracted database"""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.issues = []

    def connect(self):
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row

    def close(self):
        if self.conn:
            self.conn.close()

    def run_all_checks(self) -> Dict[str, Any]:
        """Run all data quality checks"""
        results = {
            'timestamp': datetime.now().isoformat(),
            'checks': {},
            'summary': {'passed': 0, 'failed': 0, 'warnings': 0}
        }

        checks = [
            self.check_completeness,
            self.check_revenue_expenditure_balance,
            self.check_remuneration_totals,
            self.check_confidence_scores,
            self.check_fiscal_year_coverage,
            self.check_duplicate_records,
            self.check_outliers,
            self.check_null_rates,
        ]

        for check in checks:
            try:
                check_name = check.__name__
                result = check()
                results['checks'][check_name] = result

                if result.get('status') == 'passed':
                    results['summary']['passed'] += 1
                elif result.get('status') == 'failed':
                    results['summary']['failed'] += 1
                else:
                    results['summary']['warnings'] += 1
            except Exception as e:
                results['checks'][check.__name__] = {'status': 'error', 'error': str(e)}
                results['summary']['failed'] += 1

        return results

    def check_completeness(self) -> Dict:
        """Check that we have data for expected bands and years"""
        self.connect()
        cursor = self.conn.cursor()

        # Check financial statements
        cursor.execute('''
            SELECT fiscal_year, COUNT(DISTINCT band_number) as band_count
            FROM financial_statements
            GROUP BY fiscal_year
            ORDER BY fiscal_year
        ''')
        fs_by_year = {row['fiscal_year']: row['band_count'] for row in cursor.fetchall()}

        # Check remuneration
        cursor.execute('''
            SELECT fiscal_year, COUNT(DISTINCT band_number) as band_count
            FROM remuneration
            GROUP BY fiscal_year
            ORDER BY fiscal_year
        ''')
        rem_by_year = {row['fiscal_year']: row['band_count'] for row in cursor.fetchall()}

        self.close()

        # Determine status
        expected_years = ['2019-2020', '2020-2021', '2021-2022', '2022-2023', '2023-2024']
        missing_years = [y for y in expected_years if y not in fs_by_year]

        return {
            'status': 'passed' if not missing_years else 'warning',
            'financial_statements_by_year': fs_by_year,
            'remuneration_by_year': rem_by_year,
            'missing_years': missing_years
        }

    def check_revenue_expenditure_balance(self) -> Dict:
        """Check that net_revenue = total_revenue - total_expenditures"""
        self.connect()
        cursor = self.conn.cursor()

        cursor.execute('''
            SELECT
                band_number,
                band_name,
                fiscal_year,
                total_revenue,
                total_expenditures,
                net_revenue,
                ABS(total_revenue - total_expenditures - net_revenue) as difference
            FROM financial_statements
            WHERE total_revenue IS NOT NULL
              AND total_expenditures IS NOT NULL
              AND net_revenue IS NOT NULL
              AND ABS(total_revenue - total_expenditures - net_revenue) > 1000
        ''')
        mismatches = [dict(row) for row in cursor.fetchall()]

        cursor.execute('''
            SELECT COUNT(*) FROM financial_statements
            WHERE total_revenue IS NOT NULL
              AND total_expenditures IS NOT NULL
              AND net_revenue IS NOT NULL
        ''')
        total_checked = cursor.fetchone()[0]

        self.close()

        return {
            'status': 'passed' if len(mismatches) == 0 else 'warning',
            'total_checked': total_checked,
            'mismatches_count': len(mismatches),
            'mismatches': mismatches[:10]  # First 10 examples
        }

    def check_remuneration_totals(self) -> Dict:
        """Check that remuneration totals match component sums"""
        self.connect()
        cursor = self.conn.cursor()

        cursor.execute('''
            SELECT
                band_number,
                band_name,
                fiscal_year,
                name,
                position,
                salary,
                honorarium,
                travel,
                contract,
                other,
                total_remuneration,
                COALESCE(salary, 0) + COALESCE(honorarium, 0) +
                COALESCE(travel, 0) + COALESCE(contract, 0) +
                COALESCE(other, 0) as calculated_total,
                ABS(total_remuneration - (
                    COALESCE(salary, 0) + COALESCE(honorarium, 0) +
                    COALESCE(travel, 0) + COALESCE(contract, 0) +
                    COALESCE(other, 0)
                )) as difference
            FROM remuneration
            WHERE total_remuneration IS NOT NULL
              AND ABS(total_remuneration - (
                    COALESCE(salary, 0) + COALESCE(honorarium, 0) +
                    COALESCE(travel, 0) + COALESCE(contract, 0) +
                    COALESCE(other, 0)
                )) > 100
        ''')
        mismatches = [dict(row) for row in cursor.fetchall()]

        cursor.execute('''
            SELECT COUNT(*) FROM remuneration WHERE total_remuneration IS NOT NULL
        ''')
        total_checked = cursor.fetchone()[0]

        self.close()

        return {
            'status': 'passed' if len(mismatches) == 0 else 'warning',
            'total_checked': total_checked,
            'mismatches_count': len(mismatches),
            'mismatches': mismatches[:10]
        }

    def check_confidence_scores(self) -> Dict:
        """Check extraction confidence scores"""
        self.connect()
        cursor = self.conn.cursor()

        cursor.execute('''
            SELECT
                AVG(extraction_confidence) as avg_confidence,
                MIN(extraction_confidence) as min_confidence,
                MAX(extraction_confidence) as max_confidence,
                COUNT(*) as total,
                SUM(CASE WHEN extraction_confidence >= 0.8 THEN 1 ELSE 0 END) as high_confidence,
                SUM(CASE WHEN extraction_confidence < 0.5 THEN 1 ELSE 0 END) as low_confidence
            FROM financial_statements
        ''')
        fs_stats = dict(cursor.fetchone())

        cursor.execute('''
            SELECT
                AVG(extraction_confidence) as avg_confidence,
                MIN(extraction_confidence) as min_confidence,
                MAX(extraction_confidence) as max_confidence,
                COUNT(*) as total,
                SUM(CASE WHEN extraction_confidence >= 0.8 THEN 1 ELSE 0 END) as high_confidence,
                SUM(CASE WHEN extraction_confidence < 0.5 THEN 1 ELSE 0 END) as low_confidence
            FROM remuneration
        ''')
        rem_stats = dict(cursor.fetchone())

        self.close()

        # Status based on average confidence
        avg_fs = fs_stats.get('avg_confidence') or 0
        avg_rem = rem_stats.get('avg_confidence') or 0
        overall_status = 'passed' if avg_fs > 0.6 and avg_rem > 0.6 else 'warning'

        return {
            'status': overall_status,
            'financial_statements': fs_stats,
            'remuneration': rem_stats
        }

    def check_fiscal_year_coverage(self) -> Dict:
        """Check coverage across fiscal years"""
        self.connect()
        cursor = self.conn.cursor()

        cursor.execute('''
            SELECT
                b.band_number,
                b.band_name,
                GROUP_CONCAT(DISTINCT fs.fiscal_year) as years_covered,
                COUNT(DISTINCT fs.fiscal_year) as year_count
            FROM (SELECT DISTINCT band_number, band_name FROM financial_statements) b
            LEFT JOIN financial_statements fs ON b.band_number = fs.band_number
            GROUP BY b.band_number
            HAVING year_count < 5
        ''')
        incomplete_coverage = [dict(row) for row in cursor.fetchall()]

        self.close()

        return {
            'status': 'passed' if len(incomplete_coverage) < 50 else 'warning',
            'bands_with_incomplete_coverage': len(incomplete_coverage),
            'examples': incomplete_coverage[:10]
        }

    def check_duplicate_records(self) -> Dict:
        """Check for unexpected duplicate records"""
        self.connect()
        cursor = self.conn.cursor()

        # For financial statements, band+year should be unique
        cursor.execute('''
            SELECT band_number, fiscal_year, COUNT(*) as count
            FROM financial_statements
            GROUP BY band_number, fiscal_year
            HAVING count > 1
        ''')
        fs_duplicates = [dict(row) for row in cursor.fetchall()]

        self.close()

        return {
            'status': 'passed' if len(fs_duplicates) == 0 else 'failed',
            'financial_statement_duplicates': fs_duplicates
        }

    def check_outliers(self) -> Dict:
        """Check for statistical outliers that might indicate extraction errors"""
        self.connect()
        cursor = self.conn.cursor()

        # Check for unusually high/low revenues
        cursor.execute('''
            SELECT
                band_number,
                band_name,
                fiscal_year,
                total_revenue,
                total_expenditures
            FROM financial_statements
            WHERE total_revenue > 1000000000  -- Over 1 billion
               OR total_revenue < 0
               OR total_expenditures > 1000000000
               OR total_expenditures < 0
        ''')
        revenue_outliers = [dict(row) for row in cursor.fetchall()]

        # Check for unusually high remuneration
        cursor.execute('''
            SELECT
                band_number,
                band_name,
                fiscal_year,
                name,
                position,
                total_remuneration
            FROM remuneration
            WHERE total_remuneration > 1000000  -- Over 1 million
               OR total_remuneration < 0
        ''')
        remuneration_outliers = [dict(row) for row in cursor.fetchall()]

        self.close()

        return {
            'status': 'passed' if len(revenue_outliers) == 0 and len(remuneration_outliers) == 0 else 'warning',
            'revenue_outliers': revenue_outliers,
            'remuneration_outliers': remuneration_outliers
        }

    def check_null_rates(self) -> Dict:
        """Check null rates for key fields"""
        self.connect()
        cursor = self.conn.cursor()

        key_fields = ['total_revenue', 'total_expenditures', 'net_revenue',
                      'total_liabilities', 'net_assets', 'cash_and_equivalents']

        null_rates = {}
        cursor.execute('SELECT COUNT(*) FROM financial_statements')
        total = cursor.fetchone()[0]

        for field in key_fields:
            cursor.execute(f'SELECT COUNT(*) FROM financial_statements WHERE {field} IS NULL')
            null_count = cursor.fetchone()[0]
            null_rates[field] = {
                'null_count': null_count,
                'null_rate': round(null_count / total * 100, 2) if total > 0 else 0
            }

        self.close()

        # Determine status - warn if key fields have high null rates
        high_null_fields = [f for f, stats in null_rates.items()
                          if stats['null_rate'] > 50 and f in ['total_revenue', 'total_expenditures']]

        return {
            'status': 'passed' if not high_null_fields else 'warning',
            'total_records': total,
            'null_rates': null_rates,
            'high_null_fields': high_null_fields
        }


def run_validation(db_path: str) -> Dict:
    """Run all validation checks and return results"""
    checker = DataQualityChecks(db_path)
    return checker.run_all_checks()


def print_validation_report(results: Dict):
    """Print a formatted validation report"""
    print("\n" + "=" * 60)
    print("FNFTA DATA QUALITY VALIDATION REPORT")
    print("=" * 60)
    print(f"Timestamp: {results['timestamp']}")
    print(f"\nSummary: {results['summary']['passed']} passed, "
          f"{results['summary']['warnings']} warnings, "
          f"{results['summary']['failed']} failed")
    print("-" * 60)

    for check_name, check_result in results['checks'].items():
        status = check_result.get('status', 'unknown')
        status_symbol = {'passed': '[OK]', 'warning': '[WARN]', 'failed': '[FAIL]', 'error': '[ERR]'}.get(status, '[?]')

        print(f"\n{status_symbol} {check_name}")

        # Print relevant details
        for key, value in check_result.items():
            if key == 'status':
                continue
            if isinstance(value, list) and len(value) > 0:
                print(f"  {key}: {len(value)} items")
                for item in value[:3]:  # Show first 3
                    print(f"    - {item}")
                if len(value) > 3:
                    print(f"    ... and {len(value) - 3} more")
            elif isinstance(value, dict):
                print(f"  {key}:")
                for k, v in value.items():
                    print(f"    {k}: {v}")
            else:
                print(f"  {key}: {value}")

    print("\n" + "=" * 60)


class IntegrationTests(unittest.TestCase):
    """Integration tests that run against actual PDF files"""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures"""
        cls.data_dir = 'fnfta_data'
        cls.test_pdf_fs = 'fnfta_data/documents/1_Abegweit/2023-2024/Audited_consolidated_financial_statements.pdf'
        cls.test_pdf_rem = 'fnfta_data/documents/1_Abegweit/2023-2024/Schedule_of_Remuneration_and_Expenses.pdf'

    def test_financial_statement_extraction(self):
        """Test extraction from a real financial statement PDF"""
        if not os.path.exists(self.test_pdf_fs):
            self.skipTest("Test PDF not found")

        extractor = FinancialStatementExtractor(
            pdf_path=self.test_pdf_fs,
            band_number=1,
            band_name="Abegweit",
            fiscal_year="2023-2024"
        )
        fs = extractor.extract()

        # Basic checks
        self.assertEqual(fs.band_number, 1)
        self.assertEqual(fs.band_name, "Abegweit")

        # Should extract some financial data
        self.assertTrue(
            fs.total_revenue is not None or
            fs.total_expenditures is not None or
            fs.net_revenue is not None,
            "Should extract at least some financial data"
        )

        # Confidence should be calculated
        self.assertGreaterEqual(fs.extraction_confidence, 0)
        self.assertLessEqual(fs.extraction_confidence, 1)

    def test_remuneration_extraction(self):
        """Test extraction from a real remuneration PDF"""
        if not os.path.exists(self.test_pdf_rem):
            self.skipTest("Test PDF not found")

        extractor = RemunerationExtractor(
            pdf_path=self.test_pdf_rem,
            band_number=1,
            band_name="Abegweit",
            fiscal_year="2023-2024"
        )
        records = extractor.extract()

        # Should extract at least the Chief
        self.assertGreater(len(records), 0, "Should extract at least one remuneration record")

        # Check Chief record
        chief_records = [r for r in records if 'chief' in r.position.lower()]
        self.assertGreater(len(chief_records), 0, "Should extract Chief record")

        # Verify data looks reasonable
        for record in records:
            self.assertIsNotNone(record.name)
            self.assertIsNotNone(record.position)
            if record.total_remuneration:
                self.assertGreater(record.total_remuneration, 0)


if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == '--validate':
        # Run validation checks
        db_path = sys.argv[2] if len(sys.argv) > 2 else 'fnfta_financials.db'
        if os.path.exists(db_path):
            results = run_validation(db_path)
            print_validation_report(results)

            # Save results to JSON
            with open('validation_results.json', 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to validation_results.json")
        else:
            print(f"Database not found: {db_path}")
            sys.exit(1)
    else:
        # Run unit tests
        unittest.main()
