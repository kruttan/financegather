#!/usr/bin/env python3
"""
Validation tests for LlamaExtract-based FNFTA financial data extraction.
"""

import sqlite3
import unittest
from pathlib import Path


class TestDatabaseSchema(unittest.TestCase):
    """Test database schema and structure"""

    @classmethod
    def setUpClass(cls):
        cls.db_path = 'fnfta_llama.db'
        cls.conn = sqlite3.connect(cls.db_path)
        cls.conn.row_factory = sqlite3.Row

    @classmethod
    def tearDownClass(cls):
        cls.conn.close()

    def test_database_exists(self):
        """Database file should exist"""
        self.assertTrue(Path(self.db_path).exists())

    def test_financial_statements_table_exists(self):
        """financial_statements table should exist"""
        cur = self.conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='financial_statements'")
        self.assertIsNotNone(cur.fetchone())

    def test_remuneration_table_exists(self):
        """remuneration table should exist"""
        cur = self.conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='remuneration'")
        self.assertIsNotNone(cur.fetchone())

    def test_financial_statements_columns(self):
        """financial_statements should have required columns"""
        cur = self.conn.cursor()
        cur.execute("PRAGMA table_info(financial_statements)")
        columns = {row['name'] for row in cur.fetchall()}
        required = {'band_number', 'band_name', 'fiscal_year', 'total_revenue',
                    'total_expenditures', 'net_revenue', 'pdf_path'}
        self.assertTrue(required.issubset(columns))

    def test_remuneration_columns(self):
        """remuneration should have required columns"""
        cur = self.conn.cursor()
        cur.execute("PRAGMA table_info(remuneration)")
        columns = {row['name'] for row in cur.fetchall()}
        required = {'band_number', 'band_name', 'fiscal_year', 'position',
                    'name', 'total_remuneration', 'pdf_path'}
        self.assertTrue(required.issubset(columns))


class TestFinancialStatementData(unittest.TestCase):
    """Test financial statement data quality"""

    @classmethod
    def setUpClass(cls):
        cls.conn = sqlite3.connect('fnfta_llama.db')
        cls.conn.row_factory = sqlite3.Row

    @classmethod
    def tearDownClass(cls):
        cls.conn.close()

    def test_has_financial_statements(self):
        """Should have at least one financial statement"""
        cur = self.conn.cursor()
        cur.execute("SELECT COUNT(*) as count FROM financial_statements")
        count = cur.fetchone()['count']
        self.assertGreater(count, 0, "No financial statements found")

    def test_revenue_positive(self):
        """Total revenue should be positive where set"""
        cur = self.conn.cursor()
        cur.execute("SELECT * FROM financial_statements WHERE total_revenue IS NOT NULL AND total_revenue < 0")
        negative = cur.fetchall()
        self.assertEqual(len(negative), 0, f"Found {len(negative)} records with negative revenue")

    def test_expenditures_positive(self):
        """Total expenditures should be positive where set"""
        cur = self.conn.cursor()
        cur.execute("SELECT * FROM financial_statements WHERE total_expenditures IS NOT NULL AND total_expenditures < 0")
        negative = cur.fetchall()
        self.assertEqual(len(negative), 0, f"Found {len(negative)} records with negative expenditures")

    def test_reasonable_revenue_range(self):
        """Revenue should be within reasonable range (0 to 500M)"""
        cur = self.conn.cursor()
        cur.execute("SELECT * FROM financial_statements WHERE total_revenue > 500000000")
        too_high = cur.fetchall()
        self.assertEqual(len(too_high), 0, f"Found {len(too_high)} records with unreasonably high revenue")

    def test_net_revenue_calculation(self):
        """Net revenue should approximately equal revenue - expenditures (with tolerance for other income items)"""
        cur = self.conn.cursor()
        cur.execute("""
            SELECT band_name, fiscal_year, total_revenue, total_expenditures, net_revenue,
                   ABS(net_revenue - (total_revenue - total_expenditures)) as diff
            FROM financial_statements
            WHERE total_revenue IS NOT NULL
              AND total_expenditures IS NOT NULL
              AND net_revenue IS NOT NULL
        """)
        mismatches = []
        for row in cur.fetchall():
            # Allow 25% tolerance - net revenue often includes other comprehensive income
            expected = row['total_revenue'] - row['total_expenditures']
            tolerance = abs(expected) * 0.25 + 5000  # 25% tolerance + $5000
            if row['diff'] > tolerance:
                mismatches.append(f"{row['band_name']} {row['fiscal_year']}: diff=${row['diff']:,.0f}")
        # Allow some mismatches - net revenue often includes items beyond simple calculation
        self.assertLess(len(mismatches), 3, f"Too many net revenue mismatches: {mismatches}")

    def test_valid_fiscal_years(self):
        """Fiscal years should be in valid format"""
        cur = self.conn.cursor()
        cur.execute("SELECT DISTINCT fiscal_year FROM financial_statements")
        valid_years = {'2019-2020', '2020-2021', '2021-2022', '2022-2023', '2023-2024'}
        for row in cur.fetchall():
            self.assertIn(row['fiscal_year'], valid_years, f"Invalid fiscal year: {row['fiscal_year']}")

    def test_unique_band_year_combinations(self):
        """Each band should have at most one record per fiscal year"""
        cur = self.conn.cursor()
        cur.execute("""
            SELECT band_number, fiscal_year, COUNT(*) as count
            FROM financial_statements
            GROUP BY band_number, fiscal_year
            HAVING COUNT(*) > 1
        """)
        duplicates = cur.fetchall()
        self.assertEqual(len(duplicates), 0, f"Found duplicate band/year combinations")


class TestRemunerationData(unittest.TestCase):
    """Test remuneration data quality"""

    @classmethod
    def setUpClass(cls):
        cls.conn = sqlite3.connect('fnfta_llama.db')
        cls.conn.row_factory = sqlite3.Row

    @classmethod
    def tearDownClass(cls):
        cls.conn.close()

    def test_has_remuneration_records(self):
        """Should have remuneration records"""
        cur = self.conn.cursor()
        cur.execute("SELECT COUNT(*) as count FROM remuneration")
        count = cur.fetchone()['count']
        self.assertGreater(count, 0, "No remuneration records found")

    def test_total_remuneration_not_negative(self):
        """Total remuneration should not be negative"""
        cur = self.conn.cursor()
        cur.execute("SELECT * FROM remuneration WHERE total_remuneration < 0")
        negative = cur.fetchall()
        self.assertEqual(len(negative), 0, f"Found {len(negative)} records with negative remuneration")

    def test_reasonable_remuneration_range(self):
        """Individual remuneration should be reasonable (< $1M)"""
        cur = self.conn.cursor()
        cur.execute("SELECT * FROM remuneration WHERE total_remuneration > 1000000")
        too_high = cur.fetchall()
        self.assertEqual(len(too_high), 0, f"Found {len(too_high)} records with unreasonably high remuneration")

    def test_has_positions(self):
        """All records should have positions"""
        cur = self.conn.cursor()
        cur.execute("SELECT * FROM remuneration WHERE position IS NULL OR position = ''")
        missing = cur.fetchall()
        self.assertEqual(len(missing), 0, f"Found {len(missing)} records without positions")

    def test_has_names(self):
        """All records should have names"""
        cur = self.conn.cursor()
        cur.execute("SELECT * FROM remuneration WHERE name IS NULL OR name = ''")
        missing = cur.fetchall()
        self.assertEqual(len(missing), 0, f"Found {len(missing)} records without names")

    def test_valid_positions(self):
        """Positions should be recognizable (Chief, Councillor, etc.)"""
        cur = self.conn.cursor()
        cur.execute("SELECT DISTINCT position FROM remuneration")
        valid_positions = {'Chief', 'Councillor', 'Council', 'Band Council', 'Deputy Chief',
                          'Band Manager', 'Administrator', 'Executive Director', 'Unknown'}
        for row in cur.fetchall():
            # Check if position contains any valid keyword
            position = row['position']
            found = any(vp.lower() in position.lower() for vp in valid_positions)
            self.assertTrue(found, f"Unknown position type: {position}")

    def test_component_sum_matches_total(self):
        """Component sums should approximately match total remuneration (with tolerance for benefits)"""
        cur = self.conn.cursor()
        cur.execute("""
            SELECT band_name, fiscal_year, name, position,
                   COALESCE(salary, 0) + COALESCE(honorarium, 0) + COALESCE(travel, 0) +
                   COALESCE(contract, 0) + COALESCE(other, 0) as component_sum,
                   total_remuneration,
                   ABS(total_remuneration - (COALESCE(salary, 0) + COALESCE(honorarium, 0) +
                       COALESCE(travel, 0) + COALESCE(contract, 0) + COALESCE(other, 0))) as diff
            FROM remuneration
            WHERE total_remuneration IS NOT NULL
        """)
        mismatches = []
        for row in cur.fetchall():
            if row['total_remuneration'] and row['component_sum']:
                # Allow 50% tolerance - totals often include benefits, pension, etc. not broken out
                tolerance = max(row['total_remuneration'] * 0.5, 500)
                if row['diff'] > tolerance:
                    mismatches.append(f"{row['name']} ({row['position']}): sum={row['component_sum']}, total={row['total_remuneration']}")
        # Many mismatches are expected since totals often include unlisted benefits
        total_records = cur.execute("SELECT COUNT(*) FROM remuneration").fetchone()[0]
        mismatch_rate = len(mismatches) / total_records if total_records > 0 else 0
        self.assertLess(mismatch_rate, 0.5, f"Too many component sum mismatches ({mismatch_rate:.1%}): {mismatches[:5]}")


class TestDataConsistency(unittest.TestCase):
    """Test cross-table data consistency"""

    @classmethod
    def setUpClass(cls):
        cls.conn = sqlite3.connect('fnfta_llama.db')
        cls.conn.row_factory = sqlite3.Row

    @classmethod
    def tearDownClass(cls):
        cls.conn.close()

    def test_band_names_consistent(self):
        """Band names should be consistent across tables"""
        cur = self.conn.cursor()
        cur.execute("SELECT DISTINCT band_name FROM financial_statements")
        fs_bands = {row['band_name'] for row in cur.fetchall()}

        cur.execute("SELECT DISTINCT band_name FROM remuneration")
        rem_bands = {row['band_name'] for row in cur.fetchall()}

        # At least some overlap expected
        overlap = fs_bands & rem_bands
        self.assertGreater(len(overlap), 0, "No overlapping band names between tables")

    def test_fiscal_years_consistent(self):
        """Fiscal years should be consistent across tables"""
        cur = self.conn.cursor()
        cur.execute("SELECT DISTINCT fiscal_year FROM financial_statements")
        fs_years = {row['fiscal_year'] for row in cur.fetchall()}

        cur.execute("SELECT DISTINCT fiscal_year FROM remuneration")
        rem_years = {row['fiscal_year'] for row in cur.fetchall()}

        # Years should all be from the same set
        all_years = fs_years | rem_years
        valid_years = {'2019-2020', '2020-2021', '2021-2022', '2022-2023', '2023-2024'}
        self.assertTrue(all_years.issubset(valid_years))

    def test_pdf_paths_exist(self):
        """PDF paths should reference existing files"""
        cur = self.conn.cursor()
        cur.execute("SELECT DISTINCT pdf_path FROM financial_statements")
        for row in cur.fetchall():
            path = Path(row['pdf_path'])
            self.assertTrue(path.exists(), f"PDF not found: {path}")

        cur.execute("SELECT DISTINCT pdf_path FROM remuneration")
        for row in cur.fetchall():
            path = Path(row['pdf_path'])
            self.assertTrue(path.exists(), f"PDF not found: {path}")


class TestDataQualityMetrics(unittest.TestCase):
    """Calculate and validate data quality metrics"""

    @classmethod
    def setUpClass(cls):
        cls.conn = sqlite3.connect('fnfta_llama.db')
        cls.conn.row_factory = sqlite3.Row

    @classmethod
    def tearDownClass(cls):
        cls.conn.close()

    def test_revenue_coverage(self):
        """Most financial statements should have revenue data"""
        cur = self.conn.cursor()
        cur.execute("SELECT COUNT(*) as total FROM financial_statements")
        total = cur.fetchone()['total']

        cur.execute("SELECT COUNT(*) as with_revenue FROM financial_statements WHERE total_revenue IS NOT NULL")
        with_revenue = cur.fetchone()['with_revenue']

        coverage = with_revenue / total if total > 0 else 0
        self.assertGreater(coverage, 0.8, f"Revenue coverage too low: {coverage:.1%}")

    def test_expenditure_coverage(self):
        """Most financial statements should have expenditure data"""
        cur = self.conn.cursor()
        cur.execute("SELECT COUNT(*) as total FROM financial_statements")
        total = cur.fetchone()['total']

        cur.execute("SELECT COUNT(*) as with_exp FROM financial_statements WHERE total_expenditures IS NOT NULL")
        with_exp = cur.fetchone()['with_exp']

        coverage = with_exp / total if total > 0 else 0
        self.assertGreater(coverage, 0.8, f"Expenditure coverage too low: {coverage:.1%}")

    def test_remuneration_total_coverage(self):
        """Most remuneration records should have total amounts"""
        cur = self.conn.cursor()
        cur.execute("SELECT COUNT(*) as total FROM remuneration")
        total = cur.fetchone()['total']

        cur.execute("SELECT COUNT(*) as with_total FROM remuneration WHERE total_remuneration IS NOT NULL")
        with_total = cur.fetchone()['with_total']

        coverage = with_total / total if total > 0 else 0
        self.assertGreater(coverage, 0.9, f"Remuneration total coverage too low: {coverage:.1%}")


def run_tests():
    """Run all tests and print summary"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestDatabaseSchema))
    suite.addTests(loader.loadTestsFromTestCase(TestFinancialStatementData))
    suite.addTests(loader.loadTestsFromTestCase(TestRemunerationData))
    suite.addTests(loader.loadTestsFromTestCase(TestDataConsistency))
    suite.addTests(loader.loadTestsFromTestCase(TestDataQualityMetrics))

    # Run tests with verbosity
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Print summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")

    if result.failures:
        print("\nFailed tests:")
        for test, traceback in result.failures:
            print(f"  - {test}")

    if result.errors:
        print("\nError tests:")
        for test, traceback in result.errors:
            print(f"  - {test}")

    return len(result.failures) == 0 and len(result.errors) == 0


if __name__ == '__main__':
    success = run_tests()
    exit(0 if success else 1)
