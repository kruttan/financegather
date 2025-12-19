#!/usr/bin/env python3
"""
FNFTA Financial Data Extractor

Extracts financial data from First Nations Financial Transparency Act PDFs
and stores in a denormalized SQLite database.
"""

import json
import os
import re
import sqlite3
import logging
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any
from datetime import datetime

# PDF extraction library
try:
    import pdfplumber
except ImportError:
    print("Installing pdfplumber...")
    import subprocess
    subprocess.check_call(['pip', 'install', 'pdfplumber'])
    import pdfplumber

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('extraction.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class FinancialStatement:
    """Extracted financial statement data"""
    band_number: int
    band_name: str
    fiscal_year: str
    fiscal_year_end: Optional[str] = None

    # Statement of Operations
    total_revenue: Optional[float] = None
    total_expenditures: Optional[float] = None
    net_revenue: Optional[float] = None

    # Revenue breakdown
    revenue_isc: Optional[float] = None  # Indigenous Services Canada
    revenue_health_canada: Optional[float] = None
    revenue_provincial: Optional[float] = None
    revenue_federal_other: Optional[float] = None
    revenue_own_source: Optional[float] = None
    revenue_other: Optional[float] = None

    # Expenditure breakdown
    exp_administration: Optional[float] = None
    exp_education: Optional[float] = None
    exp_health: Optional[float] = None
    exp_social_services: Optional[float] = None
    exp_housing: Optional[float] = None
    exp_economic_development: Optional[float] = None
    exp_infrastructure: Optional[float] = None
    exp_other: Optional[float] = None

    # Statement of Financial Position
    total_financial_assets: Optional[float] = None
    total_liabilities: Optional[float] = None
    net_financial_assets: Optional[float] = None
    total_non_financial_assets: Optional[float] = None
    net_assets: Optional[float] = None
    accumulated_surplus: Optional[float] = None

    # Key balance sheet items
    cash_and_equivalents: Optional[float] = None
    accounts_receivable: Optional[float] = None
    accounts_payable: Optional[float] = None
    deferred_revenue: Optional[float] = None
    long_term_debt: Optional[float] = None
    tangible_capital_assets: Optional[float] = None

    # Metadata
    auditor_name: Optional[str] = None
    audit_opinion: Optional[str] = None
    extraction_confidence: float = 0.0
    extraction_notes: str = ""
    pdf_path: str = ""
    extracted_at: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class RemunerationRecord:
    """Extracted remuneration data for elected officials"""
    band_number: int
    band_name: str
    fiscal_year: str

    position: str  # Chief, Councillor, etc.
    name: str
    months_in_position: Optional[int] = None

    salary: Optional[float] = None
    honorarium: Optional[float] = None
    travel: Optional[float] = None
    contract: Optional[float] = None
    other: Optional[float] = None
    total_remuneration: Optional[float] = None

    extraction_confidence: float = 0.0
    extraction_notes: str = ""
    pdf_path: str = ""
    extracted_at: str = field(default_factory=lambda: datetime.now().isoformat())


def parse_currency(value: str) -> Optional[float]:
    """Parse currency string to float"""
    if not value or value.strip() in ['-', '—', '', 'nil', 'Nil', 'NIL']:
        return None

    # Remove currency symbols, commas, spaces
    cleaned = re.sub(r'[$,\s]', '', str(value))

    # Handle parentheses for negative numbers
    if cleaned.startswith('(') and cleaned.endswith(')'):
        cleaned = '-' + cleaned[1:-1]

    try:
        return float(cleaned)
    except ValueError:
        return None


def extract_tables_from_pdf(pdf_path: str) -> List[List[List[str]]]:
    """Extract all tables from a PDF"""
    tables = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_tables = page.extract_tables()
                if page_tables:
                    tables.extend(page_tables)
    except Exception as e:
        logger.error(f"Error extracting tables from {pdf_path}: {e}")
    return tables


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract all text from a PDF"""
    text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        logger.error(f"Error extracting text from {pdf_path}: {e}")
    return text


class FinancialStatementExtractor:
    """Extracts data from audited financial statement PDFs"""

    def __init__(self, pdf_path: str, band_number: int, band_name: str, fiscal_year: str):
        self.pdf_path = pdf_path
        self.band_number = band_number
        self.band_name = band_name
        self.fiscal_year = fiscal_year
        self.tables = []
        self.text = ""
        self.confidence = 0.0
        self.notes = []

    def extract(self) -> FinancialStatement:
        """Main extraction method"""
        logger.info(f"Extracting financial statement: {self.pdf_path}")

        self.tables = extract_tables_from_pdf(self.pdf_path)
        self.text = extract_text_from_pdf(self.pdf_path)

        fs = FinancialStatement(
            band_number=self.band_number,
            band_name=self.band_name,
            fiscal_year=self.fiscal_year,
            pdf_path=self.pdf_path
        )

        # Extract fiscal year end date
        fs.fiscal_year_end = self._extract_fiscal_year_end()

        # Extract statement of operations data
        self._extract_operations(fs)

        # Extract statement of financial position data
        self._extract_financial_position(fs)

        # Extract auditor info
        fs.auditor_name = self._extract_auditor()
        fs.audit_opinion = self._extract_audit_opinion()

        # Calculate confidence score
        fs.extraction_confidence = self._calculate_confidence(fs)
        fs.extraction_notes = "; ".join(self.notes) if self.notes else ""

        return fs

    def _extract_fiscal_year_end(self) -> Optional[str]:
        """Extract fiscal year end date"""
        patterns = [
            r'March\s+31[,\s]+(\d{4})',
            r'(\d{4})-03-31',
            r'Year\s+[Ee]nded\s+March\s+31[,\s]+(\d{4})',
        ]
        for pattern in patterns:
            match = re.search(pattern, self.text)
            if match:
                year = match.group(1)
                return f"{year}-03-31"
        return None

    def _extract_operations(self, fs: FinancialStatement):
        """Extract statement of operations data"""
        # Look for tables containing revenue/expenditure data
        for table in self.tables:
            if not table:
                continue

            table_text = ' '.join([' '.join([str(c) for c in row if c]) for row in table if row]).lower()

            # Check if this looks like a statement of operations
            if 'revenue' in table_text and ('expenditure' in table_text or 'expense' in table_text):
                self._parse_operations_table(table, fs)
                break

        # Also try text-based extraction
        self._extract_operations_from_text(fs)

    def _parse_operations_table(self, table: List[List[str]], fs: FinancialStatement):
        """Parse a statement of operations table"""
        in_revenue = False
        in_expenditure = False

        for row in table:
            if not row:
                continue

            row_text = ' '.join([str(c) for c in row if c]).lower()

            # Detect sections
            if 'revenue' in row_text and 'total' not in row_text:
                in_revenue = True
                in_expenditure = False
                continue
            elif 'expenditure' in row_text or 'expense' in row_text:
                in_revenue = False
                in_expenditure = True
                continue

            # Extract totals
            if 'total revenue' in row_text or (in_revenue and 'total' in row_text):
                fs.total_revenue = self._get_number_from_row(row)
                in_revenue = False
            elif 'total expenditure' in row_text or 'total expense' in row_text or (in_expenditure and 'total' in row_text):
                fs.total_expenditures = self._get_number_from_row(row)
                in_expenditure = False
            elif 'net revenue' in row_text or 'surplus' in row_text or 'excess' in row_text:
                val = self._get_number_from_row(row)
                if val is not None:
                    fs.net_revenue = val

            # Revenue categories
            if in_revenue:
                if 'indigenous services' in row_text or 'isc' in row_text or 'inac' in row_text:
                    fs.revenue_isc = self._get_number_from_row(row)
                elif 'health canada' in row_text:
                    fs.revenue_health_canada = self._get_number_from_row(row)
                elif 'provincial' in row_text or 'province' in row_text:
                    fs.revenue_provincial = self._get_number_from_row(row)

            # Expenditure categories
            if in_expenditure:
                if 'administration' in row_text or 'admin' in row_text:
                    fs.exp_administration = self._get_number_from_row(row)
                elif 'education' in row_text:
                    fs.exp_education = self._get_number_from_row(row)
                elif 'health' in row_text:
                    fs.exp_health = self._get_number_from_row(row)
                elif 'social' in row_text:
                    fs.exp_social_services = self._get_number_from_row(row)
                elif 'housing' in row_text or 'capital' in row_text:
                    fs.exp_housing = self._get_number_from_row(row)

    def _extract_operations_from_text(self, fs: FinancialStatement):
        """Extract operations data from text when tables fail"""
        patterns = {
            'total_revenue': [
                r'total\s+revenue[^\d$]*(\$?[\d,]+(?:\.\d{2})?)',
                r'(\$?[\d,]+(?:\.\d{2})?)\s+total\s+revenue',
            ],
            'total_expenditures': [
                r'total\s+expenditure[s]?[^\d$]*(\$?[\d,]+(?:\.\d{2})?)',
                r'(\$?[\d,]+(?:\.\d{2})?)\s+total\s+expenditure',
            ],
            'net_revenue': [
                r'net\s+revenue[^\d$]*(\$?[\d,]+(?:\.\d{2})?)',
                r'annual\s+surplus[^\d$]*(\$?[\d,]+(?:\.\d{2})?)',
                r'excess\s+of\s+revenue[^\d$]*(\$?[\d,]+(?:\.\d{2})?)',
                r'accumulated\s+surplus[,\s]+end\s+of\s+year[^\d$]*(\$?[\d,]+(?:\.\d{2})?)',
            ],
        }

        for field, pattern_list in patterns.items():
            if getattr(fs, field) is None:
                for pattern in pattern_list:
                    match = re.search(pattern, self.text, re.IGNORECASE)
                    if match:
                        val = parse_currency(match.group(1))
                        if val is not None:
                            setattr(fs, field, val)
                            break

        # Also try word-based extraction for better accuracy
        if fs.total_revenue is None or fs.total_expenditures is None:
            self._extract_from_word_positions(fs)

    def _extract_financial_position(self, fs: FinancialStatement):
        """Extract statement of financial position data"""
        for table in self.tables:
            if not table:
                continue

            table_text = ' '.join([' '.join([str(c) for c in row if c]) for row in table if row]).lower()

            # Check if this looks like a statement of financial position
            if ('financial asset' in table_text or 'asset' in table_text) and 'liabilit' in table_text:
                self._parse_financial_position_table(table, fs)
                break

        # Text-based extraction as backup
        self._extract_financial_position_from_text(fs)

    def _parse_financial_position_table(self, table: List[List[str]], fs: FinancialStatement):
        """Parse a statement of financial position table"""
        for row in table:
            if not row:
                continue

            row_text = ' '.join([str(c) for c in row if c]).lower()

            # Financial assets
            if 'cash' in row_text and 'equivalent' in row_text:
                fs.cash_and_equivalents = self._get_number_from_row(row)
            elif 'accounts receivable' in row_text or 'receivable' in row_text:
                val = self._get_number_from_row(row)
                if val is not None and fs.accounts_receivable is None:
                    fs.accounts_receivable = val
            elif 'total financial asset' in row_text:
                fs.total_financial_assets = self._get_number_from_row(row)

            # Liabilities
            elif 'accounts payable' in row_text or 'payable' in row_text:
                val = self._get_number_from_row(row)
                if val is not None and fs.accounts_payable is None:
                    fs.accounts_payable = val
            elif 'deferred revenue' in row_text:
                fs.deferred_revenue = self._get_number_from_row(row)
            elif 'long-term debt' in row_text or 'long term debt' in row_text:
                fs.long_term_debt = self._get_number_from_row(row)
            elif 'total liabilit' in row_text:
                fs.total_liabilities = self._get_number_from_row(row)

            # Net positions
            elif 'net debt' in row_text or 'net financial' in row_text:
                fs.net_financial_assets = self._get_number_from_row(row)
            elif 'tangible capital' in row_text:
                fs.tangible_capital_assets = self._get_number_from_row(row)
            elif 'net asset' in row_text:
                fs.net_assets = self._get_number_from_row(row)
            elif 'accumulated surplus' in row_text:
                fs.accumulated_surplus = self._get_number_from_row(row)

    def _extract_financial_position_from_text(self, fs: FinancialStatement):
        """Extract financial position data from text"""
        patterns = {
            'cash_and_equivalents': [r'cash\s+and\s+cash\s+equivalents[^\d$]*(\$?[\d,]+)'],
            'total_liabilities': [r'total\s+liabilities[^\d$]*(\$?[\d,]+)'],
            'net_assets': [r'net\s+assets[^\d$]*(\$?[\d,]+)'],
            'accumulated_surplus': [r'accumulated\s+surplus[^\d$]*(\$?[\d,]+)'],
        }

        for field, pattern_list in patterns.items():
            if getattr(fs, field) is None:
                for pattern in pattern_list:
                    match = re.search(pattern, self.text, re.IGNORECASE)
                    if match:
                        val = parse_currency(match.group(1))
                        if val is not None:
                            setattr(fs, field, val)
                            break

    def _get_number_from_row(self, row: List[str]) -> Optional[float]:
        """Get the first valid number from a table row (typically the current year column)"""
        for i, cell in enumerate(row):
            if cell and i > 0:  # Skip first column (usually label)
                val = parse_currency(str(cell))
                if val is not None:
                    return val
        return None

    def _extract_from_word_positions(self, fs: FinancialStatement):
        """Extract financial data by analyzing word positions in PDF"""
        try:
            with pdfplumber.open(self.pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    words = page.extract_words()
                    if not words:
                        continue

                    # Group words by Y position
                    rows_dict = {}
                    for w in words:
                        y_key = round(w['top'])
                        if y_key not in rows_dict:
                            rows_dict[y_key] = []
                        rows_dict[y_key].append((w['x0'], w['text']))

                    # Merge close rows
                    sorted_y = sorted(rows_dict.keys())
                    merged_rows = []
                    current_row = []
                    prev_y = None

                    for y in sorted_y:
                        if prev_y is not None and abs(y - prev_y) > 12:
                            if current_row:
                                merged_rows.append(sorted(current_row, key=lambda x: x[0]))
                            current_row = rows_dict[y][:]
                        else:
                            current_row.extend(rows_dict[y])
                        prev_y = y

                    if current_row:
                        merged_rows.append(sorted(current_row, key=lambda x: x[0]))

                    # Look for key financial lines
                    for row in merged_rows:
                        row_text = ' '.join([w[1] for w in row]).lower()
                        numbers = [parse_currency(w[1]) for w in row if parse_currency(w[1]) is not None]

                        if not numbers:
                            continue

                        # Try to match key financial items
                        if fs.total_revenue is None and 'total' in row_text and 'revenue' in row_text:
                            fs.total_revenue = numbers[0] if numbers else None
                        elif fs.total_expenditures is None and 'total' in row_text and ('expenditure' in row_text or 'expense' in row_text):
                            fs.total_expenditures = numbers[0] if numbers else None
                        elif fs.net_revenue is None and 'net' in row_text and ('revenue' in row_text or 'surplus' in row_text):
                            fs.net_revenue = numbers[0] if numbers else None
                        elif fs.cash_and_equivalents is None and 'cash' in row_text and 'equivalent' in row_text:
                            fs.cash_and_equivalents = numbers[0] if numbers else None
                        elif fs.total_liabilities is None and 'total' in row_text and 'liabilit' in row_text:
                            fs.total_liabilities = numbers[0] if numbers else None
                        elif fs.accumulated_surplus is None and 'accumulated' in row_text and 'surplus' in row_text:
                            fs.accumulated_surplus = numbers[0] if numbers else None
                        elif fs.net_assets is None and 'net' in row_text and 'asset' in row_text:
                            fs.net_assets = numbers[0] if numbers else None

        except Exception as e:
            logger.error(f"Error in word position extraction: {e}")

    def _extract_auditor(self) -> Optional[str]:
        """Extract auditor firm name"""
        auditor_patterns = [
            r'([\w\s&]+(?:LLP|Inc\.?|Chartered Professional Accountants))',
            r'CHARTERED PROFESSIONAL ACCOUNTANTS\s*\n\s*([\w\s&]+)',
        ]

        # Common auditor names
        common_auditors = [
            'BDO', 'Deloitte', 'KPMG', 'Ernst & Young', 'EY', 'PwC',
            'MNP', 'Grant Thornton', 'RSM', 'Teed Saunders Doyle',
            'Chicken & Associates', 'Chicken',
        ]

        for auditor in common_auditors:
            if auditor.lower() in self.text.lower():
                return auditor

        return None

    def _extract_audit_opinion(self) -> Optional[str]:
        """Extract audit opinion type"""
        text_lower = self.text.lower()

        if 'adverse opinion' in text_lower:
            return 'adverse'
        elif 'disclaimer of opinion' in text_lower or 'disclaim' in text_lower:
            return 'disclaimer'
        elif 'qualified opinion' in text_lower or 'except for' in text_lower:
            return 'qualified'
        elif 'unqualified' in text_lower or 'present fairly' in text_lower:
            return 'unqualified'

        return None

    def _calculate_confidence(self, fs: FinancialStatement) -> float:
        """Calculate extraction confidence score (0-1)"""
        fields_to_check = [
            fs.total_revenue,
            fs.total_expenditures,
            fs.net_revenue,
            fs.total_liabilities,
            fs.net_assets,
            fs.cash_and_equivalents,
        ]

        extracted = sum(1 for f in fields_to_check if f is not None)
        confidence = extracted / len(fields_to_check)

        # Bonus for consistency checks
        if fs.total_revenue and fs.total_expenditures and fs.net_revenue:
            expected_net = fs.total_revenue - fs.total_expenditures
            if abs(expected_net - fs.net_revenue) < 1000:  # Allow small rounding differences
                confidence = min(1.0, confidence + 0.1)
            else:
                self.notes.append("Net revenue doesn't match revenue - expenditures")

        return round(confidence, 2)


class RemunerationExtractor:
    """Extracts data from remuneration schedule PDFs"""

    def __init__(self, pdf_path: str, band_number: int, band_name: str, fiscal_year: str):
        self.pdf_path = pdf_path
        self.band_number = band_number
        self.band_name = band_name
        self.fiscal_year = fiscal_year
        self.tables = []
        self.text = ""

    def extract(self) -> List[RemunerationRecord]:
        """Main extraction method"""
        logger.info(f"Extracting remuneration: {self.pdf_path}")

        self.tables = extract_tables_from_pdf(self.pdf_path)
        self.text = extract_text_from_pdf(self.pdf_path)

        records = []

        # First try standard table extraction
        for table in self.tables:
            if not table:
                continue

            table_text = ' '.join([' '.join([str(c) for c in row if c]) for row in table if row]).lower()

            # Check if this looks like a remuneration table
            if ('salary' in table_text or 'honorar' in table_text) and ('chief' in table_text or 'council' in table_text or 'position' in table_text):
                records.extend(self._parse_remuneration_table(table))

        # If no records found, try word-based extraction
        if not records:
            records = self._extract_from_words()

        # Set common fields
        for record in records:
            record.band_number = self.band_number
            record.band_name = self.band_name
            record.fiscal_year = self.fiscal_year
            record.pdf_path = self.pdf_path

        return records

    def _extract_from_words(self) -> List[RemunerationRecord]:
        """Extract remuneration data by analyzing word positions"""
        records = []

        try:
            with pdfplumber.open(self.pdf_path) as pdf:
                for page in pdf.pages:
                    words = page.extract_words()
                    if not words:
                        continue

                    # First, try to find data rows by looking for patterns
                    # Group all words by approximate Y position with larger tolerance
                    y_groups = {}
                    for w in words:
                        # Round to nearest 30 units to group visual rows
                        y_key = round(w['top'] / 30) * 30
                        if y_key not in y_groups:
                            y_groups[y_key] = []
                        y_groups[y_key].append((w['x0'], w['text'], w['top']))

                    # Sort each group by x position and collect text
                    for y_key in sorted(y_groups.keys()):
                        group = sorted(y_groups[y_key], key=lambda x: x[0])
                        row_text = [w[1] for w in group]
                        row_str = ' '.join(row_text).lower()

                        # Look for rows with position names and dollar amounts
                        has_position = any(p in row_str for p in ['chief', 'councillor', 'councilor'])
                        has_dollar = any('$' in t for t in row_text)

                        if has_position and has_dollar:
                            record = self._parse_word_row(row_text)
                            if record:
                                records.append(record)

                    # If no records found with grouping, try sequential pattern matching
                    if not records:
                        records = self._extract_sequential_pattern(words)

        except Exception as e:
            logger.error(f"Error in word-based extraction: {e}")

        return records

    def _extract_sequential_pattern(self, words: List[dict]) -> List[RemunerationRecord]:
        """Extract remuneration by finding position words and following dollar amounts"""
        records = []

        # Sort words by Y then X position
        sorted_words = sorted(words, key=lambda w: (w['top'], w['x0']))

        # Find all position indicators and all dollar amounts
        positions = []
        dollars = []
        names = []

        for i, w in enumerate(sorted_words):
            text = w['text']
            text_lower = text.lower()

            if text_lower in ['chief', 'councillor', 'councilor']:
                positions.append((i, w['top'], text.capitalize()))
            elif '$' in text:
                val = parse_currency(text)
                if val is not None:
                    dollars.append((i, w['top'], val))
            elif len(text) > 1 and text[0].isupper() and text.isalpha():
                if text_lower not in ['chief', 'councillor', 'councilor', 'number', 'position', 'name', 'months', 'salary', 'honorarium', 'travel', 'contract', 'total', 'of']:
                    names.append((i, w['top'], text))

        # For each position, find the dollar amounts that follow it (until next position)
        for p_idx, (pos_i, pos_y, position) in enumerate(positions):
            # Find next position's Y to limit search
            next_pos_y = positions[p_idx + 1][1] if p_idx + 1 < len(positions) else float('inf')

            # Get dollars between this position and the next
            position_dollars = []
            for d_i, d_y, d_val in dollars:
                # Dollar is "after" position in document flow
                if d_y > pos_y and d_y < next_pos_y:
                    position_dollars.append(d_val)
                elif len(position_dollars) > 0 and d_y >= next_pos_y:
                    break

            # Get names near this position
            position_names = []
            for n_i, n_y, n_text in names:
                # Allow names within a range of the position
                if abs(n_y - pos_y) < 5000:  # Large range due to unusual coordinates
                    position_names.append(n_text)
                    if len(position_names) >= 3:
                        break

            if position_dollars:
                record = RemunerationRecord(
                    band_number=0,
                    band_name="",
                    fiscal_year="",
                    position=position,
                    name=' '.join(position_names[:2]) if position_names else "Unknown",
                    extraction_confidence=0.6
                )

                # Assign numbers based on count
                if len(position_dollars) >= 5:
                    record.salary = position_dollars[0]
                    record.honorarium = position_dollars[1]
                    record.contract = position_dollars[2]
                    record.travel = position_dollars[3]
                    record.total_remuneration = position_dollars[4]
                elif len(position_dollars) >= 2:
                    record.salary = position_dollars[0]
                    record.total_remuneration = position_dollars[-1]
                elif len(position_dollars) == 1:
                    record.total_remuneration = position_dollars[0]

                records.append(record)

        return records

    def _parse_word_row(self, row_text: List[str]) -> Optional[RemunerationRecord]:
        """Parse a row of words into a remuneration record"""
        # Find position
        position = None
        name_parts = []
        numbers = []

        for word in row_text:
            word_lower = word.lower()
            if word_lower in ['chief', 'councillor', 'councilor']:
                position = word.capitalize()
            elif word_lower == 'council' or word_lower == 'member':
                if position:
                    position = position + ' ' + word.capitalize()
                else:
                    position = word.capitalize()
            elif '$' in word:
                val = parse_currency(word)
                if val is not None:
                    numbers.append(val)
            elif word.isdigit() and len(word) <= 2:
                # Likely months
                continue
            elif not word.isdigit() and len(word) > 1 and word[0].isupper():
                # Likely name part
                name_parts.append(word)

        if not position or not numbers:
            return None

        name = ' '.join(name_parts) if name_parts else "Unknown"

        record = RemunerationRecord(
            band_number=0,
            band_name="",
            fiscal_year="",
            position=position,
            name=name,
            extraction_confidence=0.6
        )

        # Assign numbers based on count
        if len(numbers) >= 5:
            record.salary = numbers[0]
            record.honorarium = numbers[1]
            record.contract = numbers[2]
            record.travel = numbers[3]
            record.total_remuneration = numbers[4]
        elif len(numbers) >= 4:
            record.salary = numbers[0]
            record.honorarium = numbers[1]
            record.travel = numbers[2]
            record.total_remuneration = numbers[3]
        elif len(numbers) >= 2:
            record.salary = numbers[0]
            record.total_remuneration = numbers[-1]
        elif len(numbers) == 1:
            record.total_remuneration = numbers[0]

        return record

    def _parse_remuneration_table(self, table: List[List[str]]) -> List[RemunerationRecord]:
        """Parse a remuneration table"""
        records = []

        # Find header row to identify columns
        header_row = None
        header_idx = -1

        for i, row in enumerate(table):
            if not row:
                continue
            row_text = ' '.join([str(c) for c in row if c]).lower()
            if 'position' in row_text or 'salary' in row_text or 'name' in row_text:
                header_row = row
                header_idx = i
                break

        if header_row is None:
            return records

        # Map column indices
        col_map = {}
        for i, cell in enumerate(header_row):
            if cell:
                cell_lower = str(cell).lower()
                if 'position' in cell_lower:
                    col_map['position'] = i
                elif 'name' in cell_lower:
                    col_map['name'] = i
                elif 'month' in cell_lower:
                    col_map['months'] = i
                elif 'salary' in cell_lower:
                    col_map['salary'] = i
                elif 'honorar' in cell_lower:
                    col_map['honorarium'] = i
                elif 'travel' in cell_lower:
                    col_map['travel'] = i
                elif 'contract' in cell_lower:
                    col_map['contract'] = i
                elif 'other' in cell_lower:
                    col_map['other'] = i
                elif 'total' in cell_lower:
                    col_map['total'] = i

        # Parse data rows
        for row in table[header_idx + 1:]:
            if not row or not any(row):
                continue

            # Skip if this looks like a header or total row
            row_text = ' '.join([str(c) for c in row if c]).lower()
            if 'total' in row_text and 'chief' not in row_text and 'council' not in row_text:
                continue

            record = RemunerationRecord(
                band_number=0,  # Will be set later
                band_name="",
                fiscal_year="",
                position="",
                name=""
            )

            # Extract values
            if 'position' in col_map and col_map['position'] < len(row):
                record.position = str(row[col_map['position']] or "").strip()
            if 'name' in col_map and col_map['name'] < len(row):
                record.name = str(row[col_map['name']] or "").strip()
            if 'months' in col_map and col_map['months'] < len(row):
                try:
                    record.months_in_position = int(float(str(row[col_map['months']] or 0)))
                except:
                    pass
            if 'salary' in col_map and col_map['salary'] < len(row):
                record.salary = parse_currency(str(row[col_map['salary']] or ""))
            if 'honorarium' in col_map and col_map['honorarium'] < len(row):
                record.honorarium = parse_currency(str(row[col_map['honorarium']] or ""))
            if 'travel' in col_map and col_map['travel'] < len(row):
                record.travel = parse_currency(str(row[col_map['travel']] or ""))
            if 'contract' in col_map and col_map['contract'] < len(row):
                record.contract = parse_currency(str(row[col_map['contract']] or ""))
            if 'other' in col_map and col_map['other'] < len(row):
                record.other = parse_currency(str(row[col_map['other']] or ""))
            if 'total' in col_map and col_map['total'] < len(row):
                record.total_remuneration = parse_currency(str(row[col_map['total']] or ""))

            # Calculate confidence
            if record.name and (record.salary or record.total_remuneration):
                record.extraction_confidence = 0.8

                # Verify total
                if record.total_remuneration:
                    calculated = sum(filter(None, [
                        record.salary, record.honorarium, record.travel,
                        record.contract, record.other
                    ]))
                    if abs(calculated - record.total_remuneration) < 10:
                        record.extraction_confidence = 1.0

                records.append(record)

        return records


class FNFTADatabase:
    """SQLite database for FNFTA financial data"""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = None

    def connect(self):
        """Connect to database"""
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row

    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()

    def create_tables(self):
        """Create database tables"""
        self.connect()
        cursor = self.conn.cursor()

        # Denormalized financial statements table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS financial_statements (
                id INTEGER PRIMARY KEY AUTOINCREMENT,

                -- Band identification
                band_number INTEGER NOT NULL,
                band_name TEXT NOT NULL,
                band_address TEXT,
                band_province TEXT,

                -- Fiscal period
                fiscal_year TEXT NOT NULL,
                fiscal_year_end TEXT,

                -- Statement of Operations - Revenue
                total_revenue REAL,
                revenue_isc REAL,
                revenue_health_canada REAL,
                revenue_provincial REAL,
                revenue_federal_other REAL,
                revenue_own_source REAL,
                revenue_other REAL,

                -- Statement of Operations - Expenditures
                total_expenditures REAL,
                exp_administration REAL,
                exp_education REAL,
                exp_health REAL,
                exp_social_services REAL,
                exp_housing REAL,
                exp_economic_development REAL,
                exp_infrastructure REAL,
                exp_other REAL,

                -- Statement of Operations - Net
                net_revenue REAL,

                -- Statement of Financial Position - Assets
                total_financial_assets REAL,
                cash_and_equivalents REAL,
                accounts_receivable REAL,
                tangible_capital_assets REAL,
                total_non_financial_assets REAL,

                -- Statement of Financial Position - Liabilities
                total_liabilities REAL,
                accounts_payable REAL,
                deferred_revenue REAL,
                long_term_debt REAL,

                -- Statement of Financial Position - Net
                net_financial_assets REAL,
                net_assets REAL,
                accumulated_surplus REAL,

                -- Audit info
                auditor_name TEXT,
                audit_opinion TEXT,

                -- Extraction metadata
                extraction_confidence REAL,
                extraction_notes TEXT,
                pdf_path TEXT,
                extracted_at TEXT,

                UNIQUE(band_number, fiscal_year)
            )
        ''')

        # Denormalized remuneration table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS remuneration (
                id INTEGER PRIMARY KEY AUTOINCREMENT,

                -- Band identification
                band_number INTEGER NOT NULL,
                band_name TEXT NOT NULL,
                band_address TEXT,
                band_province TEXT,

                -- Fiscal period
                fiscal_year TEXT NOT NULL,

                -- Official info
                position TEXT NOT NULL,
                name TEXT NOT NULL,
                months_in_position INTEGER,

                -- Compensation
                salary REAL,
                honorarium REAL,
                travel REAL,
                contract REAL,
                other REAL,
                total_remuneration REAL,

                -- Extraction metadata
                extraction_confidence REAL,
                extraction_notes TEXT,
                pdf_path TEXT,
                extracted_at TEXT,

                UNIQUE(band_number, fiscal_year, name)
            )
        ''')

        # Create indices for common queries
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_fs_band ON financial_statements(band_number)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_fs_year ON financial_statements(fiscal_year)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_fs_band_year ON financial_statements(band_number, fiscal_year)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_rem_band ON remuneration(band_number)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_rem_year ON remuneration(fiscal_year)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_rem_position ON remuneration(position)')

        self.conn.commit()
        self.close()

    def insert_financial_statement(self, fs: FinancialStatement, band_info: dict = None):
        """Insert a financial statement record"""
        self.connect()
        cursor = self.conn.cursor()

        data = asdict(fs)

        # Add band info if provided
        if band_info:
            data['band_address'] = band_info.get('address')
            data['band_province'] = band_info.get('province')

        columns = ', '.join(data.keys())
        placeholders = ', '.join(['?' for _ in data])

        cursor.execute(f'''
            INSERT OR REPLACE INTO financial_statements ({columns})
            VALUES ({placeholders})
        ''', list(data.values()))

        self.conn.commit()
        self.close()

    def insert_remuneration(self, record: RemunerationRecord, band_info: dict = None):
        """Insert a remuneration record"""
        self.connect()
        cursor = self.conn.cursor()

        data = asdict(record)

        # Add band info if provided
        if band_info:
            data['band_address'] = band_info.get('address')
            data['band_province'] = band_info.get('province')

        columns = ', '.join(data.keys())
        placeholders = ', '.join(['?' for _ in data])

        cursor.execute(f'''
            INSERT OR REPLACE INTO remuneration ({columns})
            VALUES ({placeholders})
        ''', list(data.values()))

        self.conn.commit()
        self.close()

    def get_stats(self) -> dict:
        """Get database statistics"""
        self.connect()
        cursor = self.conn.cursor()

        stats = {}

        cursor.execute('SELECT COUNT(*) FROM financial_statements')
        stats['financial_statements_count'] = cursor.fetchone()[0]

        cursor.execute('SELECT COUNT(*) FROM remuneration')
        stats['remuneration_count'] = cursor.fetchone()[0]

        cursor.execute('SELECT COUNT(DISTINCT band_number) FROM financial_statements')
        stats['unique_bands_fs'] = cursor.fetchone()[0]

        cursor.execute('SELECT COUNT(DISTINCT fiscal_year) FROM financial_statements')
        stats['unique_years_fs'] = cursor.fetchone()[0]

        cursor.execute('SELECT AVG(extraction_confidence) FROM financial_statements')
        stats['avg_confidence_fs'] = cursor.fetchone()[0]

        cursor.execute('SELECT AVG(extraction_confidence) FROM remuneration')
        stats['avg_confidence_rem'] = cursor.fetchone()[0]

        self.close()
        return stats


def load_metadata(data_dir: str) -> tuple:
    """Load metadata from JSON files"""
    bands = {}
    documents = []

    bands_path = os.path.join(data_dir, 'metadata', 'bands.json')
    docs_path = os.path.join(data_dir, 'metadata', 'documents.json')

    with open(bands_path) as f:
        bands_list = json.load(f)
        for band in bands_list:
            bands[band['number']] = band

    with open(docs_path) as f:
        documents = json.load(f)

    return bands, documents


def extract_all(data_dir: str, db_path: str, limit: int = None):
    """Extract all financial data from PDFs"""
    logger.info(f"Starting extraction from {data_dir}")

    # Load metadata
    bands, documents = load_metadata(data_dir)
    logger.info(f"Loaded {len(bands)} bands and {len(documents)} documents")

    # Initialize database
    db = FNFTADatabase(db_path)
    db.create_tables()

    # Process documents
    fs_count = 0
    rem_count = 0
    error_count = 0

    for i, doc in enumerate(documents):
        if limit and i >= limit:
            break

        if not doc.get('downloaded'):
            continue

        pdf_path = os.path.join(data_dir, '..', doc['local_path'])
        if not os.path.exists(pdf_path):
            logger.warning(f"PDF not found: {pdf_path}")
            continue

        band_info = bands.get(doc['band_number'], {})

        try:
            if doc['doc_type'] == 'financial_statement':
                extractor = FinancialStatementExtractor(
                    pdf_path=pdf_path,
                    band_number=doc['band_number'],
                    band_name=doc['band_name'],
                    fiscal_year=doc['fiscal_year']
                )
                fs = extractor.extract()
                db.insert_financial_statement(fs, band_info)
                fs_count += 1

            elif doc['doc_type'] == 'remuneration':
                extractor = RemunerationExtractor(
                    pdf_path=pdf_path,
                    band_number=doc['band_number'],
                    band_name=doc['band_name'],
                    fiscal_year=doc['fiscal_year']
                )
                records = extractor.extract()
                for record in records:
                    db.insert_remuneration(record, band_info)
                    rem_count += 1

        except Exception as e:
            logger.error(f"Error processing {pdf_path}: {e}")
            error_count += 1

        if (i + 1) % 100 == 0:
            logger.info(f"Processed {i + 1} documents...")

    # Final stats
    stats = db.get_stats()
    logger.info(f"Extraction complete!")
    logger.info(f"Financial statements: {fs_count}")
    logger.info(f"Remuneration records: {rem_count}")
    logger.info(f"Errors: {error_count}")
    logger.info(f"Database stats: {stats}")

    return stats


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Extract FNFTA financial data from PDFs')
    parser.add_argument('--data-dir', default='fnfta_data', help='Data directory')
    parser.add_argument('--db-path', default='fnfta_financials.db', help='SQLite database path')
    parser.add_argument('--limit', type=int, help='Limit number of documents to process')

    args = parser.parse_args()

    extract_all(args.data_dir, args.db_path, args.limit)
