# FNFTA Financial Data Extraction Project

This project extracts structured financial data from First Nations Financial Transparency Act (FNFTA) PDF documents using LlamaExtract.

## Overview

The FNFTA requires First Nations to publish audited financial statements and schedules of remuneration. This project:

1. Downloads PDFs from the FNFTA portal (already completed - 4,582 documents)
2. Extracts structured financial data using LlamaExtract REST API
3. Stores data in a denormalized SQLite database
4. Validates extracted data with comprehensive tests

## Data Sources

- **Source Portal**: https://fnp-ppn.aadnc-aandc.gc.ca
- **Total Bands**: 635 First Nations
- **Total Documents**: 4,582 PDFs
- **Fiscal Years**: 2019-2020 through 2023-2024
- **Document Types**:
  - Financial Statements: 2,317 documents
  - Remuneration Schedules: 2,265 documents

## Project Structure

```
financegather/
├── extract_llama.py          # Main extraction script using LlamaExtract
├── test_llama_extraction.py  # Validation tests (25 tests)
├── fnfta_llama.db           # SQLite database with extracted data
├── fnfta_data/
│   ├── metadata/
│   │   ├── bands.json       # 635 First Nations bands
│   │   ├── documents.json   # 4,582 document records
│   │   └── summary.json     # Aggregate statistics
│   └── documents/           # Downloaded PDFs organized by band/year
└── claude.md                # This file
```

## LlamaExtract Configuration

### API Setup

```python
LLAMA_API_KEY = "llx-..."  # Your LlamaIndex Cloud API key
LLAMA_API_BASE = "https://api.cloud.llamaindex.ai/api/v1"
```

### Extraction Modes

The script uses different extraction modes optimized for each document type:

| Mode | Use Case | Speed | Accuracy |
|------|----------|-------|----------|
| FAST | Simple OCR documents | Fastest | Basic |
| BALANCED | General documents | Medium | Good |
| MULTIMODAL | Visually rich documents (default) | Medium | Good |
| PREMIUM | Complex tables, financial statements | Slowest | Highest |

**Current Configuration**:
- Financial Statements: **PREMIUM** mode (complex tables require highest accuracy)
- Remuneration Schedules: **MULTIMODAL** mode (simpler tabular format)

### Timeout Settings

- Job polling timeout: **300 seconds** (5 minutes)
- PREMIUM mode typically takes 2-3 minutes for complex financial PDFs
- MULTIMODAL mode typically completes in 15-30 seconds

## Database Schema

### financial_statements table

```sql
CREATE TABLE financial_statements (
    id INTEGER PRIMARY KEY,
    band_number INTEGER NOT NULL,
    band_name TEXT NOT NULL,
    band_address TEXT,
    fiscal_year TEXT NOT NULL,
    fiscal_year_end_date TEXT,

    -- Revenue
    total_revenue REAL,
    revenue_isc REAL,              -- Indigenous Services Canada
    revenue_health_canada REAL,
    revenue_provincial REAL,
    revenue_own_source REAL,

    -- Expenditures
    total_expenditures REAL,
    exp_administration REAL,
    exp_education REAL,
    exp_health REAL,
    exp_social REAL,
    exp_housing REAL,
    net_revenue REAL,

    -- Balance Sheet
    cash_and_equivalents REAL,
    accounts_receivable REAL,
    total_financial_assets REAL,
    tangible_capital_assets REAL,
    accounts_payable REAL,
    deferred_revenue REAL,
    long_term_debt REAL,
    total_liabilities REAL,
    net_assets REAL,
    accumulated_surplus REAL,

    -- Audit Info
    auditor_name TEXT,
    audit_opinion TEXT,

    -- Metadata
    pdf_path TEXT,
    extracted_at TEXT,

    UNIQUE(band_number, fiscal_year)
);
```

### remuneration table

```sql
CREATE TABLE remuneration (
    id INTEGER PRIMARY KEY,
    band_number INTEGER NOT NULL,
    band_name TEXT NOT NULL,
    band_address TEXT,
    fiscal_year TEXT NOT NULL,

    -- Official Info
    position TEXT NOT NULL,        -- Chief, Councillor, etc.
    name TEXT NOT NULL,
    months_in_position INTEGER,

    -- Compensation
    salary REAL,
    honorarium REAL,
    travel REAL,
    contract REAL,
    other REAL,
    total_remuneration REAL,

    -- Metadata
    pdf_path TEXT,
    extracted_at TEXT,

    UNIQUE(band_number, fiscal_year, name)
);
```

## Usage

### Run Extraction

```bash
# Extract all documents (full dataset)
python3 extract_llama.py

# Extract limited number for testing
python3 extract_llama.py --limit 20

# Specify custom paths
python3 extract_llama.py --data-dir fnfta_data --db-path fnfta_llama.db
```

### Run Validation Tests

```bash
python3 test_llama_extraction.py
```

### Query the Database

```python
import sqlite3

conn = sqlite3.connect('fnfta_llama.db')
conn.row_factory = sqlite3.Row

# Get financial summary by band
cur = conn.cursor()
cur.execute('''
    SELECT band_name, fiscal_year, total_revenue, total_expenditures, net_revenue
    FROM financial_statements
    ORDER BY band_name, fiscal_year
''')

for row in cur.fetchall():
    print(f"{row['band_name']} {row['fiscal_year']}: ${row['total_revenue']:,.0f}")

# Get remuneration by position
cur.execute('''
    SELECT position, AVG(total_remuneration) as avg_comp, COUNT(*) as count
    FROM remuneration
    GROUP BY position
    ORDER BY avg_comp DESC
''')
```

## Validation Tests

The test suite (`test_llama_extraction.py`) includes 25 tests across 5 categories:

### TestDatabaseSchema
- Database file exists
- Required tables exist
- Required columns present

### TestFinancialStatementData
- Has financial statements
- Revenue/expenditures are positive
- Values within reasonable ranges
- Net revenue calculation validation
- Valid fiscal years
- No duplicate band/year combinations

### TestRemunerationData
- Has remuneration records
- Amounts are non-negative and reasonable
- All records have positions and names
- Valid position types (Chief, Councillor, etc.)
- Component sums approximately match totals

### TestDataConsistency
- Band names consistent across tables
- Fiscal years consistent
- PDF paths reference existing files

### TestDataQualityMetrics
- Revenue coverage > 80%
- Expenditure coverage > 80%
- Remuneration total coverage > 90%

## Extraction Performance

Based on testing with the first 20 documents:

| Metric | Value |
|--------|-------|
| Financial Statements Success Rate | 100% |
| Remuneration Success Rate | 100% |
| Avg FS Extraction Time | ~150 seconds |
| Avg REM Extraction Time | ~15 seconds |
| Timeouts | 0 |

## Troubleshooting

### Timeout Issues

If extractions timeout:
1. Increase `max_wait` parameter in `wait_for_job()` (default: 300s)
2. Check if using appropriate extraction mode
3. Consider BALANCED mode for faster (but less accurate) extraction

### Empty Extraction Results

If `elected_officials` returns empty:
1. Check PDF format - some older formats may not parse correctly
2. Try PREMIUM mode for better table detection
3. Verify PDF is not corrupted

### API Errors

Common API errors:
- **404**: Agent not found - will auto-create new agent
- **429**: Rate limited - add delays between requests
- **500**: Server error - retry the request

## API Reference

### Endpoints Used

```
POST /extraction/extraction-agents     # Create extraction agent
GET  /extraction/extraction-agents/by-name/{name}  # Get agent by name
POST /files                            # Upload PDF file
POST /extraction/jobs                  # Create extraction job
GET  /extraction/jobs/{id}             # Check job status
GET  /extraction/jobs/{id}/result      # Get extraction results
```

### Agent Configuration

```python
{
    "name": "agent-name",
    "data_schema": { ... },  # JSON Schema
    "config": {
        "extraction_target": "PER_DOC",  # or PER_PAGE, PER_TABLE_ROW
        "extraction_mode": "PREMIUM"      # FAST, BALANCED, MULTIMODAL, PREMIUM
    }
}
```

## References

- [LlamaExtract Documentation](https://developers.llamaindex.ai/python/cloud/llamaextract/getting_started/python)
- [LlamaExtract Configuration Options](https://developers.llamaindex.ai/python/cloud/llamaextract/features/options/)
- [LlamaExtract REST API](https://developers.llamaindex.ai/python/cloud/llamaextract/getting_started/api/)
- [FNFTA Portal](https://fnp-ppn.aadnc-aandc.gc.ca)
