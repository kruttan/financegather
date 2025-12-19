#!/usr/bin/env python3
"""
FNFTA Financial Data Extractor using LlamaExtract REST API

Extracts financial data from First Nations Financial Transparency Act PDFs
using LlamaIndex's LlamaExtract service via REST API.
"""

import json
import os
import sqlite3
import logging
import time
import base64
import httpx
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('llama_extraction.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# API Configuration
LLAMA_API_KEY = "llx-9LqXdTqIkBVoDCMFS28zNth8eOKUn2F3vQwqUKdom3WunJv4"
LLAMA_API_BASE = "https://api.cloud.llamaindex.ai/api/v1"


# ============================================================================
# JSON Schemas for Extraction
# ============================================================================

FINANCIAL_STATEMENT_SCHEMA = {
    "type": "object",
    "properties": {
        "fiscal_year_end_date": {
            "type": "string",
            "description": "The fiscal year end date (e.g., March 31, 2024)"
        },
        "total_revenue": {
            "type": "number",
            "description": "Total revenue for the fiscal year"
        },
        "total_expenditures": {
            "type": "number",
            "description": "Total expenditures/expenses for the fiscal year"
        },
        "net_revenue": {
            "type": "number",
            "description": "Net revenue (surplus/deficit) - total revenue minus total expenditures"
        },
        "revenue_indigenous_services_canada": {
            "type": "number",
            "description": "Revenue from Indigenous Services Canada (ISC/INAC)"
        },
        "revenue_health_canada": {
            "type": "number",
            "description": "Revenue from Health Canada"
        },
        "revenue_provincial": {
            "type": "number",
            "description": "Revenue from provincial government"
        },
        "revenue_own_source": {
            "type": "number",
            "description": "Own-source revenue from business operations"
        },
        "expenditure_administration": {
            "type": "number",
            "description": "Administration expenses"
        },
        "expenditure_education": {
            "type": "number",
            "description": "Education expenses"
        },
        "expenditure_health": {
            "type": "number",
            "description": "Health services expenses"
        },
        "expenditure_social": {
            "type": "number",
            "description": "Social services expenses"
        },
        "expenditure_housing": {
            "type": "number",
            "description": "Housing and capital expenses"
        },
        "cash_and_equivalents": {
            "type": "number",
            "description": "Cash and cash equivalents"
        },
        "accounts_receivable": {
            "type": "number",
            "description": "Accounts receivable"
        },
        "total_financial_assets": {
            "type": "number",
            "description": "Total financial assets"
        },
        "tangible_capital_assets": {
            "type": "number",
            "description": "Tangible capital assets"
        },
        "accounts_payable": {
            "type": "number",
            "description": "Accounts payable"
        },
        "deferred_revenue": {
            "type": "number",
            "description": "Deferred revenue"
        },
        "long_term_debt": {
            "type": "number",
            "description": "Long-term debt"
        },
        "total_liabilities": {
            "type": "number",
            "description": "Total liabilities"
        },
        "net_assets": {
            "type": "number",
            "description": "Net assets"
        },
        "accumulated_surplus": {
            "type": "number",
            "description": "Accumulated surplus at end of year"
        },
        "auditor_firm": {
            "type": "string",
            "description": "Name of the auditing firm"
        },
        "audit_opinion": {
            "type": "string",
            "description": "Type of audit opinion (unqualified, qualified, adverse, disclaimer)"
        }
    }
}

REMUNERATION_SCHEMA = {
    "type": "object",
    "properties": {
        "fiscal_year_end_date": {
            "type": "string",
            "description": "The fiscal year end date"
        },
        "elected_officials": {
            "type": "array",
            "description": "List of all elected officials and their compensation",
            "items": {
                "type": "object",
                "properties": {
                    "position": {
                        "type": "string",
                        "description": "Position (Chief, Councillor, etc.)"
                    },
                    "name": {
                        "type": "string",
                        "description": "Full name of the official"
                    },
                    "months_in_position": {
                        "type": "integer",
                        "description": "Number of months in position"
                    },
                    "salary": {
                        "type": "number",
                        "description": "Base salary"
                    },
                    "honorarium": {
                        "type": "number",
                        "description": "Honorarium payments"
                    },
                    "travel": {
                        "type": "number",
                        "description": "Travel expenses"
                    },
                    "contract": {
                        "type": "number",
                        "description": "Contract payments"
                    },
                    "other": {
                        "type": "number",
                        "description": "Other compensation"
                    },
                    "total": {
                        "type": "number",
                        "description": "Total remuneration"
                    }
                }
            }
        }
    }
}


# ============================================================================
# LlamaExtract REST API Client
# ============================================================================

class LlamaExtractClient:
    """REST API client for LlamaExtract"""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = LLAMA_API_BASE
        self.client = httpx.Client(timeout=300.0)
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        self.fs_agent_id = None
        self.rem_agent_id = None

    def _make_request(self, method: str, endpoint: str, **kwargs) -> Dict:
        """Make API request"""
        url = f"{self.base_url}{endpoint}"
        if 'headers' not in kwargs:
            kwargs['headers'] = self.headers

        response = self.client.request(method, url, **kwargs)

        if response.status_code >= 400:
            logger.error(f"API error: {response.status_code} - {response.text}")
            raise Exception(f"API error: {response.status_code} - {response.text}")

        return response.json() if response.text else {}

    def list_agents(self) -> List[Dict]:
        """List all extraction agents"""
        result = self._make_request("GET", "/extraction/extraction-agents")
        return result if isinstance(result, list) else result.get('items', [])

    def get_agent_by_name(self, name: str) -> Optional[Dict]:
        """Get agent by name"""
        try:
            return self._make_request("GET", f"/extraction/extraction-agents/by-name/{name}")
        except Exception:
            return None

    def create_agent(self, name: str, schema: Dict, extraction_mode: str = "MULTIMODAL") -> Dict:
        """Create a new extraction agent

        Args:
            name: Agent name
            schema: JSON schema for extraction
            extraction_mode: One of FAST, BALANCED, MULTIMODAL, PREMIUM
                - FAST: Simple documents with OCR, fastest
                - BALANCED: Good balance between speed and accuracy
                - MULTIMODAL: Default, for visually rich documents
                - PREMIUM: Highest accuracy for complex tables/financial statements
        """
        payload = {
            "name": name,
            "data_schema": schema,
            "config": {
                "extraction_target": "PER_DOC",
                "extraction_mode": extraction_mode
            }
        }
        return self._make_request("POST", "/extraction/extraction-agents", json=payload)

    def get_or_create_agent(self, name: str, schema: Dict, extraction_mode: str = "MULTIMODAL") -> str:
        """Get existing agent or create new one"""
        # Try to get by name first
        agent = self.get_agent_by_name(name)
        if agent and agent.get('id'):
            logger.info(f"Found existing agent: {name} -> {agent['id']}")
            return agent['id']

        # Create new agent with specified mode
        logger.info(f"Creating new agent: {name} (mode={extraction_mode})")
        result = self.create_agent(name, schema, extraction_mode)
        return result['id']

    def upload_file(self, file_path: str) -> str:
        """Upload a file and return file_id"""
        with open(file_path, 'rb') as f:
            file_content = f.read()

        filename = os.path.basename(file_path)
        url = f"{self.base_url}/files"

        files = {'upload_file': (filename, file_content, 'application/pdf')}

        response = self.client.post(
            url,
            headers={"Authorization": f"Bearer {self.api_key}"},
            files=files
        )

        if response.status_code >= 400:
            logger.error(f"Upload error: {response.status_code} - {response.text}")
            raise Exception(f"Upload failed: {response.status_code}")

        result = response.json()
        return result['id']

    def create_extraction_job(self, agent_id: str, file_id: str) -> str:
        """Create an extraction job and return job_id"""
        payload = {
            "extraction_agent_id": agent_id,
            "file_id": file_id
        }
        result = self._make_request("POST", "/extraction/jobs", json=payload)
        return result['id']

    def get_job_status(self, job_id: str) -> Dict:
        """Get extraction job status"""
        return self._make_request("GET", f"/extraction/jobs/{job_id}")

    def get_job_result(self, job_id: str) -> Dict:
        """Get extraction job results"""
        return self._make_request("GET", f"/extraction/jobs/{job_id}/result")

    def wait_for_job(self, job_id: str, max_wait: int = 300) -> Dict:
        """Wait for job completion and return results

        Args:
            job_id: The extraction job ID
            max_wait: Maximum wait time in seconds (default 300s for complex docs)
        """
        start_time = time.time()
        poll_interval = 2
        while time.time() - start_time < max_wait:
            status = self.get_job_status(job_id)
            job_status = status.get('status', '').upper()

            if job_status == 'SUCCESS':
                return self.get_job_result(job_id)
            elif job_status in ('FAILED', 'ERROR', 'CANCELLED'):
                logger.error(f"Job {job_id} failed: {status}")
                return None
            elif job_status == 'PENDING':
                # Longer poll for pending jobs
                poll_interval = 3

            time.sleep(poll_interval)

        logger.error(f"Job {job_id} timed out after {max_wait}s")
        return None

    def extract_from_file(self, agent_id: str, file_path: str) -> Dict:
        """Extract data from a file using the full workflow"""
        try:
            # Step 1: Upload file
            logger.debug(f"Uploading file: {file_path}")
            file_id = self.upload_file(file_path)
            logger.debug(f"File uploaded: {file_id}")

            # Step 2: Create extraction job
            logger.debug(f"Creating extraction job...")
            job_id = self.create_extraction_job(agent_id, file_id)
            logger.debug(f"Job created: {job_id}")

            # Step 3: Wait for completion and get results
            result = self.wait_for_job(job_id)

            if result:
                # Extract the actual data from the result structure
                if isinstance(result, dict):
                    # Handle different result structures
                    if 'data' in result:
                        return {'data': result['data']}
                    elif 'extraction_result' in result:
                        return {'data': result['extraction_result']}
                    elif 'results' in result and len(result['results']) > 0:
                        return {'data': result['results'][0]}
                    else:
                        return {'data': result}
                return {'data': result}
            return None

        except Exception as e:
            logger.error(f"Extraction error for {file_path}: {e}")
            return None

    def setup_agents(self):
        """Setup financial statement and remuneration extraction agents

        Uses PREMIUM mode for financial statements (complex tables, highest accuracy)
        Uses MULTIMODAL mode for remuneration (simpler format, good balance)
        """
        logger.info("Setting up extraction agents...")

        # PREMIUM mode for financial statements - best for complex tables
        self.fs_agent_id = self.get_or_create_agent(
            "fnfta-financial-statements-v4-premium",
            FINANCIAL_STATEMENT_SCHEMA,
            extraction_mode="PREMIUM"
        )
        logger.info(f"Financial statement agent (PREMIUM): {self.fs_agent_id}")

        # MULTIMODAL for remuneration schedules - simpler format
        self.rem_agent_id = self.get_or_create_agent(
            "fnfta-remuneration-v4-multimodal",
            REMUNERATION_SCHEMA,
            extraction_mode="MULTIMODAL"
        )
        logger.info(f"Remuneration agent (MULTIMODAL): {self.rem_agent_id}")


# ============================================================================
# Database Operations
# ============================================================================

class FNFTADatabase:
    """SQLite database for FNFTA financial data"""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = None

    def connect(self):
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row

    def close(self):
        if self.conn:
            self.conn.close()

    def create_tables(self):
        """Create database tables"""
        self.connect()
        cursor = self.conn.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS financial_statements (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                band_number INTEGER NOT NULL,
                band_name TEXT NOT NULL,
                band_address TEXT,
                fiscal_year TEXT NOT NULL,
                fiscal_year_end_date TEXT,
                total_revenue REAL,
                revenue_isc REAL,
                revenue_health_canada REAL,
                revenue_provincial REAL,
                revenue_own_source REAL,
                total_expenditures REAL,
                exp_administration REAL,
                exp_education REAL,
                exp_health REAL,
                exp_social REAL,
                exp_housing REAL,
                net_revenue REAL,
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
                auditor_name TEXT,
                audit_opinion TEXT,
                pdf_path TEXT,
                extracted_at TEXT,
                UNIQUE(band_number, fiscal_year)
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS remuneration (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                band_number INTEGER NOT NULL,
                band_name TEXT NOT NULL,
                band_address TEXT,
                fiscal_year TEXT NOT NULL,
                position TEXT NOT NULL,
                name TEXT NOT NULL,
                months_in_position INTEGER,
                salary REAL,
                honorarium REAL,
                travel REAL,
                contract REAL,
                other REAL,
                total_remuneration REAL,
                pdf_path TEXT,
                extracted_at TEXT,
                UNIQUE(band_number, fiscal_year, name)
            )
        ''')

        cursor.execute('CREATE INDEX IF NOT EXISTS idx_fs_band ON financial_statements(band_number)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_fs_year ON financial_statements(fiscal_year)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_rem_band ON remuneration(band_number)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_rem_year ON remuneration(fiscal_year)')

        self.conn.commit()
        self.close()

    def insert_financial_statement(self, band_number: int, band_name: str, fiscal_year: str,
                                    data: Dict, pdf_path: str, band_address: str = None):
        """Insert a financial statement record"""
        self.connect()
        cursor = self.conn.cursor()

        cursor.execute('''
            INSERT OR REPLACE INTO financial_statements (
                band_number, band_name, band_address, fiscal_year, fiscal_year_end_date,
                total_revenue, revenue_isc, revenue_health_canada, revenue_provincial, revenue_own_source,
                total_expenditures, exp_administration, exp_education, exp_health, exp_social, exp_housing,
                net_revenue, cash_and_equivalents, accounts_receivable, total_financial_assets,
                tangible_capital_assets, accounts_payable, deferred_revenue, long_term_debt,
                total_liabilities, net_assets, accumulated_surplus, auditor_name, audit_opinion,
                pdf_path, extracted_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            band_number, band_name, band_address, fiscal_year,
            data.get('fiscal_year_end_date'),
            data.get('total_revenue'),
            data.get('revenue_indigenous_services_canada'),
            data.get('revenue_health_canada'),
            data.get('revenue_provincial'),
            data.get('revenue_own_source'),
            data.get('total_expenditures'),
            data.get('expenditure_administration'),
            data.get('expenditure_education'),
            data.get('expenditure_health'),
            data.get('expenditure_social'),
            data.get('expenditure_housing'),
            data.get('net_revenue'),
            data.get('cash_and_equivalents'),
            data.get('accounts_receivable'),
            data.get('total_financial_assets'),
            data.get('tangible_capital_assets'),
            data.get('accounts_payable'),
            data.get('deferred_revenue'),
            data.get('long_term_debt'),
            data.get('total_liabilities'),
            data.get('net_assets'),
            data.get('accumulated_surplus'),
            data.get('auditor_firm'),
            data.get('audit_opinion'),
            pdf_path,
            datetime.now().isoformat()
        ))

        self.conn.commit()
        self.close()

    def insert_remuneration(self, band_number: int, band_name: str, fiscal_year: str,
                            official: Dict, pdf_path: str, band_address: str = None):
        """Insert a remuneration record"""
        self.connect()
        cursor = self.conn.cursor()

        cursor.execute('''
            INSERT OR REPLACE INTO remuneration (
                band_number, band_name, band_address, fiscal_year,
                position, name, months_in_position,
                salary, honorarium, travel, contract, other, total_remuneration,
                pdf_path, extracted_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            band_number, band_name, band_address, fiscal_year,
            official.get('position', 'Unknown'),
            official.get('name', 'Unknown'),
            official.get('months_in_position'),
            official.get('salary'),
            official.get('honorarium'),
            official.get('travel'),
            official.get('contract'),
            official.get('other'),
            official.get('total'),
            pdf_path,
            datetime.now().isoformat()
        ))

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
        stats['unique_bands'] = cursor.fetchone()[0]

        self.close()
        return stats

    def has_financial_statement(self, band_number: int, fiscal_year: str) -> bool:
        """Check if a financial statement already exists for this band/year"""
        self.connect()
        cursor = self.conn.cursor()
        cursor.execute(
            'SELECT 1 FROM financial_statements WHERE band_number = ? AND fiscal_year = ?',
            (band_number, fiscal_year)
        )
        exists = cursor.fetchone() is not None
        self.close()
        return exists

    def has_remuneration(self, band_number: int, fiscal_year: str) -> bool:
        """Check if remuneration records already exist for this band/year"""
        self.connect()
        cursor = self.conn.cursor()
        cursor.execute(
            'SELECT 1 FROM remuneration WHERE band_number = ? AND fiscal_year = ?',
            (band_number, fiscal_year)
        )
        exists = cursor.fetchone() is not None
        self.close()
        return exists


# ============================================================================
# Main Extraction Logic
# ============================================================================

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


def extract_all(data_dir: str, db_path: str, limit: int = None,
                skip_remuneration: bool = False, resume: bool = True):
    """Extract all financial data from PDFs using LlamaExtract

    Args:
        data_dir: Directory containing metadata and documents
        db_path: Path to SQLite database
        limit: Maximum number of documents to process
        skip_remuneration: If True, only extract financial statements (saves ~50% API costs)
        resume: If True, skip documents already in database (default: True)
    """
    logger.info(f"Starting LlamaExtract extraction from {data_dir}")
    if skip_remuneration:
        logger.info("Skipping remuneration documents (--skip-remuneration enabled)")
    if resume:
        logger.info("Resume mode enabled - will skip already extracted documents")

    # Load metadata
    bands, documents = load_metadata(data_dir)
    logger.info(f"Loaded {len(bands)} bands and {len(documents)} documents")

    # Initialize database
    db = FNFTADatabase(db_path)
    db.create_tables()

    # Initialize LlamaExtract client
    client = LlamaExtractClient(LLAMA_API_KEY)
    client.setup_agents()

    # Process documents
    fs_count = 0
    rem_count = 0
    error_count = 0
    skipped_count = 0
    processed_count = 0

    for i, doc in enumerate(documents):
        if limit and processed_count >= limit:
            break

        if not doc.get('downloaded'):
            continue

        # Skip remuneration if requested
        if skip_remuneration and doc['doc_type'] == 'remuneration':
            continue

        pdf_path = os.path.join(data_dir, '..', doc['local_path'])
        if not os.path.exists(pdf_path):
            logger.warning(f"PDF not found: {pdf_path}")
            continue

        band_info = bands.get(doc['band_number'], {})
        band_address = band_info.get('address')

        # Check if already extracted (resume mode)
        if resume:
            if doc['doc_type'] == 'financial_statement':
                if db.has_financial_statement(doc['band_number'], doc['fiscal_year']):
                    skipped_count += 1
                    continue
            elif doc['doc_type'] == 'remuneration':
                if db.has_remuneration(doc['band_number'], doc['fiscal_year']):
                    skipped_count += 1
                    continue

        processed_count += 1

        try:
            if doc['doc_type'] == 'financial_statement':
                logger.info(f"[{i+1}] Extracting FS: {doc['band_name']} {doc['fiscal_year']}")

                result = client.extract_from_file(client.fs_agent_id, pdf_path)

                if result and result.get('data'):
                    db.insert_financial_statement(
                        band_number=doc['band_number'],
                        band_name=doc['band_name'],
                        fiscal_year=doc['fiscal_year'],
                        data=result['data'],
                        pdf_path=doc['local_path'],
                        band_address=band_address
                    )
                    fs_count += 1
                    logger.info(f"  -> Extracted: revenue={result['data'].get('total_revenue')}, expenses={result['data'].get('total_expenditures')}")
                else:
                    error_count += 1
                    logger.warning(f"  -> No data extracted")

            elif doc['doc_type'] == 'remuneration':
                logger.info(f"[{i+1}] Extracting REM: {doc['band_name']} {doc['fiscal_year']}")

                result = client.extract_from_file(client.rem_agent_id, pdf_path)

                if result and result.get('data'):
                    data = result['data']
                    logger.debug(f"  -> Remuneration response keys: {data.keys() if isinstance(data, dict) else 'not a dict'}")

                    # Handle different response structures
                    officials = data.get('elected_officials', [])
                    if not officials and isinstance(data, dict):
                        # Try alternative keys
                        for key in ['officials', 'compensation', 'remuneration', 'items']:
                            if key in data and isinstance(data[key], list):
                                officials = data[key]
                                break

                    if officials:
                        for official in officials:
                            db.insert_remuneration(
                                band_number=doc['band_number'],
                                band_name=doc['band_name'],
                                fiscal_year=doc['fiscal_year'],
                                official=official,
                                pdf_path=doc['local_path'],
                                band_address=band_address
                            )
                            rem_count += 1
                        logger.info(f"  -> Extracted {len(officials)} officials")
                    else:
                        error_count += 1
                        logger.warning(f"  -> No officials found in data: {json.dumps(data, indent=2)[:500]}")
                else:
                    error_count += 1
                    logger.warning(f"  -> No data extracted from result: {result}")

            # Rate limiting
            time.sleep(0.5)

        except Exception as e:
            logger.error(f"Error processing {pdf_path}: {e}")
            error_count += 1

        if processed_count % 10 == 0:
            stats = db.get_stats()
            logger.info(f"Progress: {processed_count} processed | FS: {fs_count} | REM: {rem_count} | Skipped: {skipped_count} | Errors: {error_count}")

    # Final stats
    stats = db.get_stats()
    logger.info("=" * 50)
    logger.info("EXTRACTION COMPLETE")
    logger.info(f"Financial statements extracted: {fs_count}")
    logger.info(f"Remuneration records extracted: {rem_count}")
    logger.info(f"Documents skipped (already in DB): {skipped_count}")
    logger.info(f"Errors: {error_count}")
    logger.info(f"Database stats: {stats}")

    return stats


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Extract FNFTA financial data using LlamaExtract')
    parser.add_argument('--data-dir', default='fnfta_data', help='Data directory')
    parser.add_argument('--db-path', default='fnfta_llama.db', help='SQLite database path')
    parser.add_argument('--limit', type=int, help='Limit number of documents to process')
    parser.add_argument('--skip-remuneration', action='store_true',
                        help='Skip remuneration documents (only extract financial statements)')
    parser.add_argument('--no-resume', action='store_true',
                        help='Disable resume mode (reprocess all documents)')

    args = parser.parse_args()

    extract_all(
        args.data_dir,
        args.db_path,
        args.limit,
        skip_remuneration=args.skip_remuneration,
        resume=not args.no_resume
    )
