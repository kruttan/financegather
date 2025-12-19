#!/usr/bin/env python3
"""
First Nations Financial Transparency Act (FNFTA) Data Scraper

Scrapes financial statements and remuneration schedules from:
https://fnp-ppn.aadnc-aandc.gc.ca/fnp/Main/Search/SearchFF.aspx

Downloads documents for the last 5 fiscal years for all First Nations bands.
"""

import os
import re
import json
import time
import logging
import argparse
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional
from urllib.parse import urljoin, quote

import requests
from bs4 import BeautifulSoup

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

BASE_URL = "https://fnp-ppn.aadnc-aandc.gc.ca/fnp/Main/Search/"
SEARCH_URL = BASE_URL + "SearchFF.aspx?lang=eng"
LIST_URL = BASE_URL + "FFListGrid.aspx?lang=eng"
BAND_URL = BASE_URL + "FederalFundingMain.aspx?BAND_NUMBER={band_number}&lang=eng"
DOCUMENT_URL = BASE_URL + "DisplayBinaryData.aspx?BAND_NUMBER_FF={band_number}&FY={fiscal_year}&DOC={doc_name}&lang=eng"

# Last 5 fiscal years (Canadian fiscal year: April 1 - March 31)
TARGET_FISCAL_YEARS = [
    "2023-2024",
    "2022-2023",
    "2021-2022",
    "2020-2021",
    "2019-2020",
]


@dataclass
class Band:
    """Represents a First Nation band."""
    number: int
    name: str
    address: Optional[str] = None
    province: Optional[str] = None


@dataclass
class Document:
    """Represents a financial document."""
    band_number: int
    band_name: str
    fiscal_year: str
    doc_type: str  # 'financial_statement' or 'remuneration'
    doc_name: str
    date_received: Optional[str] = None
    url: Optional[str] = None
    downloaded: bool = False
    local_path: Optional[str] = None


class FNFTAScraper:
    """Scraper for First Nations Financial Transparency Act data."""

    def __init__(self, output_dir: str = "fnfta_data", delay: float = 1.0):
        self.output_dir = Path(output_dir)
        self.delay = delay  # Delay between requests to be respectful
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (compatible; FNFTA-Research-Scraper/1.0)',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
        })

        # Create output directories
        self.docs_dir = self.output_dir / "documents"
        self.metadata_dir = self.output_dir / "metadata"
        self.docs_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)

    def _get_viewstate(self, soup: BeautifulSoup) -> dict:
        """Extract ASP.NET viewstate fields from page."""
        viewstate = {}
        for field in ['__VIEWSTATE', '__VIEWSTATEGENERATOR', '__EVENTVALIDATION', '__EVENTTARGET', '__EVENTARGUMENT']:
            elem = soup.find('input', {'name': field})
            if elem:
                viewstate[field] = elem.get('value', '')
        return viewstate

    def _respectful_delay(self):
        """Add delay between requests to be respectful to the server."""
        time.sleep(self.delay)

    def get_all_bands(self) -> list[Band]:
        """Get all First Nations bands from the FNFTA search."""
        bands = []

        # First, get the search page to obtain viewstate
        logger.info("Fetching search page...")
        resp = self.session.get(SEARCH_URL)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, 'html.parser')
        viewstate = self._get_viewstate(soup)

        # Iterate through each letter A-Z
        for letter in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
            logger.info(f"Fetching bands starting with '{letter}'...")
            self._respectful_delay()

            # Make POST request to trigger letter filter
            data = {
                **viewstate,
                '__EVENTTARGET': f'ctl00$plcMain$lb{letter}',
                '__EVENTARGUMENT': '',
            }

            resp = self.session.post(SEARCH_URL, data=data)
            resp.raise_for_status()

            # Parse the results page
            soup = BeautifulSoup(resp.text, 'html.parser')

            # Check if we got redirected to the list page
            if 'FFListGrid' in resp.url:
                # We need to handle pagination
                bands.extend(self._parse_band_list_page(soup))

                # Check for pagination and get all pages
                while True:
                    next_link = soup.find('div', class_='dataTables_paginate')
                    if not next_link:
                        break

                    next_btn = next_link.find('a', string='Next')
                    if not next_btn or 'disabled' in next_btn.get('class', []):
                        break

                    # For DataTables pagination, we need to use JavaScript
                    # Since this is server-side, let's try getting all entries
                    break
            else:
                # Results might be on same page or empty
                bands.extend(self._parse_band_list_page(soup))

        # Deduplicate bands by number
        seen = set()
        unique_bands = []
        for band in bands:
            if band.number not in seen:
                seen.add(band.number)
                unique_bands.append(band)

        logger.info(f"Found {len(unique_bands)} unique bands")
        return unique_bands

    def get_all_bands_by_province(self) -> list[Band]:
        """Get all bands by iterating through provinces."""
        bands = []
        provinces = [
            ('BC', 'British Columbia'),
            ('AB', 'Alberta'),
            ('SK', 'Saskatchewan'),
            ('MB', 'Manitoba'),
            ('ON', 'Ontario'),
            ('QC', 'Quebec'),
            ('NB', 'New Brunswick'),
            ('NS', 'Nova Scotia'),
            ('PE', 'Prince Edward Island'),
            ('NF', 'Newfoundland and Labrador'),
            ('NT', 'Northwest Territories'),
            ('YT', 'Yukon'),
        ]

        for prov_code, prov_name in provinces:
            logger.info(f"Fetching bands in {prov_name}...")

            # Get fresh search page for each province
            self._respectful_delay()
            resp = self.session.get(SEARCH_URL)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, 'html.parser')
            viewstate = self._get_viewstate(soup)

            self._respectful_delay()
            data = {
                **viewstate,
                '__EVENTTARGET': f'ctl00$plcMain$lb{prov_code}',
                '__EVENTARGUMENT': '',
            }

            resp = self.session.post(SEARCH_URL, data=data, allow_redirects=True)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, 'html.parser')

            province_bands = self._parse_band_list_page(soup, province=prov_name)
            bands.extend(province_bands)
            logger.info(f"  Found {len(province_bands)} bands in {prov_name}")

        # Deduplicate
        seen = set()
        unique_bands = []
        for band in bands:
            if band.number not in seen:
                seen.add(band.number)
                unique_bands.append(band)

        logger.info(f"Found {len(unique_bands)} unique bands total")
        return unique_bands

    def get_all_bands_by_letter(self) -> list[Band]:
        """Get all bands by iterating through letters A-Z."""
        bands = []

        for letter in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
            logger.info(f"Fetching bands starting with '{letter}'...")

            # Get fresh search page for each letter
            self._respectful_delay()
            resp = self.session.get(SEARCH_URL)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, 'html.parser')
            viewstate = self._get_viewstate(soup)

            self._respectful_delay()
            data = {
                **viewstate,
                '__EVENTTARGET': f'ctl00$plcMain$lb{letter}',
                '__EVENTARGUMENT': '',
            }

            resp = self.session.post(SEARCH_URL, data=data, allow_redirects=True)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, 'html.parser')

            letter_bands = self._parse_band_list_page(soup)
            bands.extend(letter_bands)
            logger.info(f"  Found {len(letter_bands)} bands starting with '{letter}'")

        # Deduplicate
        seen = set()
        unique_bands = []
        for band in bands:
            if band.number not in seen:
                seen.add(band.number)
                unique_bands.append(band)

        logger.info(f"Found {len(unique_bands)} unique bands total")
        return unique_bands

    def _parse_band_list_page(self, soup: BeautifulSoup, province: Optional[str] = None) -> list[Band]:
        """Parse the band list from a search results page."""
        bands = []

        # Find all rows in the data table
        # The table uses DataTables, look for the grid rows
        table = soup.find('table', {'id': lambda x: x and 'DataTables' in str(x)}) or \
                soup.find('div', {'class': 'dataTables_wrapper'})

        if not table:
            # Try finding rows directly
            rows = soup.find_all('tr')
        else:
            rows = table.find_all('tr')

        for row in rows:
            cells = row.find_all(['td', 'gridcell'])
            if len(cells) >= 2:
                # First cell should be band number, second is name
                number_cell = cells[0]
                name_cell = cells[1]

                # Extract band number
                number_link = number_cell.find('a')
                if number_link:
                    try:
                        band_number = int(number_link.get_text(strip=True))
                    except ValueError:
                        continue
                else:
                    try:
                        band_number = int(number_cell.get_text(strip=True))
                    except ValueError:
                        continue

                # Extract band name
                name_link = name_cell.find('a')
                if name_link:
                    band_name = name_link.get_text(strip=True)
                else:
                    band_name = name_cell.get_text(strip=True)

                if not band_name or band_name in ['Official Name', 'Band Number']:
                    continue

                # Extract address if available
                address = None
                if len(cells) >= 3:
                    address = cells[2].get_text(strip=True)
                    if address in ['Address', '']:
                        address = None

                bands.append(Band(
                    number=band_number,
                    name=band_name,
                    address=address,
                    province=province
                ))

        return bands

    def get_band_documents(self, band: Band) -> list[Document]:
        """Get all available documents for a band."""
        documents = []

        # Go directly to the main page (skip the note page)
        url = BAND_URL.format(band_number=band.number)
        logger.debug(f"Fetching documents for band {band.number} ({band.name})")

        try:
            resp = self.session.get(url)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, 'html.parser')

            # Now parse the documents table
            table = soup.find('table')
            if not table:
                logger.warning(f"No document table found for band {band.number}")
                return documents

            rows = table.find_all('tr')
            for row in rows:
                cells = row.find_all('td')
                if len(cells) < 3:
                    continue

                fiscal_year = cells[0].get_text(strip=True)

                # Only process target fiscal years
                if fiscal_year not in TARGET_FISCAL_YEARS:
                    continue

                doc_cell = cells[1]
                date_received = cells[2].get_text(strip=True)

                # Skip if not yet posted
                if 'Not yet posted' in date_received:
                    continue

                # Get document link and name
                doc_link = doc_cell.find('a')
                if not doc_link:
                    continue

                doc_name = doc_link.get_text(strip=True)
                href = doc_link.get('href', '')

                # Skip placeholder links
                if href == '#' or not href:
                    continue

                # Determine document type
                doc_type = 'financial_statement'
                if 'remuneration' in doc_name.lower() or 'expenses' in doc_name.lower():
                    doc_type = 'remuneration'

                # Construct full URL
                if href.startswith('http'):
                    doc_url = href
                else:
                    doc_url = urljoin(BASE_URL, href)

                documents.append(Document(
                    band_number=band.number,
                    band_name=band.name,
                    fiscal_year=fiscal_year,
                    doc_type=doc_type,
                    doc_name=doc_name,
                    date_received=date_received,
                    url=doc_url,
                ))

        except Exception as e:
            logger.error(f"Error fetching documents for band {band.number}: {e}")

        return documents

    def download_document(self, doc: Document) -> bool:
        """Download a document and save it locally."""
        if not doc.url:
            return False

        # Create directory structure: documents/{band_number}_{band_name}/{fiscal_year}/
        safe_name = re.sub(r'[^\w\s-]', '', doc.band_name).strip().replace(' ', '_')
        band_dir = self.docs_dir / f"{doc.band_number}_{safe_name}" / doc.fiscal_year
        band_dir.mkdir(parents=True, exist_ok=True)

        # Create filename
        safe_doc_name = re.sub(r'[^\w\s-]', '', doc.doc_name).strip().replace(' ', '_')
        filename = f"{safe_doc_name}.pdf"
        filepath = band_dir / filename

        # Skip if already downloaded
        if filepath.exists():
            doc.downloaded = True
            doc.local_path = str(filepath)
            logger.debug(f"Already exists: {filepath}")
            return True

        try:
            logger.info(f"Downloading: {doc.doc_name} ({doc.fiscal_year})")
            self._respectful_delay()

            resp = self.session.get(doc.url, stream=True, timeout=60)
            resp.raise_for_status()

            # Check content type
            content_type = resp.headers.get('Content-Type', '')
            if 'pdf' not in content_type.lower() and 'octet-stream' not in content_type.lower():
                logger.warning(f"Unexpected content type for {doc.doc_name}: {content_type}")

            # Write file
            with open(filepath, 'wb') as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)

            doc.downloaded = True
            doc.local_path = str(filepath)
            logger.info(f"Downloaded: {filepath}")
            return True

        except Exception as e:
            logger.error(f"Error downloading {doc.doc_name}: {e}")
            return False

    def save_metadata(self, bands: list[Band], documents: list[Document]):
        """Save metadata to JSON files."""
        # Save bands
        bands_file = self.metadata_dir / "bands.json"
        with open(bands_file, 'w', encoding='utf-8') as f:
            json.dump([asdict(b) for b in bands], f, indent=2, ensure_ascii=False)
        logger.info(f"Saved bands metadata to {bands_file}")

        # Save documents
        docs_file = self.metadata_dir / "documents.json"
        with open(docs_file, 'w', encoding='utf-8') as f:
            json.dump([asdict(d) for d in documents], f, indent=2, ensure_ascii=False)
        logger.info(f"Saved documents metadata to {docs_file}")

        # Save summary
        summary = {
            'total_bands': len(bands),
            'total_documents': len(documents),
            'downloaded_documents': sum(1 for d in documents if d.downloaded),
            'fiscal_years': TARGET_FISCAL_YEARS,
            'by_year': {},
            'by_type': {'financial_statement': 0, 'remuneration': 0},
        }

        for fy in TARGET_FISCAL_YEARS:
            summary['by_year'][fy] = sum(1 for d in documents if d.fiscal_year == fy)

        for doc in documents:
            summary['by_type'][doc.doc_type] += 1

        summary_file = self.metadata_dir / "summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Saved summary to {summary_file}")

    def run(self, download: bool = True, limit: Optional[int] = None, method: str = 'letter'):
        """Run the full scraping process.

        Args:
            download: Whether to download the PDF documents
            limit: Limit to first N bands (for testing)
            method: 'letter' to search A-Z, 'province' to search by province
        """
        logger.info("Starting FNFTA data collection...")
        logger.info(f"Target fiscal years: {TARGET_FISCAL_YEARS}")

        # Get all bands
        logger.info("Step 1: Getting list of all First Nations bands...")
        if method == 'province':
            bands = self.get_all_bands_by_province()
        else:
            bands = self.get_all_bands_by_letter()

        if limit:
            bands = bands[:limit]
            logger.info(f"Limited to first {limit} bands")

        # Get documents for each band
        logger.info("Step 2: Getting document information for each band...")
        all_documents = []

        for i, band in enumerate(bands, 1):
            logger.info(f"Processing band {i}/{len(bands)}: {band.name} (#{band.number})")
            self._respectful_delay()

            docs = self.get_band_documents(band)
            all_documents.extend(docs)
            logger.info(f"  Found {len(docs)} documents for last 5 years")

        logger.info(f"Total documents found: {len(all_documents)}")

        # Download documents
        if download:
            logger.info("Step 3: Downloading documents...")
            for i, doc in enumerate(all_documents, 1):
                logger.info(f"Downloading {i}/{len(all_documents)}: {doc.band_name} - {doc.doc_name}")
                self.download_document(doc)

        # Save metadata
        logger.info("Step 4: Saving metadata...")
        self.save_metadata(bands, all_documents)

        # Print summary
        downloaded = sum(1 for d in all_documents if d.downloaded)
        logger.info("=" * 60)
        logger.info("SCRAPING COMPLETE")
        logger.info(f"Total bands processed: {len(bands)}")
        logger.info(f"Total documents found: {len(all_documents)}")
        logger.info(f"Documents downloaded: {downloaded}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description='Scrape First Nations Financial Transparency Act data'
    )
    parser.add_argument(
        '-o', '--output',
        default='fnfta_data',
        help='Output directory (default: fnfta_data)'
    )
    parser.add_argument(
        '-d', '--delay',
        type=float,
        default=1.0,
        help='Delay between requests in seconds (default: 1.0)'
    )
    parser.add_argument(
        '--no-download',
        action='store_true',
        help='Only collect metadata, do not download documents'
    )
    parser.add_argument(
        '-l', '--limit',
        type=int,
        help='Limit to first N bands (for testing)'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    parser.add_argument(
        '-m', '--method',
        choices=['letter', 'province'],
        default='letter',
        help='Method to list bands: "letter" (A-Z) or "province" (default: letter)'
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    scraper = FNFTAScraper(
        output_dir=args.output,
        delay=args.delay
    )

    scraper.run(
        download=not args.no_download,
        limit=args.limit,
        method=args.method
    )


if __name__ == '__main__':
    main()
