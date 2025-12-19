#!/usr/bin/env python3
"""
Create SQLite database for Canadian First Nations bands with population data.

Combines:
- First Nations Location data (from Indigenous Services Canada)
- Population Registered under the Indian Act by Gender and Residence, 2022

Data sources:
- https://open.canada.ca/data/en/dataset/b6567c5c-8339-4055-99fa-63f92114d9e4 (First Nations Location)
- https://open.canada.ca/data/en/dataset/6a493874-853b-4dbf-869d-22544fec79ec (Population 2022)
"""

import csv
import sqlite3
from pathlib import Path


def create_database(db_path: str = "first_nations.db"):
    """Create the First Nations SQLite database."""

    data_dir = Path(__file__).parent / "first_nations_data"
    location_file = data_dir / "Premiere_Nation_First_Nation.csv"
    population_file = data_dir / "population_2022.csv"

    # Create database connection
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Create tables
    cursor.executescript("""
        -- Drop existing tables if they exist
        DROP TABLE IF EXISTS population;
        DROP TABLE IF EXISTS bands;
        DROP TABLE IF EXISTS regions;
        DROP TABLE IF EXISTS provinces;

        -- Provinces/Territories lookup table
        CREATE TABLE provinces (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            code TEXT UNIQUE NOT NULL,
            name TEXT NOT NULL
        );

        -- Regions lookup table
        CREATE TABLE regions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL
        );

        -- Main bands table
        CREATE TABLE bands (
            band_number INTEGER PRIMARY KEY,
            band_name TEXT NOT NULL,
            region_id INTEGER,
            province_id INTEGER,
            district TEXT,
            longitude REAL,
            latitude REAL,
            FOREIGN KEY (region_id) REFERENCES regions(id),
            FOREIGN KEY (province_id) REFERENCES provinces(id)
        );

        -- Population data table (allows for multiple years)
        CREATE TABLE population (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            band_number INTEGER NOT NULL,
            year INTEGER NOT NULL,
            on_reserve_men INTEGER,
            on_reserve_women INTEGER,
            on_reserve_total INTEGER,
            off_reserve_men INTEGER,
            off_reserve_women INTEGER,
            off_reserve_total INTEGER,
            total_men INTEGER,
            total_women INTEGER,
            total_population INTEGER,
            FOREIGN KEY (band_number) REFERENCES bands(band_number),
            UNIQUE(band_number, year)
        );

        -- Create indexes for common queries
        CREATE INDEX idx_bands_region ON bands(region_id);
        CREATE INDEX idx_bands_province ON bands(province_id);
        CREATE INDEX idx_population_year ON population(year);
        CREATE INDEX idx_population_band ON population(band_number);
    """)

    # Province mappings
    province_map = {
        "Alberta": "AB",
        "British Columbia": "BC",
        "Manitoba": "MB",
        "New Brunswick": "NB",
        "Newfoundland and Labrador": "NL",
        "Northwest Territories": "NT",
        "Nova Scotia": "NS",
        "Nunavut": "NU",
        "Ontario": "ON",
        "Prince Edward Island": "PE",
        "Quebec": "QC",
        "Saskatchewan": "SK",
        "Yukon": "YT",
    }

    # Insert provinces
    for name, code in province_map.items():
        cursor.execute("INSERT INTO provinces (code, name) VALUES (?, ?)", (code, name))

    # Load location data into a dict for lookup
    location_data = {}
    with open(location_file, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            band_num = int(row['BAND_NUMBER'])
            location_data[band_num] = {
                'name': row['BAND_NAME'],
                'longitude': float(row['LONGITUDE']) if row['LONGITUDE'] else None,
                'latitude': float(row['LATITUDE']) if row['LATITUDE'] else None,
            }

    print(f"Loaded {len(location_data)} bands from location file")

    # Process population data
    regions_inserted = set()
    bands_inserted = set()
    population_count = 0

    with open(population_file, 'r', encoding='latin-1') as f:
        reader = csv.DictReader(f)

        for row in reader:
            band_num = int(row['registry_group_number'])
            region_name = row['region']
            province_name = row['prov_terr']

            # Insert region if not exists
            if region_name and region_name not in regions_inserted:
                cursor.execute("INSERT OR IGNORE INTO regions (name) VALUES (?)", (region_name,))
                regions_inserted.add(region_name)

            # Get region_id
            cursor.execute("SELECT id FROM regions WHERE name = ?", (region_name,))
            region_result = cursor.fetchone()
            region_id = region_result[0] if region_result else None

            # Get province_id
            cursor.execute("SELECT id FROM provinces WHERE name = ?", (province_name,))
            province_result = cursor.fetchone()
            province_id = province_result[0] if province_result else None

            # Insert band if not exists
            if band_num not in bands_inserted:
                # Get location data if available
                loc = location_data.get(band_num, {})

                # Use name from population file, fall back to location file
                band_name = row['registry_group_name']
                if not band_name and loc:
                    band_name = loc.get('name', f'Band {band_num}')

                cursor.execute("""
                    INSERT OR REPLACE INTO bands
                    (band_number, band_name, region_id, province_id, district, longitude, latitude)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    band_num,
                    band_name,
                    region_id,
                    province_id,
                    row['district'] if row['district'] != 'NA' else None,
                    loc.get('longitude'),
                    loc.get('latitude'),
                ))
                bands_inserted.add(band_num)

            # Insert population data
            def safe_int(val):
                try:
                    return int(val) if val and val != 'NA' else None
                except ValueError:
                    return None

            cursor.execute("""
                INSERT OR REPLACE INTO population
                (band_number, year, on_reserve_men, on_reserve_women, on_reserve_total,
                 off_reserve_men, off_reserve_women, off_reserve_total,
                 total_men, total_women, total_population)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                band_num,
                safe_int(row['year']),
                safe_int(row['pop_on_reserve_crown_land_men']),
                safe_int(row['pop_on_reserve_crown_land_women']),
                safe_int(row['pop_on_reserve_crown_land_total']),
                safe_int(row['pop_off_reserve_men']),
                safe_int(row['pop_off_reserve_women']),
                safe_int(row['pop_off_reserve_total']),
                safe_int(row['pop_total_men']),
                safe_int(row['pop_total_women']),
                safe_int(row['pop_total_total']),
            ))
            population_count += 1

    # Add bands from location file that aren't in population file
    for band_num, loc in location_data.items():
        if band_num not in bands_inserted:
            cursor.execute("""
                INSERT OR IGNORE INTO bands
                (band_number, band_name, longitude, latitude)
                VALUES (?, ?, ?, ?)
            """, (
                band_num,
                loc['name'],
                loc['longitude'],
                loc['latitude'],
            ))
            bands_inserted.add(band_num)

    conn.commit()

    # Print summary statistics
    cursor.execute("SELECT COUNT(*) FROM bands")
    total_bands = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM population")
    total_pop_records = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM regions")
    total_regions = cursor.fetchone()[0]

    cursor.execute("SELECT SUM(on_reserve_total), SUM(off_reserve_total), SUM(total_population) FROM population WHERE year = 2022")
    pop_sums = cursor.fetchone()

    print(f"\n{'='*60}")
    print("DATABASE CREATED SUCCESSFULLY")
    print(f"{'='*60}")
    print(f"Database file: {db_path}")
    print(f"Total bands: {total_bands}")
    print(f"Total regions: {total_regions}")
    print(f"Population records: {total_pop_records}")
    print(f"\n2022 Population Summary:")
    print(f"  On-reserve population: {pop_sums[0]:,}")
    print(f"  Off-reserve population: {pop_sums[1]:,}")
    print(f"  Total registered population: {pop_sums[2]:,}")

    # Show sample data
    print(f"\n{'='*60}")
    print("SAMPLE DATA (Top 10 bands by population)")
    print(f"{'='*60}")

    cursor.execute("""
        SELECT b.band_number, b.band_name, p.name as province,
               pop.on_reserve_total, pop.off_reserve_total, pop.total_population
        FROM bands b
        JOIN population pop ON b.band_number = pop.band_number
        LEFT JOIN provinces p ON b.province_id = p.id
        WHERE pop.year = 2022
        ORDER BY pop.total_population DESC
        LIMIT 10
    """)

    print(f"{'Band #':<8} {'Band Name':<35} {'Province':<15} {'On-Res':<10} {'Off-Res':<10} {'Total':<10}")
    print("-" * 98)

    for row in cursor.fetchall():
        print(f"{row[0]:<8} {row[1][:34]:<35} {row[2] or 'N/A':<15} {row[3] or 0:<10,} {row[4] or 0:<10,} {row[5] or 0:<10,}")

    # Show by region
    print(f"\n{'='*60}")
    print("POPULATION BY REGION (2022)")
    print(f"{'='*60}")

    cursor.execute("""
        SELECT r.name, COUNT(DISTINCT b.band_number) as band_count,
               SUM(pop.on_reserve_total) as on_reserve,
               SUM(pop.off_reserve_total) as off_reserve,
               SUM(pop.total_population) as total
        FROM regions r
        JOIN bands b ON r.id = b.region_id
        JOIN population pop ON b.band_number = pop.band_number
        WHERE pop.year = 2022
        GROUP BY r.name
        ORDER BY total DESC
    """)

    print(f"{'Region':<20} {'Bands':<8} {'On-Reserve':<15} {'Off-Reserve':<15} {'Total':<15}")
    print("-" * 73)

    for row in cursor.fetchall():
        print(f"{row[0]:<20} {row[1]:<8} {row[2]:>12,}   {row[3]:>12,}   {row[4]:>12,}")

    conn.close()
    print(f"\n✓ Database saved to: {db_path}")


if __name__ == "__main__":
    create_database()
