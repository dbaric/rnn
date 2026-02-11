#!/usr/bin/env python3
"""
Download CROBEX index history from ZSE (Zagreb Stock Exchange) API.
Loops from 2010-02-01 to today and saves CSV data to a file.
"""

import requests
from datetime import date, timedelta
from pathlib import Path

BASE_URL = (
    "https://rest.zse.hr/web/Bvt9fe2peQ7pwpyYqODM/index-history"
    "/XZAG/HRZB00ICBE11/{start_date}/{end_date}/csv?language=HR"
)
START_DATE = date(2010, 2, 1)
CHUNK_DAYS = 31
OUTPUT_FILE = Path(__file__).parent / "crobex_history.csv"


def main():
    end_date = date.today()
    all_lines = []
    header = None
    chunk_start = START_DATE

    while chunk_start <= end_date:
        chunk_end = min(chunk_start + timedelta(days=CHUNK_DAYS - 1), end_date)
        url = BASE_URL.format(
            start_date=chunk_start.strftime("%Y-%m-%d"),
            end_date=chunk_end.strftime("%Y-%m-%d"),
        )
        print(f"Fetching {chunk_start} .. {chunk_end} ...")
        try:
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            text = r.text.strip()
            if not text:
                chunk_start = chunk_end + timedelta(days=1)
                continue
            lines = text.splitlines()
            if not lines:
                chunk_start = chunk_end + timedelta(days=1)
                continue
            if header is None:
                header = lines[0]
                all_lines.append(header)
            start_idx = 1 if lines[0] == header else 0
            for line in lines[start_idx:]:
                if line.strip():
                    all_lines.append(line)
        except requests.RequestException as e:
            print(f"  Error: {e}")
        chunk_start = chunk_end + timedelta(days=1)

    if all_lines:
        OUTPUT_FILE.write_text("\n".join(all_lines) + "\n", encoding="utf-8")
        print(f"Saved {len(all_lines) - 1} rows to {OUTPUT_FILE}")
    else:
        print("No data downloaded.")


if __name__ == "__main__":
    main()
