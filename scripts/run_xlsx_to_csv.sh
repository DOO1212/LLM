#!/bin/bash
# data 폴더의 xlsx → csv 변환 (cron 등에서 호출용)
cd "$(dirname "$0")/.." && python3 scripts/xlsx_to_csv.py
