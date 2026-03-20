#!/bin/bash
OUT="/home/doohyeon/chatbot/d2r_probe_out.txt"
D2R="/home/doohyeon/Desktop/d2r"
echo "=== ini grep ===" > "$OUT"
grep -niE 'join|slave|master|auto|window' "$D2R/d2r_settings.ini" 2>&1 >> "$OUT" || echo "grep fail" >> "$OUT"
echo "=== ini head ===" >> "$OUT"
head -80 "$D2R/d2r_settings.ini" >> "$OUT" 2>&1
echo "=== dat strings sample ===" >> "$OUT"
strings "$D2R/default.dat" 2>/dev/null | head -100 >> "$OUT"
echo "done" >> "$OUT"
