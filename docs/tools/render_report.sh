#!/bin/bash
# Render REPORT.md to REPORT.pdf the way a reader would print it, and print the page count.
# US Letter (shorter than A4), 2 cm margins, 11 pt body, 9 pt tables. Needs pandoc and Google Chrome.
set -e
cd "$(dirname "$0")/../.."
CSS=$(mktemp -t report).css; HTML=$(mktemp -t report).html
cat > "$CSS" <<'C'
@page { size: Letter; margin: 2cm; }
body { font-family: Helvetica, Arial, sans-serif; font-size: 11pt; line-height: 1.3; max-width: none; margin: 0; }
h1 { font-size: 15pt; margin: 0 0 6pt; } h2 { font-size: 12.5pt; margin: 10pt 0 4pt; } p, li { margin: 3pt 0; }
ul, ol { margin: 3pt 0; padding-left: 16pt; }
table { border-collapse: collapse; font-size: 9pt; margin: 4pt 0; width: 100%; }
th, td { border: 0.5pt solid #999; padding: 1.5pt 4pt; text-align: left; vertical-align: top; }
code { font-size: 9.5pt; }
col { width: auto !important; } table { table-layout: auto; }
C
pandoc REPORT.md -s --metadata title=" " -c "$CSS" --embed-resources -o "$HTML" 2>/dev/null || pandoc REPORT.md -s --metadata title=" " -c "$CSS" -o "$HTML"
"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" --headless=new --disable-gpu --no-pdf-header-footer \
  --print-to-pdf=REPORT.pdf "file://$HTML" 2>/dev/null
pdfinfo REPORT.pdf | grep Pages
