#!/usr/bin/env bash
# Download PTB-XL v1.0.3 from PhysioNet into data/ptb-xl/ (skipped if already present).
set -euo pipefail

cd "$(dirname "$0")/.."
URL="https://physionet.org/static/published-projects/ptb-xl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3.zip"
DEST="data/ptb-xl"

if [ -f "$DEST/ptbxl_database.csv" ]; then
    echo "PTB-XL already present in $DEST"
    exit 0
fi

mkdir -p data
curl -L --fail -o data/ptb-xl.zip "$URL"
unzip -q data/ptb-xl.zip -d data/
mv data/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3 "$DEST"
rm data/ptb-xl.zip
echo "PTB-XL downloaded to $DEST"
