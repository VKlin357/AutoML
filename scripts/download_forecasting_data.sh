#!/usr/bin/env bash
# Download forecasting datasets (ETTh1, ETTh2, Weather) to cache
# Run on the GPU server before experiments

set -euo pipefail

CACHE_DIR="${HOME}/.cache/nas_datasets/forecasting"
mkdir -p "$CACHE_DIR"

echo "Downloading forecasting datasets to $CACHE_DIR ..."

BASE="https://raw.githubusercontent.com/thuml/Time-Series-Library/main/dataset"

datasets=(
    "ETT-small/ETTh1.csv:etth1.csv"
    "ETT-small/ETTh2.csv:etth2.csv"
    "ETT-small/ETTm1.csv:ettm1.csv"
    "ETT-small/ETTm2.csv:ettm2.csv"
    "weather/weather.csv:weather.csv"
    "exchange_rate/exchange_rate.csv:exchange.csv"
)

for entry in "${datasets[@]}"; do
    src="${entry%%:*}"
    dst="${entry##*:}"
    out="$CACHE_DIR/$dst"
    if [ -f "$out" ]; then
        echo "  [skip] $dst already exists ($(wc -l < "$out") lines)"
    else
        echo "  Downloading $dst ..."
        wget -q "$BASE/$src" -O "$out"
        echo "  [ok] $dst ($(wc -l < "$out") lines)"
    fi
done

echo ""
echo "Done. Files in $CACHE_DIR:"
ls -lh "$CACHE_DIR"/*.csv
