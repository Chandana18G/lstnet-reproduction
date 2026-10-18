#!/usr/bin/env bash
# Download the Electricity and Solar-Energy datasets used by LSTNet (Lai et al., 2018)
# from the authors' data repository into data/. Run from the repo root:
#   bash scripts/download_data.sh
set -euo pipefail

BASE="https://raw.githubusercontent.com/laiguokun/multivariate-time-series-data/master"
mkdir -p data

download() {
  local url="$1" out="$2"
  if [ -f "$out" ]; then
    echo "already there: $out"
  else
    echo "downloading $out"
    curl -fL --retry 3 -o "$out" "$url"
  fi
}

download "$BASE/electricity/electricity.txt.gz" data/electricity.txt.gz
download "$BASE/solar-energy/solar_AL.txt.gz"   data/solar_AL.txt.gz

echo "done:"
ls -lh data
