#!/usr/bin/env bash
# Download the datasets of the LSTNet paper into the folder data/.
# Run from the repo root:
#   bash scripts/download_data.sh               -> both datasets
#   bash scripts/download_data.sh electricity   -> only Electricity
#   bash scripts/download_data.sh solar         -> only Solar-Energy

URL="https://raw.githubusercontent.com/laiguokun/multivariate-time-series-data/master"
mkdir -p data

if [ "$1" != "solar" ]; then
  curl -L -o data/electricity.txt.gz "$URL/electricity/electricity.txt.gz"
fi

if [ "$1" != "electricity" ]; then
  curl -L -o data/solar_AL.txt.gz "$URL/solar-energy/solar_AL.txt.gz"
fi
