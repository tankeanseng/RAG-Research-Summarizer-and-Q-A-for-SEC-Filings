#!/usr/bin/env bash
set -euo pipefail

rm -rf build package.zip
mkdir -p build

python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt -t build

cp handler.py build/

cd build
zip -r ../package.zip .
cd ..
echo "Built package.zip"
ls -lh package.zip