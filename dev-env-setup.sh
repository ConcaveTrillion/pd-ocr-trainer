#!/bin/bash

rm -rf .venv
python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip

# need the full doctr package, not included with pip install

cd ..
git clone https://github.com/mindee/doctr.git doctr
cd doctr
pip install -e .[torch]
cd ..
cd train_pgdp_ocr

# use pd_book_tools from local install as well

pip install -e ../pd_book_tools

pip install -r requirements.txt
