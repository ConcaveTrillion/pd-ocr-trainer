#!/bin/bash

# need the full doctr package, not included with pip install

git clone https://github.com/mindee/doctr.git doctr_package
cd doctr_package
pip install -e .[torch]
