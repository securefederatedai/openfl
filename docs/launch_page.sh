#!/bin/bash

# Command sequence
make clean
make html
cd _build/html
python -m http.server 8081 --bind 0.0.0.0

