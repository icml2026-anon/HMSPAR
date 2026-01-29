#!/bin/bash

echo "=========================================="
echo "Step 2: Converting Modalities"
echo "=========================================="

cd /root/HMSPAR
python data/modality_converter.py

echo ""
echo "Modality conversion complete!"

