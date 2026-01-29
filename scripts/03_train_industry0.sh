#!/bin/bash

echo "=========================================="
echo "Training HMSPAR on Industry-0"
echo "=========================================="

cd /root/HMSPAR
python train.py --industry Industry-0

echo ""
echo "Training complete for Industry-0!"

