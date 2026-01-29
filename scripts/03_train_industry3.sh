#!/bin/bash

echo "=========================================="
echo "Training HMSPAR on Industry-3"
echo "=========================================="

cd /root/HMSPAR
python train.py --industry Industry-3

echo ""
echo "Training complete for Industry-3!"

