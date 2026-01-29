#!/bin/bash

echo "=========================================="
echo "Training HMSPAR on Industry-1"
echo "=========================================="

cd /root/HMSPAR
python train.py --industry Industry-1

echo ""
echo "Training complete for Industry-1!"

