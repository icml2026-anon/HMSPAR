#!/bin/bash

echo "=========================================="
echo "Training HMSPAR on Industry-2"
echo "=========================================="

cd /root/HMSPAR
python train.py --industry Industry-2

echo ""
echo "Training complete for Industry-2!"

