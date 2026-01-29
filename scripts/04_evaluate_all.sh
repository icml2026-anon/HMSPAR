#!/bin/bash

echo "=========================================="
echo "Evaluating HMSPAR on All Industries"
echo "=========================================="

cd /root/HMSPAR

for industry in Industry-0 Industry-1 Industry-2 Industry-3
do
    echo ""
    echo "Evaluating $industry..."
    python evaluate.py --industry $industry
done

echo ""
echo "=========================================="
echo "All evaluations complete!"
echo "=========================================="

