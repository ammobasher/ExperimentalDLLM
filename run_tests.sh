#!/bin/bash
# Enable immediate exit on error
set -e

# Add current directory to PYTHONPATH so 'src' module can be found
export PYTHONPATH=$PYTHONPATH:.

echo "========================================"
echo "    Running Project Synapse Tests"
echo "========================================"

echo ""
echo "[1/3] Verifying Core Gradients (PCLayer, DiffusionSDE)..."
./venv/bin/python3 tests/verify_gradients.py
echo "✅ Gradient Checks Passed"

echo ""
echo "[2/3] Verifying PCModel Architecture (End-to-End)..."
./venv/bin/python3 tests/verify_model.py
echo "✅ Model Architecture Passed"

echo ""
echo "[3/3] Verifying Training Loop (Optax Integration)..."
./venv/bin/python3 src/train_diffusion.py
echo "✅ Training Loop Passed"

echo ""
echo "[4/4] Verifying Memory System (eLTM + Sleep)..."
./venv/bin/python3 tests/verify_memory.py
echo "✅ Memory System Passed"

echo ""
echo "========================================"
echo "🎉 ALL TESTS PASSED SUCCESSFULLY"
echo "========================================"
