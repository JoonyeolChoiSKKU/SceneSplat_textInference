#!/bin/bash
# Install local packages that require torch to be already installed
# This script should be run after conda env create -f env.yaml

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Installing local packages (pointops and pointgroup_ops)..."
echo "These packages require torch to be already installed in the conda environment."

# Install pointops
echo "Installing pointops..."
pip install -e ./libs/pointops

# Install pointgroup_ops
echo "Installing pointgroup_ops..."
pip install -e ./libs/pointgroup_ops

echo "Local packages installation completed!"

