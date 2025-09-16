#!/bin/bash

# mcp-coaia-sequential-thinking Release Script
# Prepares distribution and publishes to PyPI

set -e  # Exit on any error

echo "🚀 CoaiaPy Release Script Starting..."

# Ensure all dependencies are installed
echo "🛠️ Installing/updating dependencies..."
pip install -r requirements.txt

# Check for 'build' and 'twine' and install if missing
if ! python -c "import build" &> /dev/null; then
    echo "Installing 'build' module..."
    pip install build
fi
if ! python -c "import twine" &> /dev/null; then
    echo "Installing 'twine' module..."
    pip install twine
fi

# Clean previous builds
echo "🧹 Cleaning previous builds..."
make clean

# Bump version
echo "📈 Bumping version..."
make bump

# Build distribution
echo "🔨 Building distribution..."
make build

# Upload to PyPI
echo "📦 Publishing to PyPI..."
twine upload dist/*

# Get current version and create git tag
echo "🏷️ Creating git tag..."
VERSION=$(grep -E "^version = \"" pyproject.toml | sed -E "s/.*\"([^\"]+)\".*/\1/")
git add pyproject.toml
git commit -m "v${VERSION}" || echo "No changes to commit"
git tag "v${VERSION}"

echo "✅ Release complete!"
echo "📋 Version: v${VERSION}"
echo "📋 Next steps:"
echo "   - Push changes: git push origin main"
echo "   - Push tag: git push origin v${VERSION}"
echo "   - Verify package on PyPI: https://pypi.org/project/mcp-coaia-sequential-thinking/"
echo "   - Test installation: pip install mcp-coaia-sequential-thinking"