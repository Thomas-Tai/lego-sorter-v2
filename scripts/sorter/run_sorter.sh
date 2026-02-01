#!/bin/bash
# Lego Sorter - Pi-Side Run Script
# This script loads environment configuration and runs the sorter app.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

echo "=== Lego Sorter Pi-Side ==="
echo "Project Root: $PROJECT_ROOT"
echo ""

# Load environment configuration
ENV_FILE="$SCRIPT_DIR/sorter.env"
if [ -f "$ENV_FILE" ]; then
    echo "Loading config from: $ENV_FILE"
    set -a  # Auto-export variables
    source "$ENV_FILE"
    set +a
else
    echo "WARNING: $ENV_FILE not found!"
    echo "Please copy sorter.env.template to sorter.env and configure your PC's IP."
    echo ""
    echo "Using defaults:"
    export LEGO_API_URL="${LEGO_API_URL:-http://localhost:8000}"
    export LEGO_CAMERA_INDEX="${LEGO_CAMERA_INDEX:-0}"
fi

echo "  LEGO_API_URL: $LEGO_API_URL"
echo "  LEGO_CAMERA_INDEX: $LEGO_CAMERA_INDEX"
echo ""

# Check if API is reachable
echo "Checking API connectivity..."
if curl -s --connect-timeout 5 "$LEGO_API_URL/v1/health" > /dev/null 2>&1; then
    echo "  API is reachable!"
else
    echo "  WARNING: Cannot reach API at $LEGO_API_URL"
    echo "  Make sure the PC Inference API is running and firewall allows port 8000."
    echo ""
fi

# Run the sorter app
echo ""
echo "Starting Sorter App..."
echo "----------------------------------------"
cd "$PROJECT_ROOT"

# Use venv if available
if [ -f "$PROJECT_ROOT/venv/bin/python" ]; then
    "$PROJECT_ROOT/venv/bin/python" -m sorter_app.main "$@"
else
    python -m sorter_app.main "$@"
fi
