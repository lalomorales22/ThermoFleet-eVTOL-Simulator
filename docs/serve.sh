#!/bin/bash

# ThermoFleet Documentation Server
# Serves the documentation site on http://localhost:8000

echo "🚀 Starting ThermoFleet Documentation Server..."
echo ""
echo "📚 Documentation will be available at:"
echo "   👉 http://localhost:8000"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Change to docs directory
cd "$(dirname "$0")"

# Start Python HTTP server
python3 -m http.server 8000

