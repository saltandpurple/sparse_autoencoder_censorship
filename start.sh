#!/bin/bash

echo "Starting ChromaDB Content Visualizer..."

echo "Starting FastAPI backend..."
cd /Users/saltandpurple/dev/projects/personal/mllab/mapping-llm-censorship
python src/api.py &
API_PID=$!

echo "Waiting for API to start..."
sleep 3

echo "Starting React frontend..."
npm start &
REACT_PID=$!

echo "Both services started!"
echo "- API running on http://localhost:8001"
echo "- React app running on http://localhost:3000"
echo ""
echo "Press Ctrl+C to stop both services"

trap "kill $API_PID $REACT_PID" EXIT

wait