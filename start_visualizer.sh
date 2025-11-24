#!/bin/bash

# Script to start the visualizer API in a tmux session
SESSION_NAME="api"

# If tmux is not installed, run without tmux
if ! command -v tmux &>/dev/null; then
    echo "tmux is not installed. Running without tmux."
    # Run 'gw' if available, then start the visualizer API
    if command -v gw &>/dev/null; then
        gw
    fi
    python gridworks-visualizer/visualizer_api.py
    exit 0
fi

# Check if tmux session already exists
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "Session '$SESSION_NAME' already exists. Attaching..."
    tmux attach-session -t "$SESSION_NAME"
else
    echo "Creating new tmux session '$SESSION_NAME'..."
    
    # Create a new tmux session and run commands
    tmux new-session -d -s "$SESSION_NAME" -c "$(pwd)"
    sleep 0.5
    
    # Run 'gw' command (your alias)
    tmux send-keys -t "$SESSION_NAME" "gw" C-m
    
    # Wait a moment for gw to complete
    sleep 0.5
    
    # Start the visualizer API
    tmux send-keys -t "$SESSION_NAME" "python gridworks-visualizer/visualizer_api.py" C-m
    
    # Attach to the session
    tmux attach-session -t "$SESSION_NAME"
fi

