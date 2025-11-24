#!/bin/bash

# Script to start the visualizer API in a tmux session

SESSION_NAME="api"

# Check if tmux session already exists
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "Session '$SESSION_NAME' already exists. Attaching..."
    tmux attach-session -t "$SESSION_NAME"
else
    echo "Creating new tmux session '$SESSION_NAME'..."
    
    # Create a new tmux session and run commands
    tmux new-session -d -s "$SESSION_NAME" -c "$(pwd)"
    
    # Run 'gw' command (your alias)
    tmux send-keys -t "$SESSION_NAME" "gw" C-m
    
    # Wait a moment for gw to complete
    sleep 1
    
    # Start the visualizer API
    tmux send-keys -t "$SESSION_NAME" "python visualizer_api.py" C-m
    
    # Attach to the session
    tmux attach-session -t "$SESSION_NAME"
fi

