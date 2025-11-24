#!/bin/bash

# Array of session names
sessions=("beech" "oak" "fir" "maple" "elm")

# Start websocket servers in tmux sessions
for session in "${sessions[@]}"; do
    # Check if session already exists
    if ! tmux has-session -t "$session" 2>/dev/null; then
        tmux new-session -d -s "$session"
        tmux send-keys -t "$session" "gw" C-m
        tmux send-keys -t "$session" "cd ${session}-webinter" C-m
        tmux send-keys -t "$session" "./start_websocket_server.sh" C-m
    fi
done

echo "Started all websocket servers in tmux sessions:"
for session in "${sessions[@]}"; do
    echo "  - $session: tmux attach -t $session"
done

