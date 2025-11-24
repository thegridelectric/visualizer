#!/bin/bash

# Array of session names
sessions=("beech" "oak" "fir" "maple" "elm")

# Start websocket servers in tmux sessions
for session in "${sessions[@]}"; do
    tmux new-session -d -s "$session"
    tmux send-keys -t "$session" "gw" C-m
    tmux send-keys -t "$session" "cd ${session}-webinter" C-m
    tmux send-keys -t "$session" "./start_websocket_server.sh" C-m
done

echo "Started all websocket servers in tmux sessions:"
for session in "${sessions[@]}"; do
    echo "  - $session: tmux attach -t $session"
done

