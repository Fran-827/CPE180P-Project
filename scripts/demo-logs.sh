#!/bin/bash
tmux kill-session -t demo 2>/dev/null
tmux new-session -d -s demo 'docker logs -f fashion'
tmux split-window -h 'watch -n 2 "curl -s localhost:8000/metrics"'
tmux attach -t demo
