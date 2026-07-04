#!/usr/bin/env bash
# Start the Proton 1 model server for the atom-coder CLI.
#   Terminal 1:  ./serve_proton1.sh
#   Terminal 2:  LOCAL_ENDPOINT=http://localhost:8080 \
#                /Users/afraasheriff/Desktop/The_Atom/theatomcoder-cli/atom-coder/atomcoder
# Then pick "Proton 1" in the Select Local Model menu.
set -e
cd "$(dirname "$0")"
source .venv-mlx/bin/activate

# Use the fine-tuned adapter automatically once a good one exists, else base + identity.
ADAPTER_ARG="--no-adapter"
if [ -f adapters/proton1-bf16/adapters.safetensors ]; then
  ADAPTER_ARG="--model models/proton1-bf16 --adapter adapters/proton1-bf16"
fi

echo "Starting Proton 1 server on http://localhost:8080 ..."
exec python -m serving.proton_server $ADAPTER_ARG --port 8080
