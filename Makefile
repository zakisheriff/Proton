# Proton 1 — full pipeline shortcuts
# Override any variable: `make pretrain CONFIG=small`

CONFIG ?= nano
PY     ?= python3

.PHONY: help data tokenizer tokens pretrain sft rl eval chat serve gateway smoke clean

help:
	@echo "Proton 1 pipeline targets:"
	@echo "  make data        - build corpus.txt from local source (scripts/prepare_data.py)"
	@echo "  make download     - stream a real code corpus from HuggingFace"
	@echo "  make tokenizer    - train the BPE tokenizer"
	@echo "  make tokens       - tokenize corpus -> data/tokens.pt"
	@echo "  make pretrain     - pretrain base model (CONFIG=nano|small)"
	@echo "  make sft          - instruction-tune the base model"
	@echo "  make rl           - RL with execution rewards"
	@echo "  make eval         - pass@k code evaluation"
	@echo "  make chat         - interactive chat"
	@echo "  make serve        - run the Python inference server"
	@echo "  make gateway      - run the Node/Express API gateway"
	@echo "  make smoke        - full nano pipeline end-to-end"

data:
	$(PY) scripts/prepare_data.py --src proton1 scripts data --out data/corpus.txt

download:
	$(PY) scripts/download_data.py --max-docs 50000 --out data/corpus.txt

tokenizer:
	$(PY) -m proton1.tokenizer --corpus data/corpus.txt --vocab 2048 --out data/tokenizer.json

tokens:
	$(PY) -m proton1.data --corpus data/corpus.txt --tokenizer data/tokenizer.json --out data/tokens.pt

pretrain:
	$(PY) -m proton1.pretrain --config $(CONFIG)

sft:
	$(PY) -m proton1.sft --config $(CONFIG)

rl:
	$(PY) -m proton1.rl --config $(CONFIG)

eval:
	$(PY) -m proton1.eval --ckpt checkpoints/$(CONFIG)-sft.pt

chat:
	$(PY) -m proton1.generate --ckpt checkpoints/$(CONFIG)-sft.pt --chat

serve:
	$(PY) -m serving.server --ckpt checkpoints/$(CONFIG)-sft.pt

gateway:
	cd serving/gateway && npm install && npm start

smoke: data tokenizer tokens pretrain sft
	@echo "Smoke pipeline complete. Try: make chat"

clean:
	rm -rf checkpoints/*.pt data/tokens.pt __pycache__ proton1/__pycache__
