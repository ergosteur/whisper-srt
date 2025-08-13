# Variables you can override on the CLI:
#   make venv FLAVOUR=cpu
#   make run AUDIO=sample.mp4 MODEL_DIR=./anime_whisper LANGUAGE=en

PYTHON := python3
VENV := .venv
FLAVOUR ?= auto
AUDIO ?= input.mp4
MODEL_DIR ?= ./anime_whisper
LANGUAGE ?= 
TASK ?= transcribe
CHUNK ?= 30
BATCH ?= 8

.PHONY: venv clean run

venv:
	./setup_venv.sh $(FLAVOUR)

run:
	$(VENV)/bin/python whisper_srt.py \
		--audio "$(AUDIO)" \
		--model-dir "$(MODEL_DIR)" \
		$(if $(LANGUAGE),--language $(LANGUAGE),) \
		--task $(TASK) \
		--chunk-length-s $(CHUNK) \
		--batch-size $(BATCH)

clean:
	rm -rf $(VENV) __pycache__ *.srt
