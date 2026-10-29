PYTHON ?= python3
export PYTHONPATH := src

.PHONY: all analysis paper test clean

all: analysis paper

## Run every model, write results/, figures/ and paper/generated/
analysis:
	$(PYTHON) scripts/run_all.py

## Build the main paper and the Supplementary Information PDFs
paper:
	cd paper && latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
	cd paper && latexmk -pdf -interaction=nonstopmode -halt-on-error supplementary.tex

test:
	$(PYTHON) -m pytest -q

clean:
	cd paper && latexmk -C main.tex supplementary.tex
	rm -rf results/* figures/* paper/generated/*
