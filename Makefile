# norm-boundary reproducibility targets (plan §10 G5 AC: `make figures`)

.PHONY: figures tables all test paper

paper:
	cd paper && tectonic main.tex

all: figures tables

figures:
	uv run python -m src.theory.fig1
	uv run python -m src.synth.analyze
	uv run python -m experiments.g5_analysis

tables:
	uv run python -m experiments.g4_table1
	uv run python -m experiments.g5_ext_tables

test:
	uv run pytest -q
