# Reproduction targets. `verify-numbers` needs only results/ and no GPU.

.PHONY: verify-numbers score tables figures test

# Re-derive every number the paper prints, from the frozen CSVs.
verify-numbers:
	uv run python -m experiments.verify_paper_numbers

# Recompute the Level Predictability Score from the curated data.
score:
	uv run python -m experiments.compute_lps_official
	uv run python -m experiments.graded_lps --lps

tables:
	uv run python -m experiments.g4_table1
	uv run python -m experiments.g11_table

figures:
	uv run python -m experiments.fig_ccai_mech2
	uv run python -m experiments.fig_ccai_lps

test:
	uv run pytest -q
