PYTHON?=python -X dev
DATASET_URL?=https://raw.githubusercontent.com/alexfikl/2025-fractal-connectomes-paper-experiments/refs/heads/main/data

help: 			## Show this help
	@echo -e "\nSpecify a command. The choices are:\n"
	@grep -E '^[0-9a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[0;36m%-16s\033[m %s\n", $$1, $$2}'
	@echo ""
.PHONY: help

.PRECIOUS: %.mat

%.mat:
	curl -o $@ "$(DATASET_URL)/$@"

%.npz: %.mat
	$(PYTHON) scripts/mat2npz.py \
		--variable-name 'w:matrices' \
		--transpose \
		$<

datasets: Emotion_LR_Task.npz Rest_LR_Task.npz  ## Download datesets and convert to .npz
	@echo -e ">> \e[1;32mFinished convergint datasets!\e[0m"
.PHONY: datasets

clean:  			## Remove generated files
	rm -rf *.npz
.PHONY: clean

purge: clean  		## Remove downloaded datasets
	rm -rf *.mat
.PHONY: purge
