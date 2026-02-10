PY=python
CFG=code/config.yml

.PHONY: env clean nodes edges sim bn eval export all

env:
	conda env export --no-builds -n legalnlp > environment.yml

clean:
	rm -f data/interim/*.jsonl
	rm -f data/processed/*.csv
	mkdir -p outputs/logs

nodes:
	$(PY) code/step2_nodes.py --config $(CFG) | tee outputs/logs/nodes.log

edges:
	$(PY) code/step3_edges.py --config $(CFG) | tee outputs/logs/edges.log

sim:
	$(PY) sim/abm.py --config $(CFG) --n 3000 | tee outputs/logs/abm.log

bn:
	$(PY) code/step4_bn_template.py --config $(CFG) | tee outputs/logs/bn.log

eval:
	$(PY) eval/run_eval.py --config $(CFG) | tee outputs/logs/eval.log

export:
	@echo "Tables at data/processed and BN at outputs/bn_template.gml"

all: clean nodes edges sim bn eval export