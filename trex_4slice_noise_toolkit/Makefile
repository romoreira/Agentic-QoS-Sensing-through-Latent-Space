preview:
	python3 scripts/preview_traffic_model.py --duration 180 --dt 0.5 --csv runs/traffic_plan.csv --tb runs/tensorboard/traffic_plan

plan-dynamic:
	python3 scripts/run_trex_dynamic_4slice.py --plan-only --duration 180 --dt 1.0 --csv runs/trex_dynamic_targets.csv --tb runs/tensorboard/trex_dynamic

check-bess:
	bash bess/check_bess_4slices.sh
