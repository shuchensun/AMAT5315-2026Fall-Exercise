.PHONY: test build smoke part2 errors plots wolff compare all

test:
	cargo test

build:
	cargo build --release

smoke: build
	./target/release/ising --update metropolis --l 16 --t-from 1.8 --t-to 3.0 --t-step 0.2 --discard 100 --measure 500 --every 20 --seed 2026 --out runs/smoke

part2: build
	mkdir -p artifacts
	./target/release/ising --update metropolis --l 32 --t-from 1.5 --t-to 3.5 --t-step 0.1 --discard 2000 --measure 5000 --seed 1042 --out artifacts/coarse-l32
	./target/release/ising --update metropolis --l 64 --t-from 1.5 --t-to 3.5 --t-step 0.1 --discard 2000 --measure 5000 --seed 42 --out artifacts/coarse-l64
	./target/release/ising --update metropolis --l 32 --t-from 2.0 --t-to 2.6 --t-step 0.05 --discard 2000 --measure 100000 --seed 1042 --out artifacts/window-l32
	./target/release/ising --update metropolis --l 64 --t-from 2.0 --t-to 2.6 --t-step 0.05 --discard 2000 --measure 100000 --seed 42 --out artifacts/window-l64

errors: part2
	python3 scripts/peaks.py
	python3 scripts/errors.py

plots: errors
	python3 scripts/plots.py

wolff: build
	mkdir -p artifacts
	./target/release/ising --update wolff --l 64 --t-from 2.0 --t-to 2.6 --t-step 0.05 --discard 20000 --measure 100000 --seed 42 --out artifacts/wolff-l64
	./target/release/ising --update wolff --l 32 --t-from 2.0 --t-to 2.6 --t-step 0.05 --discard 20000 --measure 100000 --seed 1042 --out artifacts/wolff-l32

compare: wolff
	python3 scripts/compare.py

all: test plots compare
