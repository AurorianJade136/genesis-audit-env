.PHONY: verify generate test demo benchmark docker
verify:
	python verify.py
generate:
	python -m src.generator
test:
	python -m pytest -q
demo:
	python demo.py
benchmark:
	python -m src.runner benchmark
docker:
	docker build -t genesis-audit-env . && docker run --rm genesis-audit-env
