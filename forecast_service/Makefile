.PHONY: check_all format typecheck lint clean build

check_all: format typecheck lint clean

format:
	python -m black .

typecheck:
	python -m mypy .

lint:
	python -m flake8 .

clean:
	find . -name '*.pyc' -delete
	find . -name '__pycache__' -delete

build:
	docker build -t forecast_service .

run_dev:
	uvicorn app.main:forecast_service --host 0.0.0.0 --port 5051

