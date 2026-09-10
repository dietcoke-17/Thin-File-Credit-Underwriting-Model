.PHONY: install test run docker-build docker-run clean

install:
	pip install -r requirements.txt

test:
	pytest tests/ -v

run:
	python main.py

docker-build:
	docker build -t thin-file-credit-scorecard .

docker-run:
	docker run --rm -v $(PWD)/results:/app/results -v $(PWD)/reports:/app/reports thin-file-credit-scorecard

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf .pytest_cache
