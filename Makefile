.PHONY: test test-original test-all bench lint typecheck clean install

test:
	python3 -m pytest test_hybrid_frame_pytest.py -v --tb=short -x

test-original:
	python3 test_hybrid_frame.py

test-all:
	$(MAKE) test
	$(MAKE) test-original

bench:
	python3 -m pytest bench_hybrid_frame.py -v --benchmark-only

lint:
	ruff check .

typecheck:
	mypy hybrid_frame.py

clean:
	rm -rf __pycache__ .pytest_cache .ruff_cache .mypy_cache
	rm -rf *.egg-info dist/ build/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name '*.pyc' -delete
	find . -type f -name '*.pyo' -delete
	rm -f .coverage
	rm -rf htmlcov/

install:
	pip install -e ".[all]"
