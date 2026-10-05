PYTHON ?= python

.PHONY: install-extension test test-unit test-kernel check example

install-extension:
	$(PYTHON) -m pip install -e nanovllm/custom --no-build-isolation

test: test-unit test-kernel

test-unit:
	$(PYTHON) -m unittest tests.test_scheduler -v

test-kernel:
	$(PYTHON) -m unittest tests.test_custom_attention -v

check:
	$(PYTHON) -m compileall -q nanovllm benchmarks tests

example:
	$(PYTHON) example.py --chunked-prefill --custom-kernel
