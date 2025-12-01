SHELL:=/usr/bin/env bash
DEPENDENCIES:=-e support pre-commit==4.5.0 ipython pytest Cython

venv:
	@echo "Setting up virtual environment..."
	@if [ -d ".venv" ]; then \
		echo "Error: .venv directory already exists"; \
		exit 1; \
	fi
	@python3 -m venv .venv
	@echo "Installing dependencies..."
	@.venv/bin/pip install ${DEPENDENCIES}

	@echo "Setting up PyPy virtual environment..."
	@if [ -d ".pypy-venv" ]; then \
		echo "Error: .pypy-venv directory already exists"; \
		exit 1; \
	fi
	@virtualenv -p `which pypy3` .pypy-venv
	@echo "Installing dependencies..."
	@.pypy-venv/bin/pypy -m pip install ${DEPENDENCIES}
	@echo "Done."

new-day:
	@day=$(if $(day),$(day),$(shell date +%d)); \
	if [ -d "day$$day" ]; then \
		echo "Error: day$$day directory already exists"; \
		exit 1; \
	fi; \
	echo "Creating day$$day directory..."; \
	cp -r day00 day$$day

lint:
	@.venv/bin/pre-commit run --all-files

test:
	@fail=0; \
	for num in `seq -w 1 31`; do \
		day="day$$num"; \
		if [ -d "$$day" ]; then \
			echo "Testing in $$day"; \
			(cd $$day; pytest part*.py -v) || fail=1; \
		fi; \
	done; \
	if [ $$fail -eq 1 ]; then \
		exit 1; \
	fi

benchmark:
	@if [ "$(off-formatting)" != "1" ]; then \
		echo "| Day   | Part     | CPython | PyPy  |"; \
		echo "|-------|----------|---------|-------|"; \
	fi
	@for num in `seq -w 1 31`; do \
		day="day$$num"; \
		if [ -d "$$day" ]; then \
			(cd $$day; \
				for module in part*.py; do \
					result_cpython=`../.venv/bin/python $$module -b | grep "Average time" | awk '{print $$3}'`; \
					result_pypy=`../.pypy-venv/bin/pypy $$module -b | grep "Average time" | awk '{print $$3}'`; \
					if [ "$(off-formatting)" = "1" ]; then \
						echo "$$result_cpython | $$result_pypy"; \
					else \
						echo "| $$day | $$module | $$result_cpython | $$result_pypy |"; \
					fi \
				done \
			); \
		fi \
	done
