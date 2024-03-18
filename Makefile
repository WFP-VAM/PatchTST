# platforms to solve for: linux-64|win-64|osx-64|osx-arm64
PLATFORM ?= linux-64 win-64 osx-arm64 osx-64
PLATFORM_FLAGS=$(patsubst %, --platform=%, $(PLATFORM))
TAG ?= dev

default: PatchTST-env.conda-lock.yml

build-env: PatchTST-env.env

# Resolve environment for all platforms
lock:
	conda-lock lock --mamba $(PLATFORM_FLAGS) -f PatchTST-env.yml --lockfile PatchTST-env.conda-lock.yml

%.env: %.conda-lock.yml
	@echo "Creating conda environment: $*"
	conda-lock install -n $* $<

purge:
	@echo Removing lock file
	rm -f PatchTST-env.conda-lock.yml


lint:
	black SeasonTST
	isort SeasonTST
	ruff check .

lint-check:
	black --version
	black --check --diff SeasonTS
	isort --check SeasonTS
	ruff check .

mypy:
	mypy .


# resolve dependencies and rebuild the venv
update-dependencies: lock build-env
