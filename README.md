# AdaWorld Reproducibility Project

This is a fork of the original repo. The structure and code of the project will eventually become very different from the original implementation.

## Setup

### `uv` for Dependency Management

[`uv`](https://github.com/astral-sh/uv) is used for fast and reliable Python dependency management.
It handles environment creation, package installation, and lockfile syncing to ensure reproducible builds.

Create a new environment:

```bash
uv venv
```

To activate your environment:

```bash
source .venv/bin/activate
```

Sync dependencies from lockfile

```bash
uv sync
```

> Note: We have separate environments for different purposes in the `dependencies` folder. The main `pyproject.toml` in the root is used for running the actual training/evaluation code. You should change directory to the specific environment folder inside `dependencies` and run these commands in order for them to work properly. This is also required if you wanted to remove/add packages to them.

### Data Download and Dependencies

We have created dedicated environments for different purposes. To download the data, you should create and activate the environment:

```bash
cd dependencies/data_download
uv venv
source .venv/bin/activate
uv sync
```

This is only done once. After this, whenever you want to activate this environment you can just run:

```bash
source dependencies/data_download/.venv/bin/activate
```

The you should be able to downlod the data. You can download the data individually by running:

#### Something Something V2

```bash
bash data_download/something_something_v2/download_with_wget.sh
```

#### OpenX Embodiment

```bash
bash data_download/open_x/download_open_x.sh
python data_download/open_x/process_rtx.py
```

#### Miradata

```bash
python data_download/miradata/download_miradata_360p.py
```

### Ego4D

Not implemented yet.

## Sampling

TODO: Fill this section
