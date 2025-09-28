# AdaWorld Reproducibility Project

This is a fork of the original repo. The structure and code of the project will eventually become very different from the original implementation.

## Setup

### Data Download and Dependencies

We have created dedicated environments for different purposes. To download the data, you should first activate the environment:

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
