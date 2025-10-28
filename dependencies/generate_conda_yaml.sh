#!/usr/bin/env bash
set -euo pipefail

# Get cli arg for environment name and check that it exists
ENV_NAME=$1
if [ -z "$ENV_NAME" ]; then
  echo "Error: Environment name not provided."
  exit 1
fi

# Get username from $HOME
USERNAME=$(basename "$HOME")
REPO_DIR="/scratch-shared/$USERNAME/AdaWorld"

# If ENV_NAME is adaworld, use repo dir as env dir
if [ "$ENV_NAME" == "adaworld" ]; then
  ENV_DIR="$REPO_DIR"
else
  ENV_DIR="$REPO_DIR/dependencies/$ENV_NAME"

  # Make sure the environment directory exists
  if [ ! -d "$ENV_DIR" ]; then
    echo "Error: Environment '$ENV_NAME' does not exist."
    echo "Available environments are:"
    ls "$REPO_DIR/dependencies" | grep -v "generate_conda_yaml.sh"
    echo "adaworld"
    exit 1
  fi
fi

# Prefer using the env's python/pip directly (avoid relying on activation)
PY_BIN="$ENV_DIR/.venv/bin/python"
PIP_BIN="$ENV_DIR/.venv/bin/pip"
ACTIVATE="$ENV_DIR/.venv/bin/activate"

if [ ! -x "$PY_BIN" ] || [ ! -x "$PIP_BIN" ]; then
  # Try sourcing activate as a fallback (keeps original behaviour)
  if [ -f "$ACTIVATE" ]; then
    # shellcheck source=/dev/null
    echo "source $ACTIVATE"
    source "$ACTIVATE" || {
      echo "Error: Could not activate virtual environment."
      exit 1
    }
    PY_BIN="$(which python)"
    PIP_BIN="$(which pip)"
  else
    echo "Error: Python/pip not found in virtualenv and activate script missing."
    exit 1
  fi
fi

# Run sync env from $ENV_DIR
echo "cd $ENV_DIR && uv sync"
cd "$ENV_DIR" || exit # '|| exit' ensures the script stops if cd fails
uv sync

# Determine Python major.minor version from the env
echo ""$PY_BIN" -c 'import sys; print("{}.{}.{}".format(*sys.version_info[:3]))'"
PY_VER=$("$PY_BIN" -c 'import sys; print("{}.{}.{}".format(*sys.version_info[:3]))')

# Get pip freeze output (keeps VCS and editable lines)
echo "uv pip list --format freeze"
PIP_FREEZE=$(uv pip list --format freeze)

# Remove problematic packages
PIP_FREEZE=$(echo "$PIP_FREEZE" | grep -v 'tensorflow-metadata==0.5.0')

if [ "$ENV_NAME" == "adaworld" ]; then
  CONDA_YAML_PATH="$REPO_DIR/conda_environment.yaml"
else
  CONDA_YAML_PATH="$REPO_DIR/dependencies/$ENV_NAME/conda_environment.yaml"
fi

# Write conda YAML
{
  echo "name: $ENV_NAME"
  echo "channels:"
  echo "  - conda-forge"
  echo "  - defaults"
  echo "dependencies:"
  echo "  - python=$PY_VER"
  echo "  - pip"
  echo "  - pip:"
  # If there are no pip packages, still create an empty list
  if [ -z "$PIP_FREEZE" ]; then
    echo "    # no pip packages detected"
  else
    # Write each pip requirement as a quoted string to be safe
    while IFS= read -r pkg; do
      # Skip empty lines
      [ -z "$pkg" ] && continue
      # YAML safe quoting for lines that may contain special chars
      printf '    - "%s"\n' "$(printf '%s' "$pkg" | sed 's/"/\\"/g')"
    done <<< "$PIP_FREEZE"
  fi
} > "$CONDA_YAML_PATH"

echo "Generated conda environment YAML at: $CONDA_YAML_PATH"

# Deactivate current uv env
echo "deactivate"
deactivate || true

# Activate conda base env
echo "source ~/miniconda3/bin/activate"
source ~/miniconda3/bin/activate

# Create conda env from the generated YAML
create_cmd="conda env create -f $CONDA_YAML_PATH"
echo "$create_cmd"
eval "$create_cmd"