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

# Make sure the environment directory exists
ENV_DIR="$REPO_DIR/dependencies/$ENV_NAME"
if [ ! -d "$ENV_DIR" ]; then
  echo "Error: Environment '$ENV_NAME' does not exist."
  echo "Available environments are:"
  ls "$REPO_DIR/dependencies" | grep -v "generate_conda_yaml.sh"
  exit 1
fi

# Run conda deactivate in case a conda env is active
if command -v conda &> /dev/null; then
  conda deactivate || true
fi

# Prefer using the env's python/pip directly (avoid relying on activation)
PY_BIN="$ENV_DIR/.venv/bin/python"
PIP_BIN="$ENV_DIR/.venv/bin/pip"
ACTIVATE="$ENV_DIR/.venv/bin/activate"

if [ ! -x "$PY_BIN" ] || [ ! -x "$PIP_BIN" ]; then
  # Try sourcing activate as a fallback (keeps original behaviour)
  if [ -f "$ACTIVATE" ]; then
    # shellcheck source=/dev/null
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
cd "$ENV_DIR" || exit # '|| exit' ensures the script stops if cd fails
uv sync

# Determine Python major.minor version from the env
PY_VER=$("$PY_BIN" -c 'import sys; print("{}.{}".format(*sys.version_info[:2]))')

# Get pip freeze output (keeps VCS and editable lines)
PIP_FREEZE=$(uv pip list --format freeze)

# Remove problematic packages
PIP_FREEZE=$(echo "$PIP_FREEZE" | grep -v 'tensorflow-metadata==0.5.0')

CONDA_YAML_PATH="$REPO_DIR/dependencies/$ENV_NAME/conda_environment.yaml"

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
echo "To create the environment from this file run:"
echo "  conda env create -f \"$CONDA_YAML_PATH\""