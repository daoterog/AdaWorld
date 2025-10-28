
# Get username from $HOME
USERNAME=$(basename "$HOME")
REPO_DIR="/scratch-shared/$USERNAME/AdaWorld"

echo "Creating all conda environments..."

# Create the main environment
echo "conda env create -f "$REPO_DIR/conda_environment.yaml""
conda env create -f "$REPO_DIR/conda_environment.yaml"

# Create other environments
for ENV_DIR in dependencies/*; do
  ENV_NAME=$(basename "$ENV_DIR")
  echo "Creating $ENV_NAME environment..."
  echo "conda env create -f "$REPO_DIR/$ENV_DIR/conda_environment.yaml""
  conda env create -f "$REPO_DIR/$ENV_DIR/conda_environment.yaml" -n "$ENV_NAME"
done
