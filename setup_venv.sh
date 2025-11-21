#!/bin/bash
# setup_venv.sh - Script to set up a virtual environment for TopicForest

# Define colors for output
GREEN='\033[0;32m'
NC='\033[0m' # No Color

# Print colored message
print_message() {
    echo -e "${GREEN}[SETUP]${NC} $1"
}

# Check if Python 3 is installed
if ! command -v python &> /dev/null; then
    echo "Python 3 is not installed. Please install Python 3 before continuing."
    exit 1
fi

# Display Python version
python_version=$(python --version)
print_message "Found $python_version"

# Create a virtual environment in .venv folder
print_message "Creating a virtual environment in .venv folder..."
python -m venv .venv
if [ $? -ne 0 ]; then
    echo "Failed to create virtual environment. Please ensure venv is installed."
    exit 1
fi

# Activate the virtual environment
print_message "Activating virtual environment..."
source .venv/bin/activate
if [ $? -ne 0 ]; then
    echo "Failed to activate virtual environment."
    exit 1
fi

# Install required packages
print_message "Installing required packages..."
pip install --upgrade pip
pip install -r requirements.txt

# if some packages are not installed, print a warning and exit
if [ $? -ne 0 ]; then
    echo "Failed to install required packages. Please check your environment and try again."
    exit 1
fi

# Add .venv to .gitignore if it doesn't already contain it
if [ -f .gitignore ]; then
    if ! grep -q "^\.venv$" .gitignore; then
        print_message "Adding .venv to .gitignore..."
        echo ".venv" >> .gitignore
    else
        print_message ".venv already in .gitignore"
    fi
else
    print_message "Creating .gitignore and adding .venv..."
    echo ".venv" > .gitignore
fi

# Provide instructions for activation in the future
print_message "Setup complete!"
print_message "To activate the virtual environment in the future, run:"
echo "  source .venv/bin/activate"

print_message "To deactivate the virtual environment, run:"
echo "  deactivate"