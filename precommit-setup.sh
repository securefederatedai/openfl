#!/bin/bash

# Function to add the installation path to PATH
add_to_path() {
    if [[ ":$PATH:" != *":$1:"* ]]; then
        export PATH="$PATH:$1"
        echo "Added $1 to PATH"
    else
        echo "$1 is already in PATH"
    fi
}

# Function to check if Python and pip are installed
check_python_and_pip() {
    if ! command -v python3 &> /dev/null; then
        echo "Python3 is not installed. Please install Python3 and try again."
        exit 1
    fi

    if ! command -v pip &> /dev/null; then
        echo "pip is not installed. Please install pip and try again."
        exit 1
    fi
}

# Function to install pre-commit
install_precommit() {
    if ! command -v pre-commit &> /dev/null; then
        echo "pre-commit not found, installing..."
        pip install --user pre-commit
    else
        echo "pre-commit is already installed"
    fi
}

# Check if Python and pip are installed
check_python_and_pip

# Detect the operating system
OS="$(uname -s)"
case "$OS" in
    Linux*)     
        echo "Detected Linux"
        INSTALL_PATH="$HOME/.local/bin"
        install_precommit
        add_to_path "$INSTALL_PATH"
        ;;
    Darwin*)    
        echo "Detected MacOS"
        INSTALL_PATH="$HOME/.local/bin"
        install_precommit
        add_to_path "$INSTALL_PATH"
        ;;
    CYGWIN*|MINGW32*|MSYS*|MINGW*)
        echo "Detected Windows"
        INSTALL_PATH="$HOME/AppData/Local/Packages/PythonSoftwareFoundation.Python.3.12_qbz5n2kfra8p0/LocalCache/local-packages/Python312/Scripts"
        install_precommit
        add_to_path "$INSTALL_PATH"
        ;;
    *)
        echo "Unknown OS"
        exit 1
        ;;
esac

# Add the installation path to the shell profile for persistence
if [[ "$OS" == "Linux" || "$OS" == "Darwin" ]]; then
    SHELL_PROFILE="$HOME/.bashrc"
    if [[ -f "$HOME/.zshrc" ]]; then
        SHELL_PROFILE="$HOME/.zshrc"
    fi
    echo "export PATH=\$PATH:$INSTALL_PATH" >> "$SHELL_PROFILE"
    source "$SHELL_PROFILE"
elif [[ "$OS" == "CYGWIN"* || "$OS" == "MINGW"* || "$OS" == "MSYS"* ]]; then
    SHELL_PROFILE="$HOME/.bash_profile"
    echo "export PATH=\$PATH:$INSTALL_PATH" >> "$SHELL_PROFILE"
    source "$SHELL_PROFILE"
fi

# Verify the installation
if command -v pre-commit &> /dev/null; then
    echo "pre-commit installation successful"
    pre-commit --version
else
    echo "pre-commit installation failed"
    exit 1
fi
