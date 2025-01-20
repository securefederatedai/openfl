## Pre-commit with Bandit

To ensure code quality and security, we use [pre-commit](https://pre-commit.com/) with [Bandit](https://bandit.readthedocs.io/en/latest/) to automatically scan for security issues before commits. 

Follow the steps below to set up and use pre-commit in your local development environment.

### Setup

1. **Clone the repository**:
   
   ```sh
   git clone https://github.com/intel-innersource/frameworks.ai.openfl.openfl-security.git
   cd frameworks.ai.openfl.openfl-security
   ```
   
2. **Run the setup script**:

   We have provided a `precommit-setup.sh` script to simplify the installation process. This script will install pre-commit and set up the pre-commit hooks.

   ```sh
   ./precommit-setup.sh
   ```

   The `setup.sh` script performs the following actions:
   - Check for prerequisties in local: (python, pip)
   - Installs pre-commit if it is not already installed.
   - Installs the pre-commit hooks defined in the .pre-commit-config.yaml file.

3. **Verify the installation**:

   After running the setup script, you can verify that pre-commit is installed and the hooks are set up correctly by running:

   ```sh
   pre-commit --version
   pre-commit install
   ```

### Usage

Once the pre-commit hooks are installed, Bandit scans will automatically run before each commit. If any issues are found, the commit will be aborted, and you will need to fix the issues before committing again.

1. **Make changes to your code**:

   Edit your files as needed.

2. **Stage your changes**:

   ```sh
   git add <file>
   ```

3. **Commit your changes**:

   ```sh
   git commit -m "Your commit message"
   ```

   During the commit process, pre-commit will automatically run the Bandit scan. If the scan is successful, the commit will proceed. If any issues are found, the commit will be aborted, and you will need to address the issues before committing again.

### How to bypass precommit hooks?

To exclude the bandit pre-commit hook when making a Git commit, you can use the --no-verify option. This bypasses any pre-commit hooks that are set up in your repository.

```sh
git commit --no-verify -m "Your commit message"
```
