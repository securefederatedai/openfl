# Contributing to OpenFL

We welcome contributions from the community. There are several ways to contribute:
* Improvements in [documentation](https://openfl.readthedocs.io/en/latest/).
* Contributing to OpenFL's code-base: via bug-fixes or feature additions.
* Answering questions on our [discussions page](https://github.com/securefederatedai/openfl/discussions).
* Participating in our [roadmap](https://github.com/securefederatedai/openfl/blob/develop/ROADMAP.md) discussions.

We have a slack [channel](https://join.slack.com/t/openfl/shared_invite/zt-ovzbohvn-T5fApk05~YS_iZhjJ5yaTw) and we host regular [community meetings](https://github.com/securefederatedai/openfl#support).


## How to contribute code
### Step 1. Open an issue

Before you start making any changes, it is always good to open an [issue](https://github.com/securefederatedai/openfl/issues/new/choose) first (assuming one does not already exist), outlining your proposed changes. We can give you feedback, and potentially validate the proposed changes.

For minor changes (akin to a documentation or bug fix), proceed to opening a Pull Request (PR) directly.

### Step 2. Make code changes

To modify code, you need to fork the repository. Set up a development environment as covered in the section "Setup environment" below.

### Step 3. Create a Pull Request (PR)

Once the change is ready, open a PR from your branch in your fork, to the `develop` branch in [securefederatedai/openfl](https://github.com/securefederatedai/openfl). OpenFL follows standard recommendations of PR formatting. Find more details [here](https://github.blog/2015-01-21-how-to-write-the-perfect-pull-request/).

### Step 4. Sign your work

Signoff your patch commits using your real name. We discourage anonymous contributions.

    Signed-off-by: Joe Smith <joe.smith@email.com>

If you set your `user.name` and `user.email` git configs, you can sign your
commits using `git commit --signoff`.

Your signature [certifies](http://developercertificate.org/) that you wrote the patch, or, you otherwise have the right to pass it on as an open-source patch.

OpenFL is licensed under the [Apache 2.0 license](https://github.com/securefederatedai/openfl/blob/develop/LICENSE). By contributing to the project, you agree to the license and copyright terms therein and release your contribution under these terms.

### Step 5. Code review and merge

Verify that your contribution passes all tests in our CI/CD pipeline. In case of a failure, like shown below, look into the error messages and try to fix them.

![CI/CD](docs/images/CI_details.png)

Meanwhile, a reviewer will review the pull request and provide comments. Post few iterations of
reviews and changes (depending on the complexity of the changes), PR will be approved for merge.

## Setup environment

We recommend setting up a local dev environment. Clone your forked repo to your local machine and install the dependencies.

```shell
git clone https://github.com/YOUR_GITHUB_USERNAME/openfl.git
cd openfl
pip install -U pip setuptools wheel
pip install .
pip install -r linters-requirements.txt
```

## Code style

OpenFL uses [ruff](https://github.com/astral-sh/ruff) to lint/format code and [precommit](https://pre-commit.com/) checks.

Run the following command at the **root** directory of the repo to format your code.

```
sh scripts/format.sh
```
You may need to resolve errors that could not be resolved by autoformatting. To only show lint errors, run `sh scripts/lint.sh` at the **root** directory of the repo.

### Docstrings
Since docstrings cannot be checked or standardized, if you do write/edit any docstring, make sure to check them manually. OpenFL docstrings should follow the conventions below:

A **class** or a **function** docstring may contain:
* A one-line description of the class/function.
* Paragraph(s) of detailed information.
* Optional `Examples` section.
* `Args` section for arguments under `__init__()`.
