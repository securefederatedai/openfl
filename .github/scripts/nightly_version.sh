#!/bin/bash
set -e

if [ "${TEST_PYPI}" = true ]; then
    echo "Using Test PyPI"
    old_version=$(curl -s https://test.pypi.org/pypi/openfl-nightly/json | python -c "import sys, json; print(json.load(sys.stdin)['info']['version']);")
else
    old_version=$(curl -s https://pypi.org/pypi/openfl-nightly/json | python -c "import sys, json; print(json.load(sys.stdin)['info']['version']);")
fi

echo "Old version: $old_version"

version=$(grep -oP "(?<=version=')[^']+" setup.py)
date_suffix=$(date +%Y%m%d)
new_version="${version}${date_suffix}"

# Remove the last digit of old_version after the last character
truncated_old_version=$(echo "${old_version}" | sed 's/.$//')
echo "Truncated old version: $truncated_old_version"
echo "New version: $new_version"
if [ "${truncated_old_version}" == "${new_version}" ]; then
    # Increment the last digit of old_version
    last_digit=$(echo "${old_version}" | grep -o '.$')
    incremented_last_digit=$((last_digit + 1))
    new_version="${new_version}${incremented_last_digit}"
else
    # Append 0 as the last digit
    echo "No version present for current date"
    new_version="${new_version}0"
fi

echo "Final NEW_VERSION=${new_version}"

# get URL with commit hash
base_url="https://github.com/securefederatedai/openfl/tree/"
full_url="${base_url}${COMMIT_ID}"
echo "Repository URL: $full_url"

sed -i 's/name=.*/name="openfl-nightly",/' setup.py
sed -i "s/version=.*/version='$new_version',/" setup.py
sed -i 's/Development Status :: 5 - Production\/Stable/Development Status :: 4 - Beta/' setup.py
sed -i "s|'Source Code': '.*'|'Source Code': '${full_url}'|g" setup.py

echo "NEW_VERSION=${new_version}" >> $GITHUB_ENV