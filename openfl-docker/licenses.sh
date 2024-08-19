#!/bin/bash
# Downloads thirdparty licenses.
# Save the list of installed packages to base_packages.txt
dpkg --get-selections | grep -v deinstall | awk '{print $1}' > base_packages.txt

# If INSTALL_SOURCES is set to "yes", perform additional operations
if [ "$INSTALL_SOURCES" = "yes" ]; then
    # Save the list of all installed packages to all_packages.txt
    dpkg --get-selections | grep -v deinstall | awk '{print $1}' > all_packages.txt
    
    # Enable source repositories in APT sources list
    sed -Ei 's/# deb-src /deb-src /' /etc/apt/sources.list
    
    # Update the package list again after enabling source repositories
    apt-get update
    
    # Process each package and download source if it matches specific licenses
    grep -v -f base_packages.txt all_packages.txt | while read -r package; do
        name=$(echo "${package//:/ }" | awk '{print $1}')
        echo "$name" >> all_dependencies.txt
        echo "$name" >> licenses.txt
        cat /usr/share/doc/"$name"/copyright >> licenses.txt
        if grep -lE 'GPL|MPL|EPL' /usr/share/doc/"$name"/copyright; then
            apt-get source -q --download-only "$package"
        fi
    done
    
    # Clean up
    rm -rf ./*packages.txt
    echo "Download source for $(find . | wc -l) third-party packages: $(du -sh)"
    
    # Clean up APT lists again
    rm -rf /var/lib/apt/lists/*
fi

mkdir -p thirdparty
cd thirdparty

# If INSTALL_SOURCES is set to "yes", perform additional operations
if [ "$INSTALL_SOURCES" = "yes" ]; then
    # Install pip-licenses and generate license files
    pip install --no-cache-dir pip-licenses
    pip-licenses -l >> licenses.txt
    
    # Append dependency list to all_dependencies.txt
    pip-licenses | awk '{for(i=1;i<=NF;i++) if(i!=2) printf $i" "; print ""}' | tee -a all_dependencies.txt
    
    # Download source packages for Python packages with specific licenses
    pip-licenses | grep -E 'GPL|MPL|EPL' | awk '{OFS="=="} {print $1,$2}' | xargs pip download --no-binary :all:
fi
