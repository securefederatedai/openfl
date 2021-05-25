## HOW TO ACTIVATE BASH COMPLETE

1. Execute string below in your terminal to add `fx` command to bash autocomplete
    This command add will update ~/.bashrc
```bash
tee -a ~/.bashrc << EOF
_fx_completion() {
    local IFS=\$'
'
    COMPREPLY=( \$( env COMP_WORDS="\${COMP_WORDS[*]}" \
                   COMP_CWORD=\$COMP_CWORD \
                   _FX_COMPLETE=complete \$1 ) )
    return 0
}

_fx_completionetup() {
    local COMPLETION_OPTIONS=""
    local BASH_VERSION_ARR=(\${BASH_VERSION//./ })
    # Only BASH version 4.4 and later have the nosort option.
    if [ \${BASH_VERSION_ARR[0]} -gt 4 ] || ([ \${BASH_VERSION_ARR[0]} -eq 4 ] && [ \${BASH_VERSION_ARR[1]} -ge 4 ]); then
        COMPLETION_OPTIONS="-o nosort"
    fi

    complete \$COMPLETION_OPTIONS -F _fx_completion fx
}

_fx_completionetup;

EOF
```

2. Open new terminal to accept changes in `~/.bashrc`

3. Autocomplete activation finished!