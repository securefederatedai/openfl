.. # Copyright (C) 2021 Intel Corporation
.. # Licensed subject to the terms of the separately executed evaluation license agreement between Intel Corporation and you.

Bash Autocomplete Activation
############################

|productName| allows to activate bash completion in CLI mode to give faster access to all available commands.

Preparation
~~~~~~~~~~~

Make sure that you inside virtual environment with installed |productName|.
If not use the instruction :ref:`install_initial_steps`.

Create ~/.fx-autocomplete.sh script
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

   This step need to be done only one time when you don't have `~/.fx-autocomplete.sh` or `~/.fx-autocomplete.sh` have corrupted content.
   
   .. code-block:: console

      $ _FX_COMPLETE=bash_source fx > ~/.fx-autocomplete.sh

   Check that command was executed correctly.

   .. code-block:: console

      $ cat ~/.fx-autocomplete.sh

   Console output should look like example below (Click==8.0.1), but could be different depend on `Click https://click.palletsprojects.com/en/8.0.x/`_ version:
   
   .. code-block:: console

      _fx_completion() {
          local IFS=$'\n'
          local response

          response=$(env COMP_WORDS="${COMP_WORDS[*]}" COMP_CWORD=$COMP_CWORD _FX_COMPLETE=bash_complete $1)

          for completion in $response; do
              IFS=',' read type value <<< "$completion"

              if [[ $type == 'dir' ]]; then
                  COMREPLY=()
                  compopt -o dirnames
              elif [[ $type == 'file' ]]; then
                  COMREPLY=()
                  compopt -o default
              elif [[ $type == 'plain' ]]; then
                  COMPREPLY+=($value)
              fi
          done

          return 0
      }

      _fx_completion_setup() {
          complete -o nosort -F _fx_completion fx
      }

      _fx_completion_setup;

Activate autocomplete feature
~~~~~~~~~~~~~~~~~~~~~

   This step should be done every time when you open a new terminal window.

   .. code-block:: console

      $ source ~/.fx-autocomplete.sh

Auto activation autocomplete
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

   To save your time you can add autocomplete activation step to `~/.bashrc`.
   
   .. code-block:: bash
      . ~/.fx-autocomplete.sh

   Save `~/.bashrc`.
   Open new terminal to use updated `~/.bashrc`.
