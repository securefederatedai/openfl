.. # Copyright (C) 2020-2021 Intel Corporation
.. # SPDX-License-Identifier: Apache-2.0

**************************
Activate Bash Autocomplete
**************************

Get faster access to available commands by activating bash completion in CLI mode.

STEP 1: Preparation
===================

Make sure you are inside a virtual environment with Open Federated Learning (|productName|) installed. See :ref:`install_package` for details.


STEP 2: Create the fx-autocomplete.sh Script
============================================

.. note::

    Perform this procedure if you don't have a **~/.fx-autocomplete.sh** script or if the existing **~/.fx-autocomplete.sh** script is corrupted.

1. Create the script.
   
   .. code-block:: console

      _FX_COMPLETE=bash_source fx > ~/.fx-autocomplete.sh

   
2. Check that the script was created properly.

   .. code-block:: console

      cat ~/.fx-autocomplete.sh

 The output should look like the example below (Click==8.0.1), but could be different depend on `Click <https://click.palletsprojects.com/en/8.0.x/>`_ version:
   
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


STEP 3: Activate the Autocomplete Feature
=========================================

Perform this command every time you open a new terminal window.

   .. code-block:: console

      source ~/.fx-autocomplete.sh


To save time, add the script into **.bashrc** so the script is activated when you log in.

1. Edit the **.bashrc** file. The **nano** command line editor is used in this example.

   .. code-block:: console

      nano ~/.bashrc

2. Add the script.

   .. code-block:: bash
   
      . ~/.fx-autocomplete.sh

3. Save your changes.

4. Open a new terminal to use the updated bash shell.

