Running steps:
1) Download and extract data to any folder (`$DATA_PATH`). The output of `tree $DATA_PATH -L 2`:
```
.
├── MICCAI_BraTS_2019_Data_Training
│   ├── HGG
│   ├── LGG
│   ├── name_mapping.csv
│   └── survival_data.csv
```
To use a `tree` command, you have to install it first: `sudo apt-get install tree`

2) Choose a subfolder (`$SUBFOLDER`) corresponding to scan subset: 
    - `HGG`: glioblastoma scans
    - `LGG`: lower grade glioma scans
 
Let's pick `HGG`: `export SUBFOLDER=MICCAI_BraTS_2019_Data_Training/HGG`. The learning rate has been already tuned for this task, so you don't have to change it. If you pick `LGG`, all the next steps will be the same.

3) In order for each collaborator to use separate slice of data, we split main folder into `n` subfolders:
```bash
#!/bin/bash
cd $DATA_PATH/$SUBFOLDER

n=2  # Set this to the number of directories you want to create

# Get a list of all files and shuffle them
files=($(ls | shuf))

# Create the target directories if they don't exist
for ((i=0; i<n; i++)); do
  mkdir -p "$i"
done

# Split the files between the directories
for i in "${!files[@]}"; do
  target_dir=$((i % n))
  mv "${files[$i]}" "$target_dir/"
done
```
Output of `tree $DATA_PATH/$SUBFOLDER -L 1` in case when `n = 2`:
```
.
├── 0
└── 1
```
If BraTS20 has the same structure, we can split it in the same way.
Each slice contains subdirectories containing `*.nii.gz` files. According to `load_from_NIfTI` function [docstring](https://github.com/intel/openfl/blob/2e6680fedcd4d99363c94792c4a9cc272e4eebc0/openfl-workspace/tf_2dunet/src/brats_utils.py#L68), `NIfTI files for whole brains are assumed to be contained in subdirectories of the parent directory`. So we can use these slice folders as collaborator data paths.

4) We are ready to train! Try executing the [Quick Start](https://openfl.readthedocs.io/en/latest/get_started/quickstart.html) steps. Make sure you have `openfl` installed in your Python virtual environment. Be sure to set the proper collaborator data paths in [plan/data.yaml](https://github.com/securefederatedai/openfl/blob/develop/openfl-workspace/tf_2dunet/plan/data.yaml) and during the `fx collaborator create` command. Alternatively, you can run a quick test with our 'Hello Federation' script:

```bash
python tests/github/test_hello_federation.py tf_2dunet fed_work12345alpha81671 one123dragons beta34unicorns localhost --col1-data-path ../$DATA_PATH/$SUBFOLDER/0 --col2-data-path ../$DATA_PATH/$SUBFOLDER/1 --rounds-to-train 5
```
The result of the execution of the command above is 5 completed training rounds. 
