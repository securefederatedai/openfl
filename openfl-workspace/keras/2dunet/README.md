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
 
Let's pick `HGG`: `export SUBFOLDER=HGG`. The learning rate has been already tuned for this task, so you don't have to change it. If you pick `LGG`, all the next steps will be the same.

3) In order for each collaborator to use separate slice of data, we split main folder into `n` subfolders:
```bash
cd $DATA_PATH/$SUBFOLDER
i=0; 
for f in *; 
do 
    d=dir_$(printf $((i%n)));  # change n to number of data slices (number of collaborators in federation)
    mkdir -p $d; 
    mv "$f" $d; 
    let i++; 
done
```
Output of `tree $DATA_PATH/$SUBFOLDER -L 1` in case when `n = 2`:
```
.
├── 0
└── 1
```
If BraTS20 has the same structure, we can split it in the same way.
Each slice contains subdirectories containing `*.nii.gz` files. According to `load_from_NIfTI` function [docstring](https://github.com/securefederatedai/openfl/blob/2e6680fedcd4d99363c94792c4a9cc272e4eebc0/openfl-workspace/tf_2dunet/src/brats_utils.py#L68), `NIfTI files for whole brains are assumed to be contained in subdirectories of the parent directory`. So we can use these slice folders as collaborator data paths.

4) We are ready to train! Try executing the [Hello Federation](https://openfl.readthedocs.io/en/latest/running_the_federation.baremetal.html#hello-federation-your-first-federated-learning-training) steps. Make sure you have `openfl` installed in your Python virtual environment. All you have to do is to specify collaborator data paths to slice folders. We have combined all 'Hello Federation' steps in a single bash script, so it is easier to test:
```bash
bash tests/github/test_hello_federation.sh tf_2dunet fed_work12345alpha81671 one123dragons beta34unicorns localhost --col1-data-path $DATA_PATH/MICCAI_BraTS_2019_Data_Training/$SUBFOLDER/0 --col2-data-path $DATA_PATH/MICCAI_BraTS_2019_Data_Training/$SUBFOLDER/1 --rounds-to-train 5
```
The result of the execution of the command above is 5 completed training rounds. 
