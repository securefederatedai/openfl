# Implementation of Hipp Mapper for Medical Imaging

This project implements a hippocampal mapper for medical imaging. It works similarly to other workspaces as described in the [OpenFL Taskrunner Tutorial](https://openfl.readthedocs.io/en/latest/tutorials/taskrunner.html).

## Setup Instructions

To set up the data, run the following command:
```bash
python src/setup_data.py --num_collaborators $MAX_NUMBER_OF_COLLABORATORS --total_dataset_size_per_col_MB $DESIRED_DATASET_SIZE
```

## Known Issues

An error has been observed on 2nd Generation Intel® Xeon® Scalable Processors (codenamed Cascade Lake): "CPU implementation of Conv3D currently only supports the NHWC tensor format." 

### Workarounds:
1. Use a more recent CPU, such as 3rd Generation Intel® Xeon® Scalable Processors (codenamed Ice Lake), which supports the required tensor format.
2. Alternatively, use a GPU to avoid this issue entirely.

## Reference

Goubran, M., Ntiri E., Akhavein, H., Holmes, M., Nestor, S., Ramirez, J., Adamo, S., Gao, F., Ozzoude, M., Scott, C., Martel, A., Swardfager, W., Masellis, M., Swartz, R., MacIntosh B, and Black, SE. “Hippocampal segmentation for atrophied brains using three-dimensional convolutional neural networks”. Human Brain Mapping. 2020 Feb 1;41(2):291-308. https://onlinelibrary.wiley.com/doi/full/10.1002/hbm.24811