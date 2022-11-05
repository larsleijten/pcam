# Several PCAM solutions

This repository was built as a small demonstration of several deep learning solutions for the PatchCamelyon dataset. This dataset was selected due to its limited image size, which allows the use of  Google Colab GPUs for training of several models. 

This [Github Repository](https://github.com/larsleijten/pcam "larsleijten/pcam") contains the code that was used to train and test the models. The methods and results are demonstrated in the Notebook: **Demonstration.ipynb**

All models are trained, validated and tested on the [PatchCamelyon](https://github.com/basveeling/pcam) dataset. To correct an MD5 Checksum error, the names of the validation and test set files were switched. These files were loaded in the root folder before regular data loading using the [Torchvision](https://pytorch.org/vision/stable/generated/torchvision.datasets.PCAM.html) dataset.

