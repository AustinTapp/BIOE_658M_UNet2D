# UNet-2D tutorial
2D Unet walkthrough for segmenting images of you choice.

The code will not run right away. Many aspects should be updated ensure the training runs smoothly.
After training, it will be necessary to load model weights in order to make a prediction.

For more assistance, view the original MONAI tutorial, here: https://github.com/Project-MONAI/tutorials/blob/main/2d_segmentation/torch/unet_training_dict.py

Ensure all installation requirements are satisfied.
Requirements are:
python = "3.10"
pytorch-lightning = "1.7.3"
monai = "0.9.1"
jsonargparse = "4.13.0"
nibabel = "4.0.1"
scikit-image = "0.19.3"
