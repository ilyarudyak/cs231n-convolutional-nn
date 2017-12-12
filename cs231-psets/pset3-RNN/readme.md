### Q1: Image Captioning with Vanilla RNNs (40 points)
The IPython notebook `RNN_Captioning.ipynb` will walk you through 
the implementation of an image captioning system on `MS-COCO` using 
vanilla recurrent networks.

### Q2: Image Captioning with LSTMs (35 points)
The IPython notebook `LSTM_Captioning.ipynb` will walk you through 
the implementation of Long-Short Term Memory (LSTM) RNNs, and 
apply them to image captioning on `MS-COCO`.

### Q3: Image Gradients: Saliency maps and Fooling Images (10 points)
The IPython notebook `ImageGradients.ipynb` will introduce the TinyImageNet 
dataset. You will use a pretrained model on this dataset to compute 
gradients with respect to the image, and use them to produce saliency 
maps and fooling images.

### Q4: Image Generation: Classes, Inversion, DeepDream (15 points)
In the IPython notebook `ImageGeneration.ipynb` you will use the pretrained 
TinyImageNet model to generate images. In particular you will generate 
class visualizations and implement feature inversion and DeepDream.

### Q5: Do something extra! (up to +10 points)
Given the components of the assignment, try to do something cool. 
Maybe there is some way to generate images that we did not implement 
in the assignment?

### misc

hdf5 has the following notes:
    Mac users may need to set the environment variable "HDF5_USE_FILE_LOCKING" to the
    five-character string "FALSE" when accessing network mounted files.  This is an
    application run-time setting, not a configure or build setting.  Otherwise errors
    such as "unable to open file" or "HDF5 error" may be  encountered.