# SHENet: a single-horizon disease evolution network

We propose a single-horizon disease evolution network (SHENet) to predictively generate post-therapeutic SD-OCT images by inputting pre-therapeutic SD-OCT images with neovascular age-related macular degeneration (nAMD). 
In SHENet, a feature encoder converts the input SD-OCT images to deep features, then a graph evolution module predicts the process of disease evolution in high-dimensional latent space and outputs the predicted deep features, and lastly feature decoder recovers the predicted deep features to SD-OCT images. 
We further propose evolution reinforcement module to ensure the effectiveness of disease evolution learning and further obtain the realistic SD-OCT images by adversarial training.

## The model architecture is shown as follows:
![Image text](https://github.com/ZhangYH0502/SDENet/blob/main/f2.png)

## The predictive results are shown as follows:
![Image text](https://github.com/ZhangYH0502/SDENet/blob/main/f4.png)

Main File Configs: <br>
* train.py: the main file to run for training the model; <br>
* test.py: test the trained model; <br>
* patchGAN_discriminator.py: discriminator model; <br>
* basic_unet.py: generator model; <br>
* loss.py: loss function; <br>
* Dataset.py: a dataloader to read data. <br>

<br>

How to run our code: <br>
* prepare your data with the paired images; <br>
* modify the data path and reading mode in Dataset.py; <br>
* run train.py to train the model; <br>
* run test.py to test the trained model.
