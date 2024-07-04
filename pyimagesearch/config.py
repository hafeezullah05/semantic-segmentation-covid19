# import the necessary packages
import torch
import os

#print("in config.py")

# base path of the dataset
DATASET_PATH = os.path.join(os.getcwd(), "covid19-ct-scans/dataset","train")


# define the path to the images and masks dataset
#IMAGE_DATASET_PATH = os.path.join(DATASET_PATH, "images")
#MASK_DATASET_PATH = os.path.join(DATASET_PATH, "masks")
# define the path to the images and masks dataset 
IMAGE_DATASET_PATH = os.path.join(DATASET_PATH, "images")
MASK_DATASET_PATH = os.path.join(DATASET_PATH, "masks")

# define the test split
TEST_SPLIT = 0.20

# determine the device to be used for training and evaluation
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# determine if we will be pinning memory during data loading
PIN_MEMORY = True if DEVICE == "cuda" else False

# define the number of channels in the input, number of classes,
# and number of levels in the U-Net model
NUM_CHANNELS = 1
NUM_CLASSES = 1
NUM_LEVELS = 3

# initialize learning rate, number of epochs to train for, and the
# batch size
INIT_LR = 0.001
NUM_EPOCHS = 75
BATCH_SIZE = 64

# define the input image dimensions
INPUT_IMAGE_WIDTH = 128
INPUT_IMAGE_HEIGHT = 128

# define threshold to filter weak predictions
THRESHOLD = 0.5

# define the path to the base output directory
BASE_OUTPUT = os.path.join(DATASET_PATH, "output")

# define the path to the output serialized model, model training
# plot, and testing image paths
MODEL_PATH = os.path.join(BASE_OUTPUT, "unet_covid-19.pth")
LOSS_PLOT_PATH = os.path.sep.join([BASE_OUTPUT, "loss_plot.png"])
ACCURACY_PLOT_PATH = os.path.sep.join([BASE_OUTPUT, "accuracy_plot.png"])
IoU_PLOT_PATH = os.path.sep.join([BASE_OUTPUT, "IoU_plot.png"])
DICE_PLOT_PATH = os.path.sep.join([BASE_OUTPUT, "dice_plot.png"])
Mean_DICE_PLOT_PATH = os.path.sep.join([BASE_OUTPUT, "mean_dice_plot.png"])

TEST_PATHS = os.path.sep.join([BASE_OUTPUT, "test_paths.txt"])