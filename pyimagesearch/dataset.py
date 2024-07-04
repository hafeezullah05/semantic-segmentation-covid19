# import the necessary packages
from torch.utils.data import Dataset
import cv2

#print("in dataset.py")

class SegmentationDataset(Dataset):
	def __init__(self, imagePaths, maskPaths, transforms):
		# store the image and mask filepaths, and augmentation
		# transforms
		self.imagePaths = imagePaths
		self.maskPaths = maskPaths
		self.transforms = transforms

	def __len__(self):
		# return the number of total samples contained in the dataset
		return len(self.imagePaths)

	def __getitem__(self, idx):
		# grab the image path from the current index
		imagePath = self.imagePaths[idx]

		# load the image from disk, swap its channels from BGR to RGB,
		# and read the associated mask from disk in grayscale mode
        # When the image file is read with the OpenCV function imread(), the order of colors is BGR (blue, green, red). On the other hand,
        # in Pillow, the order of colors is assumed to be RGB (red, green, blue).
        # therefore, if you want to use both the Pillow function and the OpenCV function, you need to convert BGR and RGB.
        # we can use the OpenCV function cvtColor() to convert to RGB


		image = cv2.imread(imagePath)
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		mask  = cv2.imread(self.maskPaths[idx], 0)

		#############################################################################
		#originalImage = cv2.imread(imagePath) # Here 0 can be added for greyscale
		#image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		#original image converting to grayscaling for input images and masks
		#greyImage = cv2.cvtColor(originalImage, cv2.COLOR_BGR2GRAY)
		#greyMask  = cv2.imread(self.maskPaths[idx], 0) #using 0 to open in greyscale mode
		#optional ranging [0-1] rather than 0-256
		#mask[mask == 255.0] = 1.0
		#(thresh, image) = cv2.threshold(greyImage, 0, 1, cv2.THRESH_BINARY)
		#(_thresh, mask) = cv2.threshold(greyMask, 0, 1, cv2.THRESH_BINARY)
		#############################################################################

		# check to see if we are applying any transformations
		if self.transforms is not None:
			# apply the transformations to both image and its mask
			image = self.transforms(image)
			mask = self.transforms(mask)

		# return a tuple of the image and its mask
		return (image, mask)