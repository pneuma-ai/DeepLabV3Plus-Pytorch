import cv2, os
import imgaug as ia
import imgaug.augmenters as iaa
from tqdm import tqdm
from PIL import Image
from imgaug.augmentables.segmaps import SegmentationMapsOnImage

IMG_PATH = "../datasets/data/VOCdevkit/VOC2012/JPEGImages"
GT_PATH = "../datasets/data/VOCdevkit/VOC2012/SegmentationClass"
FNAMES= os.listdir(IMG_PATH)

ia.seed(1)

# Example batch of images.
# The array has shape (32, 64, 64, 3) and dtype uint8.

for f in tqdm(FNAMES):
    file = f[:-4]
    image = cv2.imread(os.path.join(IMG_PATH, f"{file}.jpg"))
    segmaps= cv2.imread(os.path.join(GT_PATH, f"{file}.png"))
    segmaps = SegmentationMapsOnImage(segmaps, shape=image.shape)

    seq = iaa.Sequential([
        iaa.Dropout([0.05, 0.2]),      # drop 5% or 20% of all pixels
        iaa.Sharpen((0.0, 1.0)),       # sharpen the image
        iaa.Fliplr(0.5), # horizontal flips
        iaa.Flipud(0.5),
        iaa.Crop(percent=(0, 0.1)), # random crops
        # Small gaussian blur with random sigma between 0 and 0.5.
        # But we only blur about 50% of all images.
        iaa.Sometimes(
            0.5,
            iaa.GaussianBlur(sigma=(0, 0.5)),
        ),
        # Strengthen or weaken the contrast in each image.
        iaa.LinearContrast((0.75, 1.5)),
        # Add gaussian noise.
        # For 50% of all images, we sample the noise once per pixel.
        # For the other 50% of all images, we sample the noise per pixel AND
        # channel. This can change the color (not only brightness) of the
        # pixels.
        iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
        # Make some images brighter and some darker.
        # In 20% of all cases, we sample the multiplier once per channel,
        # which can end up changing the color of the images.
        iaa.Multiply((0.8, 1.2), per_channel=0.2),
        # Apply affine transformations to each image.
        # Scale/zoom them, translate/move them, rotate them and shear them.
        iaa.Affine(
            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
            translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
            rotate=(-25, 25),
            shear=(-8, 8)
        )
    ], random_order=True) # apply augmenters in random order

    for i in range(100):
        image_aug, segmap_aug = seq(image=image, segmentation_maps=segmaps)
        img = Image.fromarray(image_aug)
        img.save(os.path.join(IMG_PATH+"Aug", f"{file}_aug_{i}.jpg"))
        seg = Image.fromarray(segmap_aug.get_arr()).convert("L")
        seg.save(os.path.join(GT_PATH+"Aug",f"{file}_aug_{i}.png"))