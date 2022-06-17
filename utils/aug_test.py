import cv2, os, imageio, random
import imgaug as ia
from imgaug.augmentables.segmaps import SegmentationMapsOnImage

IMG_PATH = "./datasets/data/VOCdevkit/VOC2012/JPEGImagesAug"
GT_PATH = "./datasets/data/VOCdevkit/VOC2012/SegmentationClassAug"
files = os.listdir(IMG_PATH)
FNAME= files[random.randrange(len(files))]
print(FNAME)

ia.seed(1)

file = FNAME[:-4]
image = cv2.imread(os.path.join(IMG_PATH, f"{file}.jpg"))
segmaps= cv2.imread(os.path.join(GT_PATH, f"{file}.png"))
segmaps = SegmentationMapsOnImage(segmaps, shape=image.shape)

cells = [image, segmaps.draw_on_image(image)[0]]
grid_image = ia.draw_grid(cells, cols=2)
imageio.imwrite(f"{file}.jpg", grid_image)