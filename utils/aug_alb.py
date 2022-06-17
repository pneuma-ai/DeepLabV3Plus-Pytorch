import cv2, random, os
import albumentations as A
from PIL import Image
from tqdm import tqdm

transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
])

IMG_PATH = "../datasets/data/VOCdevkit/VOC2012/JPEGImages"
GT_PATH = "../datasets/data/VOCdevkit/VOC2012/SegmentationClass"
FNAMES = os.listdir(IMG_PATH)

for f in tqdm(FNAMES):
    file = f[:-4]
    image = cv2.imread(os.path.join(IMG_PATH, f"{file}.jpg"))
    mask= cv2.imread(os.path.join(GT_PATH, f"{file}.png"))

    for i in range(1):
        transformed = transform(image=image, mask=mask)
        img = Image.fromarray(transformed['image'])
        img.save(os.path.join(IMG_PATH+"Aug", f"{file}_aug_{i}.jpg"))
        seg = Image.fromarray(transformed['mask']).convert("L")
        seg.save(os.path.join(GT_PATH+"Aug",f"{file}_aug_{i}.png"))

