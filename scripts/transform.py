import os
import numpy as np
from PIL import Image
import cv2

colors = [
    (0, 0, 0),
    (255, 182, 193),
    (255, 0, 0),
    (139, 0, 0),
    (173, 216, 230),
    (65, 105, 225),
    (0, 0, 128),
    (144, 238, 144),
    (124, 252, 0),
    (0, 100, 0),
    (255, 255, 224),
    (255, 215, 0),
    (255, 250, 205),
    (238, 130, 238),
    (138, 43, 226),
    (128, 0, 128),
    (255, 255, 255),
    (211, 211, 211),
	(128, 128, 128),
    (255, 165, 0),
    (165, 42, 42),
    (255, 192, 203),
    (0, 255, 255),
]

def transform_segments(source_dir, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    for filename in os.listdir(source_dir):
        print(filename)
        source_path = os.path.join(source_dir, filename)

        image = np.asarray(Image.open(source_path).convert('RGB'), dtype=np.uint8)
        result = np.zeros_like(image, dtype=np.uint8)
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                index = image[i, j, 0] // 3 if image[i, j, 0] % 3 == 0 else 0
                color = colors[index]
                result[i, j] = [c for c in color]

        save_path = os.path.join(save_dir, filename)
        cv2.imwrite(save_path, result)


if __name__ == '__main__':
    transform_segments("/home/liudongxv/workplace/GANInverter-dev/edit_data/seg_bisenet", "/home/liudongxv/workplace/GANInverter-dev/edit_data/seg_bisenet_transform")