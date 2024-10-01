import os
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt



def get_output_filenames(args):
    def _generate_name(fn):
        return f'{os.path.splitext(fn)[0]}_OUT.png'

    return args.output or list(map(_generate_name, args.input))



def mask_to_image(mask: np.ndarray, mask_values):
    if isinstance(mask_values[0], list):
        out = np.zeros((mask.shape[-2], mask.shape[-1], len(mask_values[0])), dtype=np.uint8)
    elif mask_values == [0, 1]:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=bool)
    else:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=np.uint8)

    if mask.ndim == 3:
        mask = np.argmax(mask, axis=0)

    for i, v in enumerate(mask_values):
        out[mask == i] = v

    return Image.fromarray(out)


def process_image(img, mask, scale_factor, w_beam):
    mask = mask[:, :w_beam]
    binary_mask = mask.astype(np.uint8) * 255
    binary_mask = cv2.erode(binary_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    binary_mask = cv2.GaussianBlur(binary_mask, (3, 3), 0)

    img_resized = cv2.resize(np.array(img), None, fx=scale_factor, fy=scale_factor)
    frame_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

    return binary_mask, frame_gray, img_resized