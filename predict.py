import argparse
import logging
import os
import numpy as np
import torch
from PIL import Image
import cv2
from utils.data_loading import BasicDataset
from unet import UNet
from utils.utils import process_image
from utils.plot import plot_detected_points


def predict_img(net, full_img, device, scale_factor=1, out_threshold=0.5):
    net.eval()
    img, _ = BasicDataset.preprocess([0, 255], full_img, full_img, scale_factor)
    img = torch.from_numpy(img).unsqueeze(0).to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img).cpu()
        mask = (output.argmax(dim=1) if net.n_classes > 1
                else (torch.sigmoid(output) > out_threshold))
    return mask[0].long().squeeze().numpy()


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', '-m', default='checkpoints/checkpoint_epoch1468.pth', metavar='FILE',
                        help='Specify the file in which the model is stored')
    parser.add_argument('--mask-threshold', '-t', type=float, default=0.5, help='Minimum mask threshold')
    parser.add_argument('--scale', '-s', type=float, default=0.25, help='Scale factor for input images')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')
    parser.add_argument('--folder', '-f', type=str, default='p6', help='Folder for input data')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')

    return parser.parse_args()


def main():
    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    path = os.path.join('data/test', args.folder)
    in_files = sorted(os.path.join(path, name) for name in os.listdir(path))
    out_files = sorted(os.path.join('data/pred', name) for name in os.listdir(path))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Loading model {args.model} on {device}')

    net = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear).to(device)
    state_dict = torch.load(args.model, map_location=device)
    net.load_state_dict(state_dict)
    logging.info('Model loaded!')

    lk_params = dict(winSize=(30, 30), maxLevel=4, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    final_points = []

    old_gray, corners = None, None

    for i, filename in enumerate(in_files):
        logging.info(f'Predicting image {filename} ...')

        img = Image.open(filename)
        mask = predict_img(net, img, device, scale_factor=args.scale, out_threshold=args.mask_threshold)

        binary_mask, frame_gray, img = process_image(img, mask, args.scale, w_beam=455)

        if i == 0:
            corners = cv2.goodFeaturesToTrack(binary_mask, 150, 0.01, 30)
        else:
            corners_new, st, _ = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, corners, None, **lk_params)
            good_new = corners_new[st == 1]
            corners = good_new.reshape(-1, 1, 2)

        final_points.append(corners[:, 0, :])
        plot_detected_points(np.array(img), corners, cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR), out_files[i])
        old_gray = frame_gray.copy()

    torch.save(torch.Tensor(final_points), f'data/points/position_points_{args.folder}.pt')


if __name__ == '__main__':
    main()
