import cv2
import logging
import matplotlib.pyplot as plt



def plot_img_and_mask(img, mask):
    classes = mask.max() + 1
    fig, ax = plt.subplots(1, classes + 1)
    ax[0].set_title('Input image')
    ax[0].imshow(img)
    for i in range(classes):
        ax[i + 1].set_title(f'Mask (class {i + 1})')
        ax[i + 1].imshow(mask == i)
    plt.xticks([]), plt.yticks([])
    plt.show()

def plot_detected_points(img, corners, output_image2, out_name):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Dibujar las esquinas en la imagen original
    for corner in corners:
        x, y = corner.ravel()
        cv2.circle(output_image2, (int(x), int(y)), 4, (0, 0, 255), -1)
        cv2.circle(img, (int(x), int(y)), 4, (0, 0, 255), -1)

    output_image = cv2.hconcat([img, output_image2])
    cv2.imwrite(out_name, output_image)
    logging.info(f'Mask saved to {out_name}')