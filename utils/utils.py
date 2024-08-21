import matplotlib.pyplot as plt
import wandb
import torch
from torchvision.utils import make_grid




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


def log_inference_example(model, dataloader, device):
    model.eval()
    for batch in dataloader:
        inputs = batch['image'].to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
        with torch.no_grad():
            outputs = model(inputs)
        # Selecciona la primera imagen del batch para inferencia
        input_image = inputs[0].cpu()
        output_image = outputs[0].cpu()
        cosi = (output_image.argmax(dim=0)).long().squeeze()

        # grid = make_grid([input_image[0, :, :], cosi], nrow=2)
        grid = make_grid([input_image[0, :, :].unsqueeze(0), cosi.unsqueeze(0)], nrow=2).numpy() * 255

        return grid.transpose(1, 2, 0)
