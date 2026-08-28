import os
import argparse
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from util.crop import center_crop_arr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to ImageNet root directory')
    parser.add_argument('--output_path', type=str, default='imagenet-train-256',
                        help='Folder where transformed images will be saved')
    parser.add_argument('--img_size', type=int, default=256,
                        help='Resolution to center-crop and resize')
    args = parser.parse_args()

    transform_train = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.img_size)),
        transforms.CenterCrop(args.img_size),
        transforms.ToTensor(),
    ])

    dataset_train = datasets.ImageFolder(
        os.path.join(args.data_path, 'train'),
        transform=transform_train
    )

    data_loader = DataLoader(
        dataset_train,
        batch_size=256,
        num_workers=32,
        shuffle=False,
        pin_memory=False
    )

    os.makedirs(args.output_path, exist_ok=True)

    to_pil = transforms.ToPILImage()
    global_idx = 0

    from tqdm import tqdm
    for batch_images, batch_labels in tqdm(data_loader):
        for i in range(batch_images.size(0)):
            img_tensor = batch_images[i]

            pil_img = to_pil(img_tensor)
            out_path = os.path.join(
                args.output_path,
                f"transformed_{global_idx:08d}.png"
            )
            pil_img.save(out_path, format='PNG', compress_level=0)
            global_idx += 1

        print(f"Saved batch up to index={global_idx} ...")

    print("Finished saving all images.")


if __name__ == "__main__":
    main()