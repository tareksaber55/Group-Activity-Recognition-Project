import os
import numpy as np
import cv2
import torch
from PIL import __version__ as PILLOW_VERSION

import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from volleyball_annot_loader import load_tracking_annot

dataset_root = '/kaggle/input/datasets/ahmedmohamed365/volleyball'



def prepare_model(image_level = False):
    if image_level:
        # image has a lot of space around objects. Let's crop around
        preprocess = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        # already croped box. Don't crop more
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    # Check if a GPU is available and if not, use a CPU

    # resnet50 alexnet
    model = models.resnet50(pretrained=True)  # You can also use 'mobilenet_v3_large'

    # Remove the classification head (i.e., the fully connected layers)
    model = nn.Sequential(*(list(model.children())[:-1]))


    # Set the model to evaluation mode
    model.eval()

    return model, preprocess


def extract_features(clip_dir_path, annot_file, output_file, model, preprocess, image_level=False):
    frame_boxes = load_tracking_annot(annot_file)

    with torch.no_grad():
        for frame_id, boxes_info in frame_boxes.items():
            try:
                img_path = os.path.join(clip_dir_path, f'{frame_id}.jpg')
                image = Image.open(img_path).convert('RGB')

                if image_level:
                    preprocessed_image = preprocess(image).unsqueeze(0) # unsqueeze(dim) : adds a dimension of size 1 at position 0, essential for torch.cat
                    dnn_repr = model(preprocessed_image)
                    dnn_repr = dnn_repr.view(1, -1)
                else:
                    # for each image player's box, extract cropped images, extract features
                    preprocessed_images = []
                    for box_info in boxes_info:
                        x1, y1, x2, y2 = box_info.box
                        cropped_image = image.crop((x1, y1, x2, y2))

                        if False:   # visualize a crop
                            cv2.imshow('Cropped Image', np.array(cropped_image))
                            cv2.waitKey(0)

                        preprocessed_images.append(preprocess(cropped_image).unsqueeze(0))

                    preprocessed_images = torch.cat(preprocessed_images)
                    dnn_repr = model(preprocessed_images)    # Batch Processing
                    dnn_repr = dnn_repr.view(len(preprocessed_images), -1)  # 12 x 2048 for resnet 50

                # uncomment to save features
                np.save(output_file, dnn_repr.numpy())
            except Exception as e:
                print(f"An error occurred: {e}")


if __name__ == '__main__':


    # image_level: extract features for the whole image or just a crop
    image_level = False
    model, preprocess = prepare_model(image_level)

    videos_root = f'{dataset_root}/volleyball_/videos'
    annot_root = f'{dataset_root}/volleyball_tracking_annotation/volleyball_tracking_annotation'
    output_root = f'/kaggle/working/features/image-level/resnet'

    videos_dirs = os.listdir(videos_root)
    videos_dirs.sort()

    # Iterate on each video and for each video iterate on each clip
    for idx, video_dir in enumerate(videos_dirs):
        video_dir_path = os.path.join(videos_root, video_dir)

        if not os.path.isdir(video_dir_path):
            continue

        print(f'{idx}/{len(videos_dirs)} - Processing Dir {video_dir_path}')

        clips_dir = os.listdir(video_dir_path)
        clips_dir.sort()

        for clip_dir in clips_dir:
            clip_dir_path = os.path.join(video_dir_path, clip_dir)

            if not os.path.isdir(clip_dir_path):
                continue

            print(f'\t{clip_dir_path}')

            annot_file = os.path.join(annot_root, video_dir, clip_dir, f'{clip_dir}.txt')
            output_file = os.path.join(output_root, video_dir)

            if not os.path.exists(output_file):
                os.makedirs(output_file)

            output_file = os.path.join(output_file, f'{clip_dir}.npy')
            extract_features(clip_dir_path, annot_file, output_file, model, preprocess, image_level = image_level)
