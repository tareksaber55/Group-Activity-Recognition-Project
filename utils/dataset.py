import os
from torch.utils.data import Dataset

class ImageLevelDataset(Dataset):
    def __init__(self, input_root, labels_root):
        videos_dir = os.listdir(input_root)
        videos_dir.sort()

        self.idx_to_output = {}
        i = 0

        for video_dir in videos_dir:
            video_path = os.path.join(input_root, video_dir)

            if not os.path.isdir(video_path):
                continue

            clips_dir = os.listdir(video_path)
            clips_dir.sort()

            for clip_dir in clips_dir:
                clip_path = os.path.join(video_path, clip_dir)

                if not os.path.isdir(clip_path):
                    continue

                input_path = os.path.join(input_root, video_dir, clip_dir, f'{clip_dir}.npy')
                label_path = os.path.join(labels_root, video_dir, clip_dir, f'{clip_dir}.json')

                self.idx_to_output[i] = (input_path, label_path)
                i += 1
    



