import os
from torch.utils.data import Dataset
import pickle

class ImageLevelDataset(Dataset):
    def __init__(self, input_root, annot_pkl):
        with open(f'/kaggle/working/annot_all.pkl', 'rb') as file:
            videos_annot = pickle.load(file)
        
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
                label_path = videos_annot[video_dir][clip_dir]['category']

                self.idx_to_output[i] = (input_path, label_path)
                i += 1
    







def temp():
    categories_dct = {
        'l-pass': 0,
        'r-pass': 1,
        'l-spike': 2,
        'r_spike': 3,
        'l_set': 4,
        'r_set': 5,
        'l_winpoint': 6,
        'r_winpoint': 7
    }

    train_ids = ["1", "3", "6", "7", "10", "13", "15", "16", "18", "22", "23", "31",
                 "32", "36", "38", "39", "40", "41", "42", "48", "50", "52", "53", "54"]


    val_ids = ["0", "2", "8", "12", "17", "19", "24", "26", "27", "28", "30", "33", "46", "49", "51"]

