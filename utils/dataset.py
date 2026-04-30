import os
from torch.utils.data import Dataset
import pickle
from PIL import Image

class ImageLevelDataset(Dataset):
    def __init__(self, input_root, annot_pkl_path, categories_dict, videos_ids, preprocess):
        self.preprocess = preprocess

        with open(annot_pkl_path, 'rb') as file:
            videos_annot = pickle.load(file)

        # this make the search O(1)
        videos_ids_set = set(videos_ids)

        self.samples = []

        for video in videos_annot:
            if video not in videos_ids_set:
                continue

            for clip in videos_annot[video]:
                clip_dict = videos_annot[video][clip]

                category = categories_dict[clip_dict['category']]

                for frame_id in clip_dict['frame_boxes_dct']:
                    input_image_path = os.path.join(
                        input_root, video, clip, f'{frame_id}.jpg'
                    )

                    self.samples.append({
                        'input_image_path': input_image_path,
                        'category': category
                    })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]

        image = Image.open(sample['input_image_path']).convert('RGB')
        image = self.preprocess(image)

        category = sample['category']

        return image, category


                    
class PersonLevelDataset(Dataset):
    pass






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

