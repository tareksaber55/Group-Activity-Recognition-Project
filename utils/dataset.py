import os
from torch.utils.data import Dataset
import pickle
from PIL import Image
from utils import boxinfo
import sys
import torch
sys.modules['boxinfo'] = boxinfo  # make pickle find it as 'boxinfo'
class ImageLevelDataset(Dataset):
    def __init__(
        self,
        input_root,
        annot_pkl_path,
        categories_dict,
        videos_ids,
        preprocess,
        one_frame=True
    ):
        self.preprocess = preprocess
        self.one_frame = one_frame

        with open(annot_pkl_path, 'rb') as file:
            videos_annot = pickle.load(file)

        videos_ids_set = set(videos_ids)

        self.samples = []

        for video in videos_annot:
            if video not in videos_ids_set:
                continue

            for clip in videos_annot[video]:
                clip_dict = videos_annot[video][clip]

                category = categories_dict[clip_dict['category']]

                sorted_frame_ids = sorted(clip_dict['frame_boxes_dct'].keys())

                if one_frame:
                    frame_id = sorted_frame_ids[4]

                    image_path = os.path.join(
                        input_root,
                        video,
                        clip,
                        f'{frame_id}.jpg'
                    )

                    self.samples.append({
                        'image_path': image_path,
                        'category': category
                    })

                else:
                    frames = []

                    for frame_id in sorted_frame_ids:
                        image_path = os.path.join(
                            input_root,
                            video,
                            clip,
                            f'{frame_id}.jpg'
                        )

                        frames.append(image_path)

                    self.samples.append({
                        'sequence_path': frames,
                        'category': category
                    })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        category = sample['category']

        if self.one_frame:
            image_path = sample['image_path']

            image = Image.open(image_path).convert('RGB')

            if self.preprocess:
                image = self.preprocess(image)

            return image, category

        else:
            images = []

            for frame_path in sample['sequence_path']:
                image = Image.open(frame_path).convert('RGB')

                if self.preprocess:
                    image = self.preprocess(image)

                images.append(image)

            images = torch.stack(images)

            return images, category
        


                    
class PlayerLevelDataset(Dataset):
    def __init__(
        self,
        input_root,
        annot_pkl_path,
        categories_dict,
        videos_ids,
        preprocess,
        max_players = 12,
        one_frame = True
    ):
        self.max_players = max_players
        self.categories_dict = categories_dict
        self.preprocess = preprocess
        self.one_frame = one_frame
        with open(annot_pkl_path,'rb') as file:
            videos_annot = pickle.load(file)
        videos_ids_set = set(videos_ids) # Set make search O(1)
        self.samples = []

        for video in videos_annot:
            if video not in videos_ids_set:
                continue
            for clip in videos_annot[video]:
                clip_dict = videos_annot[video][clip]
                sorted_frame_ids = sorted(clip_dict['frame_boxes_dct'].keys())
                if one_frame:
                    frame_id = sorted_frame_ids[4]
                    frame_path = os.path.join(input_root,video,clip,f'{frame_id}.jpg')
                    self.samples.append({'image_path':frame_path,
                                        'frame_boxes':clip_dict['frame_boxes_dct'][frame_id]})
                else:
                    frames_path = []
                    frames_boxes = []
                    for frame_id in sorted_frame_ids:
                        frame_path = os.path.join(input_root,video,clip,f'{frame_id}.jpg')
                        frame_boxes = clip_dict['frame_boxes_dct'][frame_id]
                        frames_path.append(frame_path)
                        frames_boxes.append(frame_boxes)
                    self.samples.append({'frames_path':frames_path,
                                        'frames_boxes':frames_boxes})
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        sample = self.samples[index]
        if self.one_frame:
            image = Image.open(sample['image_path']).convert('RGB')
            preprocessed_images = []
            categories = []
            for box_info in sample['frame_boxes']:
                x1, y1, x2, y2 = box_info.box
                cropped_image = image.crop((x1,y1,x2,y2))
                preprocessed_images.append(self.preprocess(cropped_image))
                categories.append(self.categories_dict[box_info.category])
            num_players = len(preprocessed_images)
            preprocessed_images = torch.stack(preprocessed_images)
            categories = torch.tensor(categories,dtype=torch.long)
            if num_players < self.max_players:
                pad_count = self.max_players - num_players
                C,H,W = preprocessed_images.shape[1:]
                image_paading = torch.zeros(pad_count,C,H,W)
                preprocessed_images = torch.cat(
                    [preprocessed_images,image_paading],dim=0
                )
                label_padding = torch.full((pad_count,),-1,dtype=torch.long)
                categories = torch.cat(
                    [categories,label_padding],dim=0
                )
            elif num_players > self.max_players:
                preprocessed_images = preprocessed_images[:self.max_players]
                categories = categories[:self.max_players]
            return preprocessed_images , categories
        else:
            all_frames_images = []
            all_frames_categories = []
            for frame_path,frame_boxes in zip(sample['frames_path'],sample['frames_boxes']):
                image = Image.open(frame_path).convert('RGB')
                preprocessed_images = []
                categories = []
                for box_info in frame_boxes:
                    x1, y1, x2, y2 = box_info.box
                    cropped_image = image.crop((x1,y1,x2,y2))
                    preprocessed_images.append(self.preprocess(cropped_image))
                    categories.append(self.categories_dict[box_info.category])
                num_players = len(preprocessed_images)
                preprocessed_images = torch.stack(preprocessed_images)
                categories = torch.tensor(categories,dtype=torch.long)
                if num_players < self.max_players:
                    pad_count = self.max_players - num_players
                    C,H,W = preprocessed_images.shape[1:]
                    image_paading = torch.zeros(pad_count,C,H,W)
                    preprocessed_images = torch.cat(
                    [preprocessed_images,image_paading],dim=0
                    )   
                    label_padding = torch.full((pad_count,),-1,dtype=torch.long)
                    categories = torch.cat(
                    [categories,label_padding],dim=0
                    )
                elif num_players > self.max_players:
                    preprocessed_images = preprocessed_images[:self.max_players]
                    categories = categories[:self.max_players]

                all_frames_images.append(preprocessed_images)
                all_frames_categories.append(categories)
            all_frames_images = torch.stack(all_frames_images)
            all_frames_categories = torch.stack(all_frames_categories)
            return all_frames_images,all_frames_categories
            

class PlayerGroupDataset(Dataset):
    def __init__(
        self,
        input_root,
        annot_pkl_path,
        player_dict,
        group_dict,
        videos_ids,
        preprocess,
        max_players = 12,
        one_frame = True
    ):
        self.max_players = max_players
        self.player_dict = player_dict
        self.group_dict = group_dict
        self.preprocess = preprocess
        self.one_frame = one_frame
        with open(annot_pkl_path,'rb') as file:
            videos_annot = pickle.load(file)
        videos_ids_set = set(videos_ids) # Set make search O(1)
        self.samples = []

        for video in videos_annot:
            if video not in videos_ids_set:
                continue
            for clip in videos_annot[video]:
                clip_dict = videos_annot[video][clip]
                sorted_frame_ids = sorted(clip_dict['frame_boxes_dct'].keys())
                if one_frame:
                    frame_id = sorted_frame_ids[4]
                    frame_path = os.path.join(input_root,video,clip,f'{frame_id}.jpg')
                    self.samples.append({'image_path':frame_path,
                                        'frame_boxes':clip_dict['frame_boxes_dct'][frame_id],
                                        'group_activity':self.group_dict[clip_dict['category']]
                                        })
                else:
                    frames_path = []
                    frames_boxes = []
                    for frame_id in sorted_frame_ids:
                        frame_path = os.path.join(input_root,video,clip,f'{frame_id}.jpg')
                        frame_boxes = clip_dict['frame_boxes_dct'][frame_id]
                        frames_path.append(frame_path)
                        frames_boxes.append(frame_boxes)
                    self.samples.append({'frames_path':frames_path,
                                        'frames_boxes':frames_boxes,
                                        'group_activity':self.group_dict[clip_dict['category']]
                                        })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        sample = self.samples[index]
        group_activity = sample['group_activity']
        if self.one_frame:
            image = Image.open(sample['image_path']).convert('RGB')
            preprocessed_images = []
            categories = []
            for box_info in sample['frame_boxes']:
                x1, y1, x2, y2 = box_info.box
                cropped_image = image.crop((x1,y1,x2,y2))
                preprocessed_images.append(self.preprocess(cropped_image))
                categories.append(self.player_dict[box_info.category])
            num_players = len(preprocessed_images)
            preprocessed_images = torch.stack(preprocessed_images)
            categories = torch.tensor(categories,dtype=torch.long)
            if num_players < self.max_players:
                pad_count = self.max_players - num_players
                C,H,W = preprocessed_images.shape[1:]
                image_paading = torch.zeros(pad_count,C,H,W)
                preprocessed_images = torch.cat(
                    [preprocessed_images,image_paading],dim=0
                )
                label_padding = torch.full((pad_count,),-1,dtype=torch.long)
                categories = torch.cat(
                    [categories,label_padding],dim=0
                )
            elif num_players > self.max_players:
                preprocessed_images = preprocessed_images[:self.max_players]
                categories = categories[:self.max_players]
            return preprocessed_images , categories , group_activity
        else:
            all_frames_images = []
            all_frames_categories = []
            for frame_path,frame_boxes in zip(sample['frames_path'],sample['frames_boxes']):
                image = Image.open(frame_path).convert('RGB')
                preprocessed_images = []
                categories = []
                for box_info in frame_boxes:
                    x1, y1, x2, y2 = box_info.box
                    cropped_image = image.crop((x1,y1,x2,y2))
                    preprocessed_images.append(self.preprocess(cropped_image))
                    categories.append(self.player_dict[box_info.category])
                num_players = len(preprocessed_images)
                preprocessed_images = torch.stack(preprocessed_images)
                categories = torch.tensor(categories,dtype=torch.long)
                if num_players < self.max_players:
                    pad_count = self.max_players - num_players
                    C,H,W = preprocessed_images.shape[1:]
                    image_paading = torch.zeros(pad_count,C,H,W)
                    preprocessed_images = torch.cat(
                    [preprocessed_images,image_paading],dim=0
                    )   
                    label_padding = torch.full((pad_count,),-1,dtype=torch.long)
                    categories = torch.cat(
                    [categories,label_padding],dim=0
                    )
                elif num_players > self.max_players:
                    preprocessed_images = preprocessed_images[:self.max_players]
                    categories = categories[:self.max_players]

                all_frames_images.append(preprocessed_images)
                all_frames_categories.append(categories)
            all_frames_images = torch.stack(all_frames_images)
            all_frames_categories = torch.stack(all_frames_categories)
            return all_frames_images,all_frames_categories,group_activity







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
    player_labels = {
        'waiting':0, 
        'setting':1, 
        'digging':2, 
        'falling':3, 
        'spiking':4, 
        'blocking':5,
        'jumping':6, 
        'moving':7, 
        'standing':8
        }

    # - Train Videos: 1 3 6 7 10 13 15 16 18 22 23 31 32 36 38 39 40 41 42 48 50 52 53 54
	# - Validation Videos: 0 2 8 12 17 19 24 26 27 28 30 33 46 49 51
	# - Test Videos: 4 5 9 11 14 20 21 25 29 34 35 37 43 44 45 47

    train_ids = ["1", "3", "6", "7", "10", "13", "15", "16", "18", "22", "23", "31",
                 "32", "36", "38", "39", "40", "41", "42", "48", "50", "52", "53", "54"]


    val_ids = ["0", "2", "8", "12", "17", "19", "24", "26", "27", "28", "30", "33", "46", "49", "51"]

    test_ids = ['4','5','9','11','14','20','21','25','29','34','35','37','43','44','45','47']

