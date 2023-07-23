import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
import json

from PIL import Image


class EmoSet(Dataset):
    ATTRIBUTES_MULTI_CLASS = [
        'scene', 'facial_expression', 'human_action', 'brightness', 'colorfulness',
    ]
    ATTRIBUTES_MULTI_LABEL = [
        'object'
    ]
    NUM_CLASSES = {
        'brightness': 11,
        'colorfulness': 11,
        'scene': 254,
        'object': 409,
        'facial_expression': 6,
        'human_action': 264,
    }

    def __init__(self,
                 data_root,
                 num_emotion_classes,
                 phase,
                 ):
        assert num_emotion_classes in (8, 2)
        assert phase in ('train', 'val', 'test')
        self.transforms_dict = self.get_data_transforms()

        self.info = self.get_info(data_root, num_emotion_classes)

        if phase == 'train':
            self.transform = self.transforms_dict['train']
        elif phase == 'val':
            self.transform = self.transforms_dict['val']
        elif phase == 'test':
            self.transform = self.transforms_dict['test']
        else:
            raise NotImplementedError

        data_store = json.load(open(os.path.join(data_root, f'{phase}.json')))
        self.data_store = [
            [
                self.info['emotion']['label2idx'][item[0]],
                item[1],
                os.path.join(data_root, item[2]),
                os.path.join(data_root, item[3])
            ]
            for item in data_store
        ]

    @classmethod
    def get_data_transforms(cls):
        transforms_dict = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'test': transforms.Compose([
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }
        return transforms_dict

    def get_info(self, data_root, num_emotion_classes):
        assert num_emotion_classes in (8, 2)
        info = json.load(open(os.path.join(data_root, 'info.json')))
        if num_emotion_classes == 8:
            pass
        elif num_emotion_classes == 2:
            emotion_info = {
                'label2idx': {
                    'amusement': 0,
                    'awe': 0,
                    'contentment': 0,
                    'excitement': 0,
                    'anger': 1,
                    'disgust': 1,
                    'fear': 1,
                    'sadness': 1,
                },
                'idx2label': {
                    '0': 'positive',
                    '1': 'negative',
                }
            }
            info['emotion'] = emotion_info
        else:
            raise NotImplementedError

        return info

    def load_image_by_path(self, path):
        image = Image.open(path).convert('RGB')
        image = self.transform(image)
        return image

    def load_annotation_by_path(self, path):
        json_data = json.load(open(path))
        return json_data

    def __getitem__(self, item):
        emotion_label_idx, image_id, image_path, annotation_path = self.data_store[item]
        image = self.load_image_by_path(image_path)
        annotation_data = self.load_annotation_by_path(annotation_path)
        data = {'image_id': image_id, 'image': image, 'emotion_label_idx': emotion_label_idx}

        for attribute in self.ATTRIBUTES_MULTI_CLASS:
            # if empty, set to -1, else set to label index
            attribute_label_idx = -1
            if attribute in annotation_data:
                attribute_label_idx = self.info[attribute]['label2idx'][str(annotation_data[attribute])]
            data.update({f'{attribute}_label_idx': attribute_label_idx})

        for attribute in self.ATTRIBUTES_MULTI_LABEL:
            # if empty, set to 0, else set to 1
            assert attribute == 'object'
            num_classes = self.NUM_CLASSES[attribute]
            attribute_label_idx = torch.zeros(num_classes)
            if attribute in annotation_data:
                for label in annotation_data[attribute]:
                    attribute_label_idx[self.info[attribute]['label2idx'][label]] = 1
            data.update({f'{attribute}_label_idx': attribute_label_idx})

        return data

    def __len__(self):
        return len(self.data_store)


if __name__ == '__main__':
    data_root = r'F:\common_file_system\EmoSet\EmoSet_v5_划分train-test-val'
    num_emotion_classes = 8
    phase = 'train'

    dataset = EmoSet(
        data_root=data_root,
        num_emotion_classes=num_emotion_classes,
        phase=phase,
    )

    # print(dataset.info)
    dataloader = DataLoader(dataset, batch_size = 16, shuffle = True)

    for i, data in enumerate(dataloader):
        pass
        # print(data['emotion_label_idx'])
        # print(data['scene_label_idx'])
        # print(data['facial_expression_label_idx'])
        # print(data['human_action_label_idx'])
        # print(data['brightness_label_idx'])
        # print(data['colorfulness_label_idx'])
        # print(data['object_label_idx'])
        # break


