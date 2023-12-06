# EmoSet: A Large-scale Visual Emotion Dataset with Rich Attributes (ICCV2023)

[Jingyuan Yang](https://jingyuanyy.github.io/), Qirui Huang, Tingting Ding, [Dani Lischinski](https://www.cs.huji.ac.il/~danix/), [Daniel Cohen-Or](https://danielcohenor.com/), and [Hui Huang*](https://vcc.tech/~huihuang)

We propose a large-scale visual emotion dataset with rich attributes, named EmoSet. With 3.3 million images in total (EmoSet-3.3M), 118,102 of these images are carefully labeled with machines and human annotators (EmoSet-118K). EmoSet is labeled with 8 emotion categories (amusement, anger, awe, contentment, disgust, excitement, fear, and sadness) in Mikels' emotion model and 6 proposed emotion attributes (brightness, colorfulness, scene type, object class, facial expression, and human action). We believe EmoSet will bring some key insights and encourage further research in visual emotion analysis and understanding.

### [Project Page](https://vcc.tech/EmoSet) | [EmoSet-118K](https://www.dropbox.com/scl/fi/myue506itjfc06m7svdw6/EmoSet-118K.zip?dl=0&rlkey=7f3oyjkr6zyndf0gau7t140rv) | [EmoSet-3.3M](https://www.dropbox.com/scl/fo/rfg7vk33lwt46clxd23wh/h?rlkey=cpdvinfufzv9pv8q9tuezmx2u&dl=0) | [Paper](https://openaccess.thecvf.com/content/ICCV2023/html/Yang_EmoSet_A_Large-scale_Visual_Emotion_Dataset_with_Rich_Attributes_ICCV_2023_paper.html) | [Supp](https://vcc-szu-cdn.s3.ap-southeast-1.amazonaws.com/EmoSet-supp.pdf)
![Teaser image](./image/teaser3-01.jpg)

## File Structure

The EmoSet dataset utilizes a specific directory and file structure under the `data_root` directory. Below is the layout:

```
data_root
│
├── info.json
├── train.json
├── val.json
├── test.json
│
├── annotation
│   ├── amusement
│   │   └── .json files
│   ├── anger
│   │   └── .json files
│   ├── ...
│   └── sadness
│       └── .json files
│
└── image
    ├── amusement
    │   └── .jpg files
    ├── anger
    │   └── .jpg files
    ├── ...
    └── sadness
        └── .jpg files

```

### Description of the Files and Folders

- `info.json`: Contains the label-to-index (`label2idx`) and index-to-label (`idx2label`) mappings for each attribute and the emotion classes. For example, the `label2idx` of emotion is { "amusement": 0,  "awe": 1,  "contentment": 2,  "excitement": 3,  "anger": 4,  "disgust": 5,  "fear": 6,  "sadness": 7}.
- `train.json`, `val.json`, `test.json`: These JSON files contain the respective data for the training, validation, and test sets.
- `annotation`: A directory that houses subdirectories for each of the eight emotion labels. Each subdirectory contains JSON files with annotation information for the corresponding images.
- `image`: A directory containing subdirectories for each of the eight emotion labels. Each subdirectory houses image files (.jpg format) corresponding to that emotion.

The path to a specific image or annotation is constructed by appending the respective path elements. For instance, if we have a `train.json` file structured like this:

```
[
    [
        "amusement",
        "amusement_12865",
        "image/amusement/amusement_12865.jpg",
        "annotation/amusement/amusement_12865.json"
    ],
    ...
]

```

The third and fourth elements in the list specify the relative paths to the image and annotation files, respectively. The path is relative to `data_root`, therefore the full path to the image would be `data_root/image/amusement/amusement_12865.jpg` and to the annotation would be `data_root/annotation/amusement/amusement_12865.json`.

## PyTorch Dataset Class

The EmoSet Dataset is a class in PyTorch's `Dataset` module. It is designed for tasks related to emotions and each attributes, and specifically handles loading, transforming, and serving the data.

### Usage

To use this dataset, specify the following parameters when creating an instance:

1. `data_root`: A string representing the path to the root directory of the dataset.
2. `num_emotion_classes`: Should be either 8 or 2. This corresponds to the number of emotion classes the dataset should consider. If it's 8, the dataset uses the detailed emotion classes as they are. If it's 2, it simplifies the emotions into two classes: "positive" and "negative".
3. `phase`: Either 'train', 'val', or 'test'. This corresponds to the phase of the experiment. The dataset will load the corresponding data (train, validation, or test) based on this parameter.

Example:

```python
data_root = 'dataroot'
num_emotion_classes = 8
phase = 'train'

dataset = EmoSet(
    data_root=data_root,
    num_emotion_classes=num_emotion_classes,
    phase=phase,
)

```

This dataset can be used with a `DataLoader` for batch processing:

```python
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

for i, data in enumerate(dataloader):
    pass
    # print(data['image'])
    # print(data['emotion_label_idx'])
    # print(data['scene_label_idx'])
    # print(data['facial_expression_label_idx'])
    # print(data['human_action_label_idx'])
    # print(data['brightness_label_idx'])
    # print(data['colorfulness_label_idx'])
    # print(data['object_label_idx'])
    # break

```

### Dataset Structure

The dataset assumes the following structure in the `data_root` directory:

- Each phase ('train', 'val', 'test') has a corresponding `.json` file with the same name.
- The `info.json` file contains the mapping of labels to indices for each attribute.
- Images and their corresponding annotation files (in `.json` format) are located in the `image` and `annotation` directories under `data_root`, respectively.

### Data Transformations

For each phase, the dataset applies a series of transformations to the images:

- For 'train' phase: Random resize crop to 224, random horizontal flip, conversion to tensor, and normalization.
- For 'val' and 'test' phases: Resize to 224, center crop to 224, conversion to tensor, and normalization.

### Data Output

When you iterate over the dataset, it yields a dictionary with the following keys:

- 'image_id': The id of the image.
- 'image': The transformed image.
- 'emotion_label_idx': The index of the emotion label.
- For each attribute in `ATTRIBUTES_MULTI_CLASS` and `ATTRIBUTES_MULTI_LABEL` (which include 'scene', 'facial_expression', 'human_action', 'brightness', 'colorfulness', 'object'), there is an additional key in the form of 'attribute_label_idx', which contains the index of the attribute label.

The output can then be used in your model for training, validation, or testing.

### Different Attribute Types

The EmoSet dataset class handles various attributes for each image, which can be classified into two types: multi-class and multi-label attributes. Multi-class attributes involve categorizing an image into one of several possible classes, while multi-label attributes allow for simultaneous classification into multiple classes.

Defined in `ATTRIBUTES_MULTI_CLASS`, the multi-class attributes comprise 'scene', 'facial_expression', 'human_action', 'brightness', and 'colorfulness'. In contrast, the multi-label attribute 'object' is defined in `ATTRIBUTES_MULTI_LABEL`.

Each attribute has a specific number of classes, as defined in `NUM_CLASSES`. For instance, 'brightness' and 'colorfulness' have 11 classes each, 'scene' has 254, 'object' has 409, 'facial_expression' has 6, and 'human_action' has 264 classes.

### Handling Absent Attributes

Each attribute in `ATTRIBUTES_MULTI_CLASS` and `ATTRIBUTES_MULTI_LABEL` generates an additional key in the output dictionary labeled as 'attribute_label_idx'. The management of an absent attribute is specific to its type and relates to the loss function used for the associated classification task:

- For multi-class attributes: If an attribute is missing, suggesting a lack of ground truth, it's set to `-1`. This aligns with the `nn.CrossEntropyLoss(ignore_index=-1)` loss function used for multi-class classification tasks. This function effectively skips the calculation of loss for these missing labels.
- For the multi-label attribute (i.e., 'object'): If an object is absent in the ground truth, it's labeled as `0`; if it's present, it's labeled as `1`. This convention is associated with the `nn.BCELoss()` loss function used for multi-label classification tasks. In this context, each class is treated as a separate binary classification, so a class's absence in the label is represented as `0`.

### Required Python Packages

Ensure your Python environment includes the necessary packages: torch, torchvision, os, json, PIL.

### Citation

If you think this project is helpful, please feel free to leave a star or cite our paper:
```
@ inproceedings {EmoSet,
  title={EmoSet: A Large-scale Visual Emotion Dataset with Rich Attributes},
  author={Yang, Jingyuan and Huang, Qirui and Ding, Tingting and Lischinski, Dani and Cohen-Or, Daniel and Huang, Hui},
  booktitle = {ICCV},
  year={2023}
}
```
