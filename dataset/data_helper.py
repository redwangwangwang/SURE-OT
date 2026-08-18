
import os
import json
import re
import numpy as np
from PIL import Image
import torch.utils.data as data
from transformers import BertTokenizer, AutoImageProcessor
import torch
from torchvision import transforms
class FieldParser:
    def __init__(
            self,
            args
    ):
        super().__init__()
        self.args = args
        self.dataset = args.dataset
        self.vit_feature_extractor = AutoImageProcessor.from_pretrained(args.vision_model)

    def _parse_image(self, img):
        pixel_values = self.vit_feature_extractor(img, return_tensors="pt").pixel_values
        return pixel_values[0]

    def clean_report(self, report):
        # clean Iu-xray reports
        if self.dataset == "iu_xray":
            report_cleaner = lambda t: t.replace('..', '.').replace('..', '.').replace('..', '.').replace('1. ', '') \
            .replace('. 2. ', '. ').replace('. 3. ', '. ').replace('. 4. ', '. ').replace('. 5. ', '. ') \
            .replace(' 2. ', '. ').replace(' 3. ', '. ').replace(' 4. ', '. ').replace(' 5. ', '. ') \
            .strip().lower().split('. ')
            sent_cleaner = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '').
                                            replace('\\', '').replace("'", '').strip().lower())
            tokens = [sent_cleaner(sent) for sent in report_cleaner(report) if sent_cleaner(sent) != []]
            report = ' . '.join(tokens) + ' .'
        # clean MIMIC-CXR reports
        else:
            report_cleaner = lambda t: t.replace('\n', ' ').replace('__', '_').replace('__', '_').replace('__', '_') \
                .replace('__', '_').replace('__', '_').replace('__', '_').replace('__', '_').replace('  ', ' ') \
                .replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ') \
                .replace('..', '.').replace('..', '.').replace('..', '.').replace('..', '.').replace('..', '.') \
                .replace('..', '.').replace('..', '.').replace('..', '.').replace('1. ', '').replace('. 2. ', '. ') \
                .replace('. 3. ', '. ').replace('. 4. ', '. ').replace('. 5. ', '. ').replace(' 2. ', '. ') \
                .replace(' 3. ', '. ').replace(' 4. ', '. ').replace(' 5. ', '. ').replace(':', ' :') \
                .strip().lower().split('. ')
            sent_cleaner = lambda t: re.sub('[.,?;*!%^&_+()\[\]{}]', '', t.replace('"', '').replace('/', '')
                                .replace('\\', '').replace("'", '').strip().lower())
            tokens = [sent_cleaner(sent) for sent in report_cleaner(report) if sent_cleaner(sent) != []]
            report = ' . '.join(tokens) + ' .' 
        # report = ' '.join(report.split()[:self.args.max_txt_len])
        return report

    def parse(self, features):
        to_return = {'id': features['id']}
        report = features.get("report", "")
        report = self.clean_report(report)
        to_return['input_text'] = report

        context_report = features.get("context_report", "")
        context_report = self.clean_report(context_report)
        to_return['context_input_text'] = context_report

        with Image.open(os.path.join(self.args.base_dir, features['image_path'][0])) as pil:
            array = np.array(pil, dtype=np.uint8)
            if array.shape[-1] != 3 or len(array.shape) != 3:
                array = np.array(pil.convert("RGB"), dtype=np.uint8)
            image = self._parse_image(array)
        to_return["image"] = image

        with Image.open(os.path.join(self.args.base_dir, features['context_image'][0])) as context_pil:
            context_array = np.array(context_pil, dtype=np.uint8)
            if context_array.shape[-1] != 3 or len(context_array.shape) != 3:
                context_array = np.array(context_pil.convert("RGB"), dtype=np.uint8)
            context_image = self._parse_image(context_array)
        to_return["context_image"] = context_image

        cls_labels = features.get("labels", "")
        cls_labels = torch.from_numpy(np.array(cls_labels)).long()
        to_return["current_labels"] = cls_labels

        return to_return

    def transform_with_parse(self, inputs):
        return self.parse(inputs)

class ParseDataset(data.Dataset):
    def __init__(self, args, split='train'):
        self.args = args
        self.meta = json.load(open(os.path.join(args.annotation,"Lmimic-contextmrg" + split + ".json"), 'r'))
        self.meta = {int(key): value for key, value in self.meta.items()}

        self.parser = FieldParser(args)

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, index):
        features = self.meta[index]
        parsed_data = self.parser.transform_with_parse(features)
        if parsed_data is None:
            return self.__getitem__((index + 1) % len(self.meta))
        return parsed_data

def create_datasets(args):
    train_dataset = ParseDataset(args, 'train')
    dev_dataset = ParseDataset(args, 'val')
    test_dataset = ParseDataset(args, 'test')

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Dev dataset size: {len(dev_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    return train_dataset, dev_dataset, test_dataset



