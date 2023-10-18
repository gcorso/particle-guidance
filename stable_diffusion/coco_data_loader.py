import torch
import torch.utils
import pandas as pd
import os
import open_clip
from PIL import Image
from torchvision import transforms as pth_transforms

class text_image_pair(torch.utils.data.Dataset):
    def __init__(self, dir_path, csv_path, prompt=None, group=False, only_text=False):
        """

        Args:
            dir_path: the path to the stored images
            file_path:
        """
        self.dir_path = dir_path
        self.use_prompt = prompt is not None
        self.group = group
        self.only_text = only_text

        if prompt is None:
            df = pd.read_csv(csv_path)
            self.text_description = df['caption']
        else:
            self.text_description = prompt
        _, _, self.preprocess = open_clip.create_model_and_transforms('ViT-g-14', pretrained='laion2b_s12b_b42k')

        self.dino_transform  = pth_transforms.Compose([
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
        #_, self.preprocess2 = clip.load("ViT-L/14", device='cuda')  # RN50x64
        # tokenizer = open_clip.get_tokenizer('ViT-g-14')

    def __len__(self):
        # count the number of files in the directoryf
        #return len([name for name in os.listdir(self.dir_path) if os.path.isfile(os.path.join(self.dir_path, name))])
        return len(self.text_description)

    def __getitem__(self, idx):

        if self.group:
            img_dir = os.path.join(self.dir_path, f'{idx}')

            if self.use_prompt:
                text = self.text_description
            else:
                text = self.text_description[idx]
            if self.only_text:
                return text

            image_list = []
            dino_image_list = []

            for img in os.listdir(img_dir):
                if not img.endswith(".png"):
                    continue
                img_path = os.path.join(img_dir, img)
                raw_image = Image.open(img_path)
                dino_image = self.dino_transform(raw_image.convert("RGB"))

                image = self.preprocess(raw_image).squeeze().float()
                image_list.append(image)
                dino_image_list.append(dino_image)

            return torch.stack(image_list), text, torch.stack(dino_image_list)
        else:
            if self.use_prompt:
                text = self.text_description
            else:
                text = self.text_description[idx]
            if self.only_text:
                return text

            
            img_path = os.path.join(self.dir_path, f'{idx}.png')
            raw_image = Image.open(img_path)
            dino_image = self.dino_transform(raw_image.convert("RGB"))

            image = self.preprocess(raw_image).squeeze().float()
            # image2 = self.preprocess2(raw_image).squeeze().float()

            return image, text, dino_image