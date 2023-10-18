from torchvision.transforms import transforms
import numpy as np
import cv2
import torch
import os


class LabDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir: str, categories: dict, transform=None, training_mode=True):
        self.root_dir = root_dir + '/dataset/'
        self.device = torch.device('cuda')
        self.image_ids = sorted(int(i) for i in os.listdir(self.root_dir + 'masks'))
        self.categories = categories
        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform

        self.training_mode = training_mode

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        # image
        image_id = self.image_ids[idx]
        image = cv2.imread(f'{self.root_dir}images/{image_id}.png').astype(np.uint8)
        image = cv2.resize(image, dsize=(800, 800))
        test_img = image
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transform(image)
        image = image.to(self.device)
        # mask and labels
        masks_path = f'{self.root_dir}masks/{image_id}/'
        labels = os.listdir(masks_path)
        masks = [(cv2.resize(cv2.imread(masks_path + label, 0), (800, 800))) for label in labels]
        masks = np.array([np.where(mask > 0, 1, 0).astype(np.uint8) for mask in masks])
        labels_encoded = [self.categories[label.split('_')[0]] for label in labels]
        # bounding box
        boxes = []
        for mask in masks:
            x, y, w, h = cv2.boundingRect(mask)
            boxes.append([x, y, x + w, y + h])
        # all target
        target = {"labels": torch.tensor(labels_encoded, dtype=torch.int64, device=self.device),
                  "boxes": torch.tensor(boxes, dtype=torch.float32, device=self.device),
                  "masks": torch.tensor(masks, dtype=torch.uint8, device=self.device)}

        if self.training_mode is True:
            return image, target
        else:
            test_target = {"labels": labels,
                           "boxes": boxes,
                           "masks": masks}
            return image, target, test_img, test_target
