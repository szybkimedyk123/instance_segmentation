from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
import torch

import datetime
import glob
import os


class MaskRCNN(torch.nn.Module):
    def __init__(self, num_clas=8, root_dir='', training=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = maskrcnn_resnet50_fpn_v2(weights_backbone='DEFAULT')
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=num_clas)
        self.device = torch.device('cuda')
        self.model.to(self.device)
        if training:
            self.model.train()

        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=1e-5)
        self.root_dir = root_dir

    def forward(self, images, targets=None):
        return self.model(images, targets)

    def train_step(self, batch):
        images, annotations = batch
        loss_dict = self.model(images, annotations)
        loss = sum(loss for loss in loss_dict.values())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        loss.item()
        return loss.item()

    def val_step(self, batch):
        images, annotations = batch
        loss_dict = self.model(images, annotations)
        loss = sum(loss for loss in loss_dict.values())
        return loss

    def test_step(self, batch):
        images, annotations = batch
        output_dict = self.model(images, annotations)
        return output_dict

    def save_checkpoint(self, checkpoint_dir=None):
        if checkpoint_dir is None:
            checkpoint_dir = f'{self.root_dir}/dataset/checkpoints'
        os.makedirs(checkpoint_dir, exist_ok=True)
        now = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        checkpoint_file = os.path.join(checkpoint_dir, f'checkpoint_{now}.pth')
        checkpoint = {'model_state_dict': self.state_dict()}
        torch.save(checkpoint, checkpoint_file)

    def load_checkpoint(self, checkpoint_path=None):
        if checkpoint_path is None:
            checkpoint_files = glob.glob(os.path.join(self.root_dir, '/dataset/checkpoint_*.pth'))
            if not checkpoint_files:
                raise ValueError("No checkpoints found in the directory.")
            latest_checkpoint = max(checkpoint_files, key=os.path.getctime)
        else:
            latest_checkpoint = checkpoint_path

        if torch.cuda.is_available():
            checkpoint = torch.load(latest_checkpoint)
        else:
            checkpoint = torch.load(latest_checkpoint, map_location=torch.device(self.device))
        self.load_state_dict(checkpoint["model_state_dict"])
