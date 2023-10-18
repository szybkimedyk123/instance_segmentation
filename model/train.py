from torch.utils.data import DataLoader, random_split
import torch

import time

from dataset import LabDataset
from maskrcnn import MaskRCNN


def training_loop(model=None, dataset=None, batch_size=3, num_epochs=1000, root_dir="", categories=None):
    model = MaskRCNN(root_dir=root_dir) if model is None else model
    dataset = LabDataset(root_dir, categories) if dataset is None else dataset
    train_dataset, val_dataset = random_split(dataset, [0.8, 0.2])
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
    model.load_checkpoint()

    for epoch in range(num_epochs):
        # training
        train_loss = 0.0
        for i, batch in enumerate(train_loader):
            if i % 10 == 0:
                print('     Training batch: ', i+1)
            try:
                train_loss += model.train_step(batch)
            except:
                print("bad mask")

        # validation
        with torch.no_grad():
            val_loss = 0.0
            for i, batch in enumerate(val_loader):
                if i % 10 == 0:
                    print('     Validation batch: ', i+1)
                try:
                    val_loss += model.val_step(batch)
                except:
                    print("bad mask")

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        with open("console_output.txt", 'a') as console_output:
            console_output.write(str(time.asctime() + '\n'))
            console_output.write(str(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}' + '\n' + '\n'))
        print(time.asctime())
        print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        model.save_checkpoint()


if __name__ == "__main__":
    categories = {"miska2d": 0,
                  "kubek2d": 1,
                  "pudelko2d": 2,
                  "czapka2d": 3,
                  "but2d": 4,
                  "kieliszek2d": 5,
                  "lampka2d": 6}
    root_dir = "/home/viv/AD/studia/sem_6/ZSD/repo/zsd-is/model/mask_v2"
    training_loop(root_dir=root_dir, categories=categories)



