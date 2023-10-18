from torchmetrics.classification import BinaryJaccardIndex
import torchvision.ops as ops
import torch

import matplotlib.pyplot as plt
import numpy as np
import random
import cv2

from dataset import LabDataset
from maskrcnn import MaskRCNN


def imshow(img):
    cv2.imshow('name', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def testing_loop(root_dir, categories, checkpoint_path=None, own_model=None, own_dataset=None, testing_batch=150):
    if own_model is None:
        model = MaskRCNN(root_dir=root_dir)
    else:
        model = own_model

    if checkpoint_path is None:
        checkpoint_path = root_dir + '/model/mask_v2/dataset/checkpoint_2023-04-20_16-36-46.pth'
    else:
        checkpoint_path = checkpoint_path

    model.load_checkpoint(checkpoint_path=checkpoint_path)
    model.eval()
    dataset = LabDataset(root_dir, categories, training_mode=False) if own_dataset is None else own_dataset
    random_images = list(np.random.randint(len(dataset), size=testing_batch))
    random.shuffle(random_images)
    mask_iou_metric = BinaryJaccardIndex(treshold=0.5)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize variables for IOU and accuracy
    box_iou_list = []
    accuracy_list = []
    mask_iou_list = []
    num_of_objects = 0
    # Iterate through each image in the testing dataset
    ids = []
    for i in random_images:
        print(f'Image nr: {i}')
        image, target, test_image, test_target = dataset[i]

        # print(test_target['labels'])
        # print(target['labels'])
        # for bbox in test_target['boxes']:
        #     cv2.rectangle(test_image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 1)
        # imshow(test_image)

        # Run model to generate predictions
        with torch.no_grad():
            output = model(image.unsqueeze(0))

        # Load image and ground truth annotations
        gt_labels = target["labels"].detach().cpu()
        gt_boxes = target["boxes"].detach().cpu()
        gt_masks = target["masks"].detach().cpu()
        num_of_objects += gt_labels.shape[0]

        scores = output[0]["scores"].detach().cpu()
        all_labels = output[0]["labels"].detach().cpu()
        all_boxes = output[0]["boxes"].detach().cpu()
        all_masks = output[0]["masks"].detach().cpu()

        # Filter out low confidence detections
        keep_indices = (scores > 0.5).nonzero().squeeze()
        if keep_indices.numel() == 0:  # nothing detected
            continue
        # print('     ', keep_indices)
        # print('     ', keep_indices.numel())

        labels = []
        boxes = []
        masks = []
        if keep_indices.numel() == 1:
            labels.append(all_labels[0])
            boxes.append(all_boxes[0])
            masks.append(all_masks[0])
        else:
            for index in np.asarray(keep_indices):
                labels.append(all_labels[index])
                boxes.append(all_boxes[index])
                masks.append(all_masks[index])

        # Calculate IOU and accuracy for each predicted object
        num_correct_labels = 0
        for box, label, mask in zip(boxes, labels, masks):
            # cv2.rectangle(test_image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 0), 1)
            # Calculate IOU for each predicted object
            # print(gt_boxes)
            # print(box.device())
            box_iou = ops.box_iou(gt_boxes, box.unsqueeze(0))
            box_iou_max, max_idx = torch.max(box_iou, dim=0)
            box_iou_list.append(float(box_iou_max))

            mask_iou = mask_iou_metric(mask, gt_masks[max_idx])
            mask_iou_list.append(float(mask_iou))

            if label == gt_labels[max_idx]:
                num_correct_labels += 1
        # print(labels)
        # imshow(test_image)

        # Calculate accuracy for the image
        accuracy = num_correct_labels / len(labels)
        accuracy_list.append(accuracy)

    # Calculate average IOU and accuracy over testing batch
    avg_iou = sum(box_iou_list) / num_of_objects
    avg_accuracy = sum(accuracy_list) / num_of_objects
    avg_mask_iou = sum(mask_iou_list) / num_of_objects

    print('ids: ', ids)
    print(f"Average IOU: {avg_iou:.3f}")
    print(f"Average Accuracy: {avg_accuracy:.3f}")
    print(f"Average Mask IOU: {avg_mask_iou:.3f}")


def print_training_graph():
    file_path = '/home/viv/AD/studia/sem_6/ZSD/repo/zsd-is/model/mask_v2/console_output.txt'
    sum_epoch = 0
    train_loss = []
    val_loss = []
    epoch_list = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            split_line = line.split(' ')
            if split_line[0] == 'Epoch':
                sum_epoch += 1
                epoch_list.append(sum_epoch)

                train_letters = list(split_line[4])
                train_letters.pop()
                train_numbers = ''.join(train_letters)
                train_loss.append(float(train_numbers))

                val_letters = list(split_line[7])
                val_letters.pop()
                val_letters.pop()
                val_numbers = ''.join(val_letters)
                val_loss.append(float(val_numbers))

    plt.plot(epoch_list, train_loss, 'ro', label='Training loss')
    plt.plot(epoch_list, val_loss, 'gx', label='Validation loss')
    plt.title('Training and validation loss')
    plt.grid()
    plt.legend()
    plt.show()


if __name__ == "__main__":
    categories = {"miska2d": 0,
                  "kubek2d": 1,
                  "pudelko2d": 2,
                  "czapka2d": 3,
                  "but2d": 4,
                  "kieliszek2d": 5,
                  "lampka2d": 6}
    root_dir = '/home/lab/ZSD-IS'
    checkpoint_path = '/home/lab/ZSD-IS/repo/zsd-is/dataset/checkpoint_2023-05-31_07-38-56.pth'
    testing_loop(root_dir, categories, checkpoint_path=checkpoint_path)
    # print_training_graph()

