import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F
from PIL import Image
import os
import numpy as np

class SmokeDataset(Dataset):
    def __init__(self, root, type, transforms=None):
        self.root = os.path.join(root, 'images', type)
        self.label_root = os.path.join(root, 'labels', type)
        self.transforms = transforms
        self.imgs = list(sorted(os.listdir(self.root)))
        self.labels = list(sorted(os.listdir(self.label_root)))

    def __getitem__(self, idx):

        img_name = self.imgs[idx]
        img_path = os.path.join(self.root, img_name)
        img = Image.open(img_path).convert("RGB")

        label_name = img_name.split('.')[0] + '.txt'
        label_path = os.path.join(self.label_root, label_name)

        if not os.path.exists(label_path):
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            area = torch.zeros(0, dtype=torch.float32)
            labels = torch.ones((len(boxes),), dtype=torch.int64) 
            image_id = torch.tensor([idx])
            iscrowd = torch.zeros((len(boxes),), dtype=torch.int32)

            target = {}
            target["boxes"] = boxes
            target["labels"] = labels
            target["image_id"] = image_id
            target["area"] = area
            target["iscrowd"] = iscrowd

            if self.transforms:
                img, target = self.transforms(img, target)

            return img, target


        img = Image.open(img_path).convert("RGB")
        
        boxes = []
        with open(label_path) as f:
            for line in f:
                parts = line.strip().split()
                print(parts)
                _,x_center, y_center, width, height = map(float, parts)
                x_min = x_center - width / 2
                y_min = y_center - height / 2
                x_max = x_center + width / 2
                y_max = y_center + height / 2
                boxes.append([x_min, y_min, x_max, y_max])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        if boxes.shape[0] == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            area = torch.zeros(0, dtype=torch.float32)
        else:
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        labels = torch.ones((len(boxes),), dtype=torch.int64)
        image_id = torch.tensor([idx]) 
        iscrowd = torch.zeros((len(boxes),), dtype=torch.int32)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)

class ToTensor(object):
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target


def get_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes) 

    return model

dataset_train = SmokeDataset(root='DS-71c1fd51-sam-synthetic',type="train", transforms=ToTensor())
dataset_val = SmokeDataset(root='DS-71c1fd51-sam-synthetic', type="val", transforms=ToTensor())

data_loader_train = DataLoader(dataset_train, batch_size=2, shuffle=True, num_workers=4, collate_fn=lambda x: tuple(zip(*x)))
data_loader_val = DataLoader(dataset_val, batch_size=1, shuffle=False, num_workers=4, collate_fn=lambda x: tuple(zip(*x)))


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

num_classes = 2 
model = get_model(num_classes)
model.to(device)

params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)


num_epochs = 3

for epoch in range(num_epochs):
    model.train()
    i = 0
    for images, targets in data_loader_train:
        if i == 0:
            print(targets)
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())
        
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if i % 10 == 0:
            print(f"Iteration #{i} loss: {losses.item()}")
        i += 1

    lr_scheduler.step()

    print(f"Epoch #{epoch} completed")



import matplotlib.pyplot as plt

def show_prediction(model, device, dataset, idx):
    img, _ = dataset[idx]
    model.eval()
    with torch.no_grad():
        prediction = model([img.to(device)])
    
    print(prediction)

    img = img.mul(255).permute(1, 2, 0).byte().numpy()
    plt.figure(figsize=(10,10))
    plt.imshow(img)
    
    for element in range(len(prediction[0]['boxes'])):
        boxes = prediction[0]['boxes'][element].cpu().numpy()
        score = np.round(prediction[0]['scores'][element].cpu().numpy(), decimals=4)
        
        if score > 0.5:
            plt.gca().add_patch(plt.Rectangle((boxes[0], boxes[1]), boxes[2] - boxes[0], boxes[3] - boxes[1], linewidth=1, edgecolor='r', facecolor='none'))
            plt.text(boxes[0], boxes[1], score, bbox=dict(facecolor='white', alpha=0.5))
    
    plt.axis('off')
    plt.show()

img_idx = 5

show_prediction(model, device, dataset_val, img_idx)