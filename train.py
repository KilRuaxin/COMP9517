import os
import sys
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from tqdm import tqdm

from model_Resnet import resnet34
from model_Mobilenetv2 import mobilenet_v2_custom

def get_data_loaders(image_path, batch_size=16):
    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406],
                                                          [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406],
                                                        [0.229, 0.224, 0.225])])
    }

    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),
                                         transform=data_transform["train"])
    val_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                       transform=data_transform["val"])

    class_dict = dict((val, key) for key, val in train_dataset.class_to_idx.items())
    with open(os.path.join(image_path, "class_indices.json"), 'w') as f:
        json.dump(class_dict, f, indent=4)

    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                               shuffle=True, num_workers=nw)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size,
                                             shuffle=False, num_workers=nw)
    return train_loader, val_loader, len(train_dataset), len(val_dataset)

def train_model(model_fn, num_classes, image_path, weight_path=None, save_path="best_model.pth",
                epochs=3, lr=0.0001):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"using {device} device.")

    train_loader, val_loader, train_num, val_num = get_data_loaders(image_path)
    print(f"using {train_num} images for training, {val_num} images for validation.")

    # 创建模型
    net = model_fn(num_classes=num_classes)

    if weight_path:
        print(f"Loading pretrained weights from {weight_path}")
        weights = torch.load(weight_path, map_location="cpu",weights_only=False)
        weights = {k: v for k, v in weights.items() if "fc" not in k and "classifier" not in k and "head" not in k}
        missing_keys, _ = net.load_state_dict(weights, strict=False)
        print("missing keys:", missing_keys)

    if hasattr(net, 'fc'):
        in_features = net.fc.in_features
        net.fc = nn.Linear(in_features, num_classes)
    elif hasattr(net, 'classifier') and isinstance(net.classifier, nn.Sequential):
        in_features = net.classifier[-1].in_features
        net.classifier[-1] = nn.Linear(in_features, num_classes)
    elif hasattr(net, 'head'):  # e.g. ViT
        in_features = net.head.in_features
        net.head = nn.Linear(in_features, num_classes)
    elif hasattr(net, 'base_model') and hasattr(net.base_model, 'classifier'):
        in_features = net.base_model.classifier[-1].in_features
        net.base_model.classifier[-1] = nn.Linear(in_features, num_classes)
    else:
        raise ValueError("Unknown model structure. Cannot replace classifier.")

    net.to(device)

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr)

    best_acc = 0.0
    train_steps = len(train_loader)

    for epoch in range(epochs):
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            logits = net(images.to(device))
            loss = loss_function(logits, labels.to(device))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            train_bar.desc = f"train epoch[{epoch+1}/{epochs}] loss:{loss.item():.3f}"

        net.eval()
        acc = 0.0
        with torch.no_grad():
            val_bar = tqdm(val_loader, file=sys.stdout)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()
                val_bar.desc = f"valid epoch[{epoch+1}/{epochs}]"

        val_acc = acc / val_num
        print(f"[epoch {epoch+1}] train_loss: {running_loss / train_steps:.3f}  val_accuracy: {val_acc:.3f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(net.state_dict(), save_path)

    print("Finished Training")

if __name__ == '__main__':
    image_path = 'D:\CS\WorkPlace\Python\comp9517-project\Aerial_Landscapes_Split'

    # resnet_weights = 'D:\CS\WorkPlace\Python\comp9517-project\\resnet34-pre.pth'
    # train_model(model_fn=resnet34,
    #             num_classes=15,
    #             image_path=image_path,
    #             weight_path=resnet_weights,
    #             save_path='D:\CS\WorkPlace\Python\comp9517-project\\resNet34.pth',
    #             epochs=3)

    mobilenet_weights = None
    train_model(model_fn=lambda num_classes: mobilenet_v2_custom(num_classes=num_classes, pretrained=True),
                num_classes=15,
                image_path=image_path,
                weight_path=None,
                save_path='mobilenetv2.pth',
                epochs=3)
    #