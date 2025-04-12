
import os
import json
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# 支持多个模型导入
from model_Resnet import resnet34
from model_Mobilenetv2 import mobilenet_v2_custom

def load_model(model_name, num_classes, device):
    if model_name == "resnet34":
        model = resnet34(num_classes=num_classes)
        weights_path = r"D:\CS\WorkPlace\Python\comp9517\resnet34.pth"
    elif model_name == "mobilenetv2":
        model = mobilenet_v2_custom(num_classes=num_classes, pretrained=False)
        weights_path = r"D:\CS\WorkPlace\Python\comp9517\mobilenetv2.pth"
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    assert os.path.exists(weights_path), f"file: '{weights_path}' does not exist."
    model.load_state_dict(torch.load(weights_path, map_location=device))
    return model.to(device)


def main(model_name):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"using {device} device.")


    num_classes = 15

    data_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    imgs_root = r"D:\CS\WorkPlace\Python\Project\Aerial_Landscapes"
    assert os.path.exists(imgs_root), f"file: '{imgs_root}' does not exist."

    img_path_list = []
    y_true = []

    class_names = sorted(os.listdir(imgs_root))
    class_to_idx = {cls_name: idx for idx, cls_name in enumerate(class_names)}

    for cls in class_names:
        cls_folder = os.path.join(imgs_root, cls)
        if not os.path.isdir(cls_folder):
            continue
        for file in os.listdir(cls_folder):
            if file.endswith((".jpg", ".png", ".jpeg")):
                img_path_list.append(os.path.join(cls_folder, file))
                y_true.append(class_to_idx[cls])

    print(f"Found {len(img_path_list)} images for batch prediction.")

    json_path = r"D:\CS\WorkPlace\Python\COMP9517\class_indices.json"
    assert os.path.exists(json_path), f"file: '{json_path}' does not exist."
    with open(json_path, "r") as f:
        class_indict = json.load(f)
    idx_to_class = {int(k): v for k, v in class_indict.items()}

    model = load_model(model_name, num_classes, device)
    model.eval()

    y_pred = []
    with torch.no_grad():
        for img_path in tqdm(img_path_list, desc="Batch Predicting"):
            img = Image.open(img_path).convert('RGB')
            img = data_transform(img)
            img = torch.unsqueeze(img, dim=0)

            output = model(img.to(device)).cpu()
            prob = torch.softmax(output, dim=1)
            pred_class = torch.argmax(prob, dim=1).item()
            pred_name = idx_to_class[pred_class]
            pred_prob = prob[0][pred_class].item()
            print(f"image: {os.path.basename(img_path):30}  predicted: {pred_name:15}  prob: {pred_prob:.3f}")
            y_pred.append(pred_class)

    print("\n=== Classification Report ===")
    print(classification_report(y_true, y_pred, target_names=[idx_to_class[i] for i in range(num_classes)], digits=3))

    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[idx_to_class[i] for i in range(num_classes)])
    disp.plot(xticks_rotation=45, cmap="Reds")
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main("mobilenetv2" )
#