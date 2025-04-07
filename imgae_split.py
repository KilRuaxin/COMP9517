import os
import shutil
import random
from tqdm import tqdm

dataset_dir = 'D:\CS\WorkPlace\Python\Project\Aerial_Landscapes'


output_dir = 'D:\CS\WorkPlace\Python\comp9517-project\Aerial_Landscapes_Split'


test_ratio = 0.2



train_dir = os.path.join(output_dir, 'train')
test_dir = os.path.join(output_dir, 'test')
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)


classes = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
print("classes:", classes)


split_records = []


for cls in tqdm(classes, desc="splitting"):
    cls_path = os.path.join(dataset_dir, cls)
    images = os.listdir(cls_path)
    random.shuffle(images)

    num_test = int(len(images) * test_ratio)
    test_images = images[:num_test]
    train_images = images[num_test:]


    os.makedirs(os.path.join(train_dir, cls), exist_ok=True)
    os.makedirs(os.path.join(test_dir, cls), exist_ok=True)


    for img in train_images:
        src = os.path.join(cls_path, img)
        dst = os.path.join(train_dir, cls, img)
        shutil.copy2(src, dst)
        split_records.append({'filename': img, 'class': cls, 'split': 'train'})


    for img in test_images:
        src = os.path.join(cls_path, img)
        dst = os.path.join(test_dir, cls, img)
        shutil.copy2(src, dst)
        split_records.append({'filename': img, 'class': cls, 'split': 'test'})

print("Done")

