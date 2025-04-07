import os
import shutil
import random
from tqdm import tqdm
import pandas as pd

dataset_dir = 'D:\CS\WorkPlace\Python\Project\Aerial_Landscapes'

# 输出路径
output_dir = 'D:\CS\WorkPlace\Python\comp9517-project\Aerial_Landscapes_Split'

# 测试集比例
test_ratio = 0.2

# 是否生成 CSV
save_csv = True
csv_path = 'D:\CS\WorkPlace\Python\comp9517-project\split_record.csv'
# ---------------------------

# 创建输出目录
train_dir = os.path.join(output_dir, 'train')
test_dir = os.path.join(output_dir, 'test')
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# 获取类别
classes = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
print("检测到的类别:", classes)

# 划分记录
split_records = []

# 开始划分
for cls in tqdm(classes, desc="划分中"):
    cls_path = os.path.join(dataset_dir, cls)
    images = os.listdir(cls_path)
    random.shuffle(images)

    num_test = int(len(images) * test_ratio)
    test_images = images[:num_test]
    train_images = images[num_test:]

    # 创建子文件夹
    os.makedirs(os.path.join(train_dir, cls), exist_ok=True)
    os.makedirs(os.path.join(test_dir, cls), exist_ok=True)

    # 复制 train
    for img in train_images:
        src = os.path.join(cls_path, img)
        dst = os.path.join(train_dir, cls, img)
        shutil.copy2(src, dst)
        split_records.append({'filename': img, 'class': cls, 'split': 'train'})

    # 复制 test
    for img in test_images:
        src = os.path.join(cls_path, img)
        dst = os.path.join(test_dir, cls, img)
        shutil.copy2(src, dst)
        split_records.append({'filename': img, 'class': cls, 'split': 'test'})

print("✅ 数据集划分完成！")

# 保存 CSV
if save_csv:
    df = pd.DataFrame(split_records)
    df.to_csv(csv_path, index=False)
    print(f"✅ 划分记录已保存到: {csv_path}")