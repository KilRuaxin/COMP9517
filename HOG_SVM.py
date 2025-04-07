import os
import cv2
import numpy as np
from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
from tqdm import tqdm

# 图像路径
data_root = r"D:\CS\WorkPlace\Python\Project\Aerial_Landscapes"

# HOG 参数
hog_params = {
    'orientations': 9,
    'pixels_per_cell': (8, 8),
    'cells_per_block': (2, 2),
    'block_norm': 'L2-Hys'
}

X, y = [], []
label_map = {}
label_counter = 0

print("Extracting HOG features...")
for cls_name in sorted(os.listdir(data_root)):
    cls_path = os.path.join(data_root, cls_name)
    if not os.path.isdir(cls_path): continue
    if cls_name not in label_map:
        label_map[cls_name] = label_counter
        label_counter += 1
    for file in tqdm(os.listdir(cls_path), desc=cls_name):
        if file.lower().endswith((".jpg", ".png", ".jpeg")):
            img_path = os.path.join(cls_path, file)
            img = cv2.imread(img_path)
            img = cv2.resize(img, (128, 128))  # HOG 特征不需要太大分辨率
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            feat = hog(gray, **hog_params)
            X.append(feat)
            y.append(label_map[cls_name])

X = np.array(X)
y = np.array(y)

# 标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 训练测试划分
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# SVM 训练
print("Training SVM...")
svm = SVC(kernel="rbf", C=10, gamma="scale", probability=True)
svm.fit(X_train, y_train)

# 保存
joblib.dump(svm, "svm_hog_model.pkl")
joblib.dump(scaler, "svm_hog_scaler.pkl")
joblib.dump(label_map, "svm_hog_labels.pkl")

# 测试
y_pred = svm.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=[k for k, v in sorted(label_map.items(), key=lambda x: x[1])]))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
