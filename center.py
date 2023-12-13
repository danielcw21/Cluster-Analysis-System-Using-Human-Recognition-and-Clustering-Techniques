import cv2
import numpy as np
from sklearn.cluster import KMeans

# 이미지 파일 경로
image_path = 'C:/Users/user/Desktop/img/sh2.jpg'

# 이미지 크기 조정
resize_width = 1280
resize_height = 720

# 객체 탐지를 위한 모델 로드
model_weights = 'C:/Users/user/yolo/yolov3.weights'
model_config = 'C:/Users/user/yolo/yolov3.cfg'
net = cv2.dnn.readNetFromDarknet(model_config, model_weights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# 클래스 이름 로드
classes_file = 'C:/Users/user/yolo/coco.names'
with open(classes_file, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# 이미지 로드 및 크기 조정
image = cv2.imread(image_path)
image = cv2.resize(image, (resize_width, resize_height))

# 객체 탐지 수행
blob = cv2.dnn.blobFromImage(image, 1/255, (416, 416), swapRB=True, crop=False)
net.setInput(blob)
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
outs = net.forward(output_layers)

# 객체 탐지 결과 처리
class_ids = []
confidences = []
boxes = []
for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5 and classes[class_id] == 'person':
            center_x = int(detection[0] * image.shape[1])
            center_y = int(detection[1] * image.shape[0])
            width = int(detection[2] * image.shape[1])
            height = int(detection[3] * image.shape[0])
            x = int(center_x - width / 2)
            y = int(center_y - height / 2)
            class_ids.append(class_id)
            confidences.append(float(confidence))
            boxes.append([x, y, width, height])

# 객체가 없을 경우 처리
if len(boxes) == 0:
    print("No objects detected.")
    exit()

# K-means 군집화
num_clusters = 3  # 군집 수를 3으로 설정
person_centers = np.array([(box[0] + box[2] // 2, box[1] + box[3] // 2) for box in boxes])

# 예외 처리: 군집 수보다 데이터 샘플 수가 적은 경우
if len(person_centers) < num_clusters:
    num_clusters = len(person_centers)

kmeans = KMeans(n_clusters=num_clusters, random_state=42)
labels = kmeans.fit_predict(person_centers)

# 군집 수가 3개보다 작으면 중심점을 기준으로 가장 가까운 군집을 할당
if len(np.unique(labels)) < num_clusters:
    dists = kmeans.transform(person_centers)
    closest_clusters = np.argmin(dists, axis=1)
    labels = closest_clusters

# 군집화 결과를 위한 별도의 도형 그리기
colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0)]  # 군집별로 다른 색상 설정
merged_boxes = []
merged_colors = []
for cluster_id in range(num_clusters):
    cluster_boxes = np.array(boxes)[labels == cluster_id]
    if len(cluster_boxes) > 0:
        x_min = np.min(cluster_boxes[:, 0])
        y_min = np.min(cluster_boxes[:, 1])
        x_max = np.max(cluster_boxes[:, 0] + cluster_boxes[:, 2])
        y_max = np.max(cluster_boxes[:, 1] + cluster_boxes[:, 3])
        merged_boxes.append([x_min, y_min, x_max - x_min, y_max - y_min])
        merged_colors.append(colors[cluster_id])

# 군집화된 좌표 표시
for i, box in enumerate(merged_boxes):
    x, y, width, height = box
    color = merged_colors[i]
    cv2.rectangle(image, (x, y), (x + width, y + height), color, 2)

    # 군집의 중심 좌표 표시
    center_x = x + width // 2
    center_y = y + height // 2
    cv2.circle(image, (center_x, center_y), 5, color, -1)

# 군집화 결과 이미지 표시
cv2.namedWindow("Clustered Image", cv2.WINDOW_NORMAL)
cv2.imshow("Clustered Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()