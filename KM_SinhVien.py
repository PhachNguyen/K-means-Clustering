import numpy as np
import matplotlib.pyplot as plt

# 1. Hàm tính khoảng cách Euclidean
def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2)) #căn2((Xa - Xb)^2 + (Ya - Yb)^2)

# 2. Thuật toán K-Means 
class KMeans:
    def __init__(self, n_clusters, max_iter=100, tol=1e-4):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol  
        
    def fit(self, X):
        # 3. Khởi tạo tâm cụm ngẫu nhiên
        np.random.seed(0)
        random_indices = np.random.choice(len(X), self.n_clusters, replace=False)
        self.centroids = X[random_indices]

        for i in range(self.max_iter):
            # 4. Gán nhãn cho mỗi điểm (tìm cụm gần nhất)
            self.labels = self._assign_clusters(X)
            
            # 5. Tính tâm cụm mới
            new_centroids = self._update_centroids(X)
            
            # 6. Kiểm tra điều kiện dừng
            if np.all(np.abs(new_centroids - self.centroids) < self.tol):
                break
            
            self.centroids = new_centroids    
        
    def _assign_clusters(self, X):
        labels = []
        for point in X:
            distances = [euclidean_distance(point, centroid) for centroid in self.centroids]
            labels.append(np.argmin(distances))  # Gán vào cụm gần nhất
        return np.array(labels)

    def _update_centroids(self, X):
        centroids = []
        for i in range(self.n_clusters):
            points_in_cluster = X[self.labels == i]
            if len(points_in_cluster) > 0:
                centroids.append(np.mean(points_in_cluster, axis=0))
            else:  # Nếu cụm không có điểm nào, giữ nguyên tâm cụm
                centroids.append(self.centroids[i])
        return np.array(centroids)
    


# 7. Dữ liệu thử nghiệm
from sklearn.datasets import make_blobs

X, _ = make_blobs(n_samples=100, centers=5, cluster_std=1, random_state=9)
X_min = np.min(X[:, 0])
X_max = np.max(X[:, 0])
Y_min = np.min(X[:, 1])
X[:, 0] = 10 * (X[:, 0] - X_min) / (X_max - X_min)
X[:, 1] = X[:, 1] - Y_min

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)  # Biểu đồ bên trái
plt.scatter(X[:, 0], X[:, 1], c='gray', alpha=0.6)
plt.title("Dữ liệu trước khi phân cụm")
plt.xlabel("Điểm trung bình")
plt.ylabel("Hoạt động ngoại khoá")

# 5. Huấn luyện mô hình K-Means
kmeans = KMeans(n_clusters=5)
kmeans.fit(X)

# 6. Trực quan hóa kết quả sau phân cụm**
plt.subplot(1, 2, 2)  # Biểu đồ bên phải
plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels, cmap='viridis', alpha=0.6)
plt.scatter(kmeans.centroids[:, 0], kmeans.centroids[:, 1], s=200, c='red', marker='X')  # Đánh dấu tâm cụm
plt.title("Dữ liệu sau khi phân cụm với K-Means")
plt.xlabel("Điểm trung bình")
plt.ylabel("Hoạt động ngoại khoá")

plt.tight_layout()
plt.show()