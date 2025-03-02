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
    

# 7. Sinh dữ liệu thử nghiệm
from sklearn.datasets import make_blobs
n_samples = 100
n_features = 3
np.random.seed(0)

purchases = np.random.randint(1, 50, size=n_samples) 
spending = np.random.randint(1, 100, size=n_samples)  
satisfaction = np.random.randint(0, 11, size=n_samples)  

X = np.column_stack([purchases, spending, satisfaction])

# 8. Huấn luyện mô hình
kmeans = KMeans(n_clusters=5)
kmeans.fit(X)

# 9. Trực quan hóa kết quả
fig = plt.figure(figsize=(10, 8))

# Đồ thị cho số lần mua, số tiền, độ hài lòng
ax = fig.add_subplot(projection='3d')
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=kmeans.labels, cmap='viridis')
ax.scatter(kmeans.centroids[:, 0], kmeans.centroids[:, 1], kmeans.centroids[:, 2], s=200, c='red', marker='X')  # Dấu X cho tâm cụm
ax.set_xlabel('Số lần mua')
ax.set_ylabel('Số tiền')
ax.set_zlabel('Độ hài lòng')
ax.set_title("K-Means Clustering")

plt.tight_layout()
plt.show()