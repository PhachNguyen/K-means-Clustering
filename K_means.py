# import numpy as np
# import matplotlib.pyplot as plt

# # Dữ liệu mẫu (4 điểm)
# data = np.array([[1, 1], [1.5, 2], [5, 5], [6, 6]])

# # Số cụm
# k = 2  

# # Khởi tạo tâm cụm ngẫu nhiên
# centroids = data[np.random.choice(data.shape[0], k, replace=False)]

# # Hàm tính khoảng cách Euclidean
# def euclidean_distance(a, b):
#     return np.sqrt(np.sum((a - b) ** 2))

# # Thuật toán K-Means
# def k_means(data, centroids, max_iter=100):
#     for _ in range(max_iter):
#         # Gán cụm cho từng điểm
#         clusters = {}
#         for i in range(k):
#             clusters[i] = []
        
#         for point in data:
#             distances = [euclidean_distance(point, centroid) for centroid in centroids]
#             cluster = np.argmin(distances)
#             clusters[cluster].append(point)
        
#         # Lưu lại tâm cụm cũ
#         old_centroids = centroids.copy()
        
#         # Cập nhật lại tâm cụm
#         for cluster in clusters:
#             if clusters[cluster]:  # Tránh lỗi chia cho 0
#                 centroids[cluster] = np.mean(clusters[cluster], axis=0)
        
#         # Kiểm tra hội tụ (nếu tâm cụm không thay đổi)
#         if np.allclose(centroids, old_centroids):
#             break
    
#     return clusters, centroids

# # Thực hiện thuật toán K-Means
# clusters, final_centroids = k_means(data, centroids)

# # Trực quan hóa kết quả
# colors = ['red', 'blue']
# for cluster, points in clusters.items():
#     points = np.array(points)
#     plt.scatter(points[:, 0], points[:, 1], color=colors[cluster], label=f'Cluster {cluster + 1}')

# # Hiển thị tâm cụm
# plt.scatter(final_centroids[:, 0], final_centroids[:, 1], color='black', marker='x', s=100, label='Centroids')
# plt.title('K-Means Clustering Visualization')
# plt.xlabel('X-axis')
# plt.ylabel('Y-axis')
# plt.legend()
# plt.grid(True)
# plt.show()

# Bài 2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# 1. Sinh một tập dữ liệu gồm 100 điểm ngẫu nhiên
np.random.seed(42)  # Đảm bảo kết quả ngẫu nhiên nhất quán
data = np.random.rand(100, 2) * 10  # Tạo 100 điểm ngẫu nhiên trong không gian 2D (giới hạn [0, 10])

# 2. Trực quan hóa tập dữ liệu đã sinh
plt.figure(figsize=(8, 6))
plt.scatter(data[:, 0], data[:, 1], color='gray', s=50)
plt.title("Tập dữ liệu ban đầu (100 điểm ngẫu nhiên)")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.grid(True)
plt.show()

# 3. Áp dụng thuật toán K-Means để phân cụm 100 điểm thành 5 cụm
k = 5  # Số cụm
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(data)
labels = kmeans.labels_  # Nhãn cụm của từng điểm
centroids = kmeans.cluster_centers_  # Tâm cụm

# 4. Trực quan hóa kết quả phân cụm
colors = ['red', 'blue', 'green', 'purple', 'orange']
plt.figure(figsize=(8, 6))

# Vẽ các điểm dữ liệu với màu sắc theo cụm
for i in range(k):
    cluster_points = data[labels == i]  # Lấy các điểm thuộc cụm i
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], color=colors[i], s=50, label=f'Cluster {i + 1}')

# Vẽ tâm cụm
plt.scatter(centroids[:, 0], centroids[:, 1], color='black', marker='x', s=100, label='Centroids')

# Thêm tiêu đề và nhãn
plt.title("Kết quả phân cụm (K-Means)")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.legend()
plt.grid(True)
plt.show()