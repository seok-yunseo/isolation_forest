# 튜토리얼 코드
from sklearn.ensemble import IsolationForest
from sklearn.datasets import make_blobs
from numpy import quantile, where, random
import matplotlib.pyplot as plt

# 랜덤 데이터 생성
random.seed(3)
x, _ = make_blobs(
    n_samples=200,
    centers=1,
    cluster_std=0.3,
    center_box=(20, 5)
)

# 데이터 확인
plt.scatter(x[:,0], x[:,1])
plt.title("Original Dataset")
plt.show()


# -------------------------
# 방법 1 : contamination 사용
# -------------------------

iforest = IsolationForest(
    n_estimators=100,
    contamination=0.02
)

pred = iforest.fit_predict(x)

anom_index = where(pred == -1)
values = x[anom_index]

plt.scatter(x[:,0], x[:,1])
plt.scatter(values[:,0], values[:,1], color='red')
plt.title("Anomaly Detection (Method 1)")
plt.show()


# -------------------------
# 방법 2 : score 기반
# -------------------------

iforest = IsolationForest(n_estimators=100)

iforest.fit(x)

scores = iforest.score_samples(x)

thresh = quantile(scores, 0.02)

index = where(scores <= thresh)
values = x[index]

plt.scatter(x[:,0], x[:,1])
plt.scatter(values[:,0], values[:,1], color='red')
plt.title("Anomaly Detection (Method 2)")
plt.show()
