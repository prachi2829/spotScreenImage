import time
from predict import predict

IMAGE = "dataset/real/real_001.jpeg"

N = 100

start = time.perf_counter()

for _ in range(N):
    predict(IMAGE)

end = time.perf_counter()

latency = (end-start)/N*1000

print(f"Average Latency: {latency:.2f} ms")
