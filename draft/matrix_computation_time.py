import time
import torch

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available.")

device = "cuda"
size = 1024
iterations = 10

print("tensor generation")

torch.cuda.synchronize()
gen_start = time.perf_counter()

a = torch.rand(size, size, device=device)
b = torch.rand(size, size, device=device)

torch.cuda.synchronize()
gen_end = time.perf_counter()

print(f"tensor generation time: {(gen_end - gen_start) * 1000:.3f} ms")

print("sleep")
time.sleep(2)

for _ in range(10):
    _ = a @ b
torch.cuda.synchronize()

print("run benchmark")

torch.cuda.synchronize()
bench_start = time.perf_counter()

for _ in range(iterations):
    _ = a @ b

torch.cuda.synchronize()
bench_end = time.perf_counter()

bench_time = bench_end - bench_start

print(f"Matrix multiplication time: {bench_time:.4f} s")
print(f"Average per matmul: {bench_time / iterations * 1000:.3f} ms")
print(f"Throughput: {iterations / bench_time:.2f} matmuls/s")