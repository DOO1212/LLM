import hashlib

s = "HYD-002 유압 호스 3/8인치 유압/공압 C-1 최지원 정상"
h = hashlib.sha256(s.encode("utf-8")).digest()
out = []
for i in range(12):
    chunk = bytes([h[(i + j) % len(h)] for j in range(8)])
    u = int.from_bytes(chunk, "little") / (2**64)
    out.append(round(u * 2 - 1, 6))
with open(__file__.replace("_hash_out_runner.py", "_hash_out.txt"), "w", encoding="utf-8") as f:
    f.write("sha256=" + hashlib.sha256(s.encode()).hexdigest() + "\n")
    f.write("demo12=" + str(out) + "\n")
