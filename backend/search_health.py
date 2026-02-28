lines = open("main.py", encoding="utf-8").readlines()
res = [(i+1, l.rstrip()) for i, l in enumerate(lines) if "degraded" in l or "/health" in l]
for r in res[:30]:
    print(r[0], r[1])
