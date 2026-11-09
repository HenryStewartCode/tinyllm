import json, sys
a = json.load(open(sys.argv[1])); b = json.load(open(sys.argv[2]))
def norm(x):
    calls = [(n, json.loads(args) if args.strip() else None) for n, args in x["calls"]]
    return (x["content"] or "", calls, x["finish"])
same = total = 0
for key in a:
    xs, ys = a[key], b[key]
    if isinstance(xs, dict): xs, ys = [xs], [ys]
    s = sum(norm(x) == norm(y) for x, y in zip(xs, ys))
    same += s; total += len(xs)
    print(f"  {key}: {s}/{len(xs)} the same")
    for x, y in zip(xs, ys):
        if norm(x) != norm(y):
            print("     A:", json.dumps(norm(x))[:170]); print("     B:", json.dumps(norm(y))[:170])
print(f"same in {same} of {total}")
