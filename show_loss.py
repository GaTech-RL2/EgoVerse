import csv, sys
rows = list(csv.DictReader(open(sys.argv[1])))
out = []
for r in rows:
    al = r.get('Train/action_loss', '')
    tl = r.get('Train/Loss', '')
    ep = r.get('epoch', '')
    if al not in ('', None):
        out.append((ep, al, tl))
# print first 3 and last 8 to see the trend
for ep, al, tl in out[:3]:
    print(f"ep={ep} action_loss={al} total={tl}")
print("...")
for ep, al, tl in out[-8:]:
    print(f"ep={ep} action_loss={al} total={tl}")
