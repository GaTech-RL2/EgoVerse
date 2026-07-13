import cv2, sys, os
path = sys.argv[1]; outdir = sys.argv[2]
os.makedirs(outdir, exist_ok=True)
cap = cv2.VideoCapture(path)
n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print("frames", n)
for fn in [1, n // 3, 2 * n // 3]:
    cap.set(cv2.CAP_PROP_POS_FRAMES, fn)
    r, f = cap.read()
    if r:
        cv2.imwrite(outdir + "/vf%d.png" % fn, f)
        print("wrote", fn, f.shape)
