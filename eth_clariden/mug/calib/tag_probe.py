import h5py, numpy as np, cv2
print("cv2", cv2.__version__, "aruco:", hasattr(cv2, "aruco"))
f = h5py.File("/cluster/scratch/agavryushin/ben_icra/demos/20260913_152109.h5")
for cam in ("front", "wrist"):
    K = f[f"camera/intrinsics/{cam}"][:]; print(cam, "intrinsics vector:", np.array2string(K, precision=3, max_line_width=200))
front = f["camera/front"][0]; wrist = f["camera/wrist"][0]
if hasattr(cv2, "aruco"):
    dicts = {n: getattr(cv2.aruco, n) for n in dir(cv2.aruco) if n.startswith("DICT_")}
    for name, img in (("front", front), ("wrist", wrist)):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        for dn, dv in dicts.items():
            try:
                det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(dv), cv2.aruco.DetectorParameters())
                corners, ids, _ = det.detectMarkers(gray)
            except Exception as e:
                corners, ids = [], None
            if ids is not None and len(ids):
                print(f"  {name}: {dn} -> ids {ids.ravel().tolist()} corners[0] {corners[0].reshape(-1,2).round(1).tolist()}")
    # how many frames of the wrist cam see a tag (36h11 if found)?
