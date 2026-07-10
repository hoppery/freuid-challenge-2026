import numpy as np, pandas as pd, cv2

a = pd.read_parquet("/tmp/pub_scores.parquet").rename(columns={"score": "s1"})
b = pd.read_parquet("/tmp/pub_e2_scores.parquet").rename(columns={"e2_score": "s2"})
df = a.merge(b, on="id")
TEST = "data/raw/freuid/public_test/public_test"
df["path"] = df["id"].map(lambda i: f"{TEST}/{i}.jpeg")


def montage(rows, title, fname, cols=5):
    rows = rows.head(cols * 3)
    th, pad, lab = 240, 4, 26
    n = len(rows)
    ncol = cols
    nrow = (n + ncol - 1) // ncol
    canvas = np.full((max(1, nrow) * (th + lab + pad), ncol * (th + pad), 3), 30, np.uint8)
    for k, (_, r) in enumerate(rows.iterrows()):
        img = cv2.imread(r["path"])
        if img is None:
            continue
        h, w = img.shape[:2]
        sc = th / max(h, w)
        img = cv2.resize(img, (int(w * sc), int(h * sc)))
        ph, pw = img.shape[:2]
        row, col = k // ncol, k % ncol
        y0 = row * (th + lab + pad)
        x0 = col * (th + pad)
        canvas[y0 + lab:y0 + lab + ph, x0:x0 + pw] = img
        cv2.putText(canvas, f"f={r['s1']:.2f} e2={r['s2']:.2f}", (x0 + 2, y0 + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    cv2.imwrite(fname, canvas)
    print(f"  {fname}: {n} imgs — {title}")


ca = df[(df.s1 > 0.99) & (df.s2 > 0.99)].sample(15, random_state=1)
montage(ca, "confident ATTACK (both >0.99)", "/tmp/grid_A_attack.png")
cg = df[(df.s1 < 0.01) & (df.s2 < 0.01)].sample(15, random_state=1)
montage(cg, "confident GENUINE (both <0.01)", "/tmp/grid_B_genuine.png")
un = df[(df.s1 > 0.1) & (df.s1 < 0.9)]
montage(un, "UNCERTAIN (FREUID-only 0.1-0.9)", "/tmp/grid_C_uncertain.png")
dis = df[((df.s1 < 0.2) & (df.s2 > 0.8)) | ((df.s1 > 0.8) & (df.s2 < 0.2))]
print(f"\n  strong-disagreement count: {len(dis)}")
if len(dis) > 0:
    montage(dis, "DISAGREEMENT (models flip)", "/tmp/grid_D_disagree.png")
