import cv2, json, os
import numpy as np

def _load_gray(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None: raise FileNotFoundError(path)
    return img, cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def _biggest_contour(gray):
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    th = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 35, 5
    )
    cs, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cs: return None
    return max(cs, key=cv2.contourArea)

def _prob(bw,bh,area,align_off,w_med,h_med,a_med,tol):
    size_z = max(abs(bw-w_med)/max(1e-6,w_med),
                 abs(bh-h_med)/max(1e-6,h_med))
    area_z = abs(area-a_med)/max(1e-6,a_med)
    align_z = abs(align_off)/max(1.0, tol)
    score = 0.4*size_z + 0.3*area_z + 0.3*align_z
    score = min(1.5, score)
    return min(1.0, score/0.8)  # map to [0,1]

def detect_pins(image_path, out_image_path, out_json_path=None, p_thresh=0.5):
    img, gray = _load_gray(image_path)
    body = _biggest_contour(gray)
    if body is None:
        cv2.imwrite(out_image_path, img)
        return {"error":"no IC body found"}

    x,y,w,h = cv2.boundingRect(body)
    pad = int(0.12*max(w,h))  # ring thickness
    x0,y0 = max(0,x-pad), max(0,y-pad)
    x1,y1 = min(img.shape[1],x+w+pad), min(img.shape[0],y+h+pad)
    roi = img[y0:y1, x0:x1]
    g = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # highlight shiny pins
    tophat = cv2.morphologyEx(g, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_RECT,(7,7)))
    edges  = cv2.Canny(tophat, 50, 150)
    otsu   = cv2.threshold(tophat, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    comb   = cv2.bitwise_or(edges, otsu)

    # closing + dilation (recommended hint)
    k3 = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    comb = cv2.morphologyEx(comb, cv2.MORPH_CLOSE, k3, 1)
    comb = cv2.dilate(comb, k3, 1)

    # remove body interior (keep ring)
    ring = comb.copy()
    cv2.rectangle(ring, (pad,pad), (pad+w,pad+h), 0, -1)

    cnts,_ = cv2.findContours(ring, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in cnts:
        bx,by,bw,bh = cv2.boundingRect(c)
        area = bw*bh
        if area < 10 or area > 0.04*(w*h):    # throw obvious noise/merges
            continue
        aspect = max(bw,bh)/max(1,min(bw,bh))
        if aspect < 1.3:                      # pins are skinny
            continue
        # must touch ring border
        if not (bx<pad or by<pad or (bx+bw)>(pad+w) or (by+bh)>(pad+h)):
            continue
        boxes.append((bx,by,bw,bh))

    # group by nearest border of ROI
    W,H = (x1-x0),(y1-y0)
    groups = {"top":[], "bottom":[], "left":[], "right":[]}
    for (bx,by,bw,bh) in boxes:
        d = {"top":by, "left":bx, "bottom":H-(by+bh), "right":W-(bx+bw)}
        side = min(d, key=d.get)
        groups[side].append((bx,by,bw,bh))

    consistent, inconsistent, info = [], [], []
    anno = img.copy()
    cv2.rectangle(anno,(x0,y0),(x1,y1),(255,0,0),1)

    for side, arr in groups.items():
        if not arr: continue
        ws = np.array([w for (_,_,w,_) in arr])
        hs = np.array([h for (_,_,_,h) in arr])
        areas = ws*hs
        w_med,h_med,a_med = np.median(ws),np.median(hs),np.median(areas)
        centers = np.array([[bx+bw/2, by+bh/2] for (bx,by,bw,bh) in arr], dtype=np.float32)

        if side in ("top","bottom"):
            y_line = np.median(centers[:,1]); tol = max(3, 0.12*h_med)
            for (bx,by,bw,bh),(cx,cy) in zip(arr,centers):
                p = _prob(bw,bh,bw*bh, cy-y_line, w_med,h_med,a_med,tol)
                status = "inconsistent" if p>=p_thresh else "consistent"
                (inconsistent if status=="inconsistent" else consistent).append((bx,by,bw,bh))
                info.append({"side":side,"box":[int(bx),int(by),int(bw),int(bh)],
                             "area":int(bw*bh),"prob_inconsistent":round(float(p),3),
                             "status":status})
        else:
            x_line = np.median(centers[:,0]); tol = max(3, 0.12*w_med)
            for (bx,by,bw,bh),(cx,cy) in zip(arr,centers):
                p = _prob(bw,bh,bw*bh, cx-x_line, w_med,h_med,a_med,tol)
                status = "inconsistent" if p>=p_thresh else "consistent"
                (inconsistent if status=="inconsistent" else consistent).append((bx,by,bw,bh))
                info.append({"side":side,"box":[int(bx),int(by),int(bw),int(bh)],
                             "area":int(bw*bh),"prob_inconsistent":round(float(p),3),
                             "status":status})

    # draw
    def shift(bxs): return [(x0+bx, y0+by, bw, bh) for (bx,by,bw,bh) in bxs]
    for (bx,by,bw,bh) in shift(consistent):
        cv2.rectangle(anno,(bx,by),(bx+bw,by+bh),(0,255,0),2)
    for (bx,by,bw,bh) in shift(inconsistent):
        cv2.rectangle(anno,(bx,by),(bx+bw,by+bh),(0,0,255),2)

    cv2.imwrite(out_image_path, anno)
    result = {"image": os.path.basename(image_path),
              "pins_total": len(consistent)+len(inconsistent),
              "consistent": len(consistent),
              "inconsistent": len(inconsistent),
              "pins": info}
    if out_json_path: open(out_json_path,"w").write(json.dumps(result, indent=2))
    return result

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("-o","--out", default="annotated.png")
    ap.add_argument("-j","--json", default=None)
    ap.add_argument("--pthresh", type=float, default=0.5)
    args = ap.parse_args()
    print(json.dumps(detect_pins(args.image, args.out, args.json, args.pthresh), indent=2))
