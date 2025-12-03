import os, json, cv2, numpy as np

def _pre(gray):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    g = clahe.apply(gray)
    g = cv2.medianBlur(g, 3)
    g = cv2.addWeighted(g, 1.5, cv2.GaussianBlur(g,(0,0),3), -0.5, 0)
    return g

def _crop_body(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),0)
    th = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_MEAN_C,
                               cv2.THRESH_BINARY_INV,35,7)
    cs,_ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cs: return gray, (0,0,gray.shape[1],gray.shape[0])
    c = max(cs, key=cv2.contourArea)
    x,y,w,h = cv2.boundingRect(c)
    pad = int(0.12*max(w,h))
    x0,y0 = max(0,x+pad), max(0,y+pad)
    x1,y1 = min(img.shape[1],x+w-pad), min(img.shape[0],y+h-pad)
    return gray[y0:y1, x0:x1], (x0,y0,x1,y1)

def run_ocr(image_path, out_json, dbg_dir):
    os.makedirs(dbg_dir, exist_ok=True)
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    crop,_ = _crop_body(img)
    prep = _pre(crop)
    cv2.imwrite(os.path.join(dbg_dir,"preprocessed.png"), prep)

    text, engine, ok = "", None, False
    try:
        import easyocr
        r = easyocr.Reader(['en'], gpu=False)
        lines = r.readtext(prep, detail=0, paragraph=True)
        text = "\n".join(lines); engine, ok = "easyocr", True
    except Exception as e:
        text = f"[EasyOCR not executed: {e}]"

    if not ok:
        try:
            import pytesseract
            cfg = "--psm 6 -c preserve_interword_spaces=1"
            text = pytesseract.image_to_string(prep, config=cfg)
            engine, ok = "tesseract", True
        except Exception as e:
            text = f"[OCR failed: {e}]"

    res = {"image": os.path.basename(image_path), "ocr_ok": ok, "engine": engine, "text": text}
    open(out_json,"w").write(json.dumps(res, indent=2))
    return res

if __name__ == "__main__":
    import argparse, json
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("-o","--out", default="ocr.json")
    ap.add_argument("-d","--debug_dir", default="ocr_dbg")
    args = ap.parse_args()
    print(json.dumps(run_ocr(args.image, args.out, args.debug_dir), indent=2))
