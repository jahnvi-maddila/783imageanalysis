import os, glob, json
from pin_detect import detect_pins
from text_ocr import run_ocr

SAMPLES = "samples"
OUT = "outputs"

def main():
    os.makedirs(OUT, exist_ok=True)
    imgs = sorted(glob.glob(os.path.join(SAMPLES,"*.png")) + glob.glob(os.path.join(SAMPLES,"*.jpg")))
    summary = []
    for p in imgs:
        base = os.path.splitext(os.path.basename(p))[0]
        pins_png = os.path.join(OUT, f"{base}_pins.png")
        pins_json = os.path.join(OUT, f"{base}_pins.json")
        ocr_json  = os.path.join(OUT, f"{base}_ocr.json")
        ocr_dbg   = os.path.join(OUT, f"{base}_ocr_dbg")

        r1 = detect_pins(p, pins_png, pins_json)
        r2 = run_ocr(p, ocr_json, ocr_dbg)
        summary.append({"image": base, "pins": r1, "ocr": r2})

    open(os.path.join(OUT,"summary.json"),"w").write(json.dumps(summary, indent=2))
    print("Done. See outputs/")

if __name__ == "__main__":
    main()
