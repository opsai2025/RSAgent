"""Real LAE-DINO regression smoke, without router/VLM endpoint calls."""
import json
from pathlib import Path
import numpy as np
from PIL import Image
import config
import drawing
import vlm
from tools import detection


def main():
    folder=Path("validation/detection_smoke").resolve()
    folder.mkdir(parents=True,exist_ok=True)
    source=Path(config.OBJECT_DETECTION_DIR)/"Google_Earth/Capture06.PNG"
    with Image.open(source) as src:
        image=src.convert("RGB")
    image.save(folder/"input.png")
    try:
        out=detection.detect(str(folder/"input.png"),["small vehicle","large vehicle"])
        assert len(out.raw)>=len(out.nms)
        assert [d["id"] for d in out.nms]==["d%d"%i for i in range(len(out.nms))]
        assert all(0<=x<=1 for d in out.nms for x in d["bbox"])
        payload=vlm.vlm_detections(out)
        assert all(not d["id"].startswith("raw") for d in payload)
        drawing.draw_detections(image,out.nms).save(folder/"overlay.png")
        info={"raw":len(out.raw),"nms":len(out.nms),"counts":out.counts(),
              "seconds":out.seconds,"worker":detection.worker().load_info,"vlm":"not called"}
        (folder/"smoke_result.json").write_text(json.dumps(info,indent=2))
        print(json.dumps(info,indent=2),flush=True)
    finally:
        detection.worker().close()


if __name__=="__main__":
    main()
