
from PIL import Image
from lang_sam import LangSAM
import cv2 as cv
import numpy as np

model = LangSAM()
image_pil = Image.open("./assets/box.jpeg").convert("RGB")
text_prompt = "box."
results = model.predict([image_pil], [text_prompt])

image_cv = cv.cvtColor(np.array(image_pil), cv.COLOR_RGB2BGR)

for mask in results[0]["masks"]:
    mask = mask.astype(np.uint8)
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        x, y, w, h = cv.boundingRect(cnt)
        cv.rectangle(image_cv, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv.putText(image_cv, text_prompt, (x, y - 5), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

cv.imwrite("./assets/result.jpeg", image_cv)