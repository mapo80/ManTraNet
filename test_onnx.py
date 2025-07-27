import os
import csv
import cv2
import numpy as np
import onnxruntime as ort
from datetime import datetime


def read_rgb_image(path):
    rgb = cv2.imread(path, 1)[..., ::-1]
    return rgb


def predict(img, session):
    img = cv2.resize(img, (256, 256))
    x = img.astype(np.float32) / 255.0 * 2 - 1
    x = np.expand_dims(x, 0)
    y = session.run(None, {"img_in": x})[0]
    return y[0, ..., 0]


def main():
    model_path = os.path.join("pretrained_weights", "ManTraNet_Ptrain4.onnx")
    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])

    forged_scores = []
    orig_scores = []
    timings = []
    with open("data/samplePairs.csv", newline="") as f:
        reader = csv.reader(f)
        for idx, (forged_rel, orig_rel) in enumerate(reader):
            if idx >= 3:
                break
            forged_img = read_rgb_image(os.path.join("data", forged_rel))
            orig_img = read_rgb_image(os.path.join("data", orig_rel))
            start = datetime.now()
            forged_pred = predict(forged_img, session)
            forged_time = (datetime.now() - start).total_seconds()
            start = datetime.now()
            orig_pred = predict(orig_img, session)
            orig_time = (datetime.now() - start).total_seconds()
            timings.append((forged_time, orig_time))
            forged_scores.append(forged_pred.mean())
            orig_scores.append(orig_pred.mean())
    print("Mean forged score: {:.6f}".format(np.mean(forged_scores)))
    print("Mean original score: {:.6f}".format(np.mean(orig_scores)))
    for i, (ft, ot) in enumerate(timings, 1):
        print(f"Image {i} forged time: {ft:.4f}s, original time: {ot:.4f}s")


if __name__ == "__main__":
    main()
