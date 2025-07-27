# ManTraNet: Manipulation Tracing Network for Detection and Localization of Image Forgeries

<img src="https://www.isi.edu/images/isi-logo.jpg" width="300"/> <img src="http://cvpr2019.thecvf.com/images/CVPRLogo.png" width="300"/>

---

This repository contains the official implementation of **ManTraNet**, presented at CVPR 2019. The network predicts a pixel-wise forgery likelihood map from a single image.

For detailed information please refer to the following citation:

```text
@inproceedings{Wu2019ManTraNet,
    title={ManTra-Net: Manipulation Tracing Network For Detection And Localization of Image ForgeriesWith Anomalous Features},
    author={Yue Wu, Wael AbdAlmageed, and Premkumar Natarajan},
    journal={The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
    year={2019}
}
```

---

## Overview

ManTraNet is an end-to-end solution for image forgery detection and localization. It takes an image of arbitrary size as input and outputs a forgery heat map. The method offers:

1. **Simplicity** – no additional pre/post processing is required;
2. **Speed** – the entire computation is performed by a single network;
3. **Robustness** – it relies only on the assumption that a region in the image is manipulated differently from the rest.

![Sample results](https://github.com/ISICV/ManTraNet/blob/master/data/result0.png) ![Sample results](https://github.com/ISICV/ManTraNet/blob/master/data/result1.png)

### Architecture

The network is composed of two main blocks:

1. **Image Manipulation Trace Feature Extractor** – trained on an image manipulation classification task and sensitive to various editing operations. It produces a fixed-length feature vector for each patch.
2. **Local Anomaly Detection Network** – compares a local feature to the dominant feature averaged from its neighborhood and activates based on the deviation rather than the absolute value.

![Architecture](https://github.com/ISICV/ManTraNet/blob/master/data/ManTraNet-overview.png)

### Extending the model

The released weights are pretrained entirely on synthetic data:

1. Pretrain the Image Manipulation Classification task over [385 classes](https://github.com/ISICV/ManTraNet/blob/master/data/IMC385.png).
2. Train ManTraNet end-to-end on copy-move, splicing, removal and enhancement forgeries.

New manipulation types can be introduced at either stage. The IMC task can also be treated as a **self-supervised** learning problem.

## Requirements

The code is written for Keras on top of TensorFlow. Tested versions are:

- **Python** 3.12
- **Keras** 3.10
- **TensorFlow** 2.16.1

Earlier versions may still work but are no longer maintained.

## Converting to ONNX

Install the required packages and run the conversion script:

```bash
pip install tensorflow==2.16.1 keras==3.10.0 tf2onnx
python convert_to_onnx.py
```

This loads `pretrained_weights/ManTraNet_Ptrain4.h5` and produces `pretrained_weights/ManTraNet_Ptrain4.onnx`.

## Testing the ONNX model

After conversion you can verify the exported model on a few sample images:

```bash
pip install onnxruntime opencv-python-headless
python test_onnx.py
```

The script prints the mean score for forged and original examples and reports the inference time for each image.

## .NET CLI

The `ManTraNetCLI` directory contains a small .NET 8 command line program. After building it with:

```bash
dotnet run -c Release --project ManTraNetCLI
```

the program loads `pretrained_weights/ManTraNet_Ptrain4.onnx` and runs inference on the sample pairs listed in `data/samplePairs.csv`.

## Demo

You can explore ManTraNet interactively using the supplied Jupyter notebook or via [Google Colab](https://colab.research.google.com/drive/1ai4kVlI6w9rREqqYnTfpk3gM3YX9k-Ek).

## Contact

For paper related questions please contact **rex.yue.wu(AT)gmail.com**.

## License

The software is provided for academic or non-commercial purposes only. Individuals seeking a commercial license must contact USC Stevens Institute for Innovation.

```
USC Stevens Institute for Innovation
University of Southern California
1150 S. Olive Street, Suite 2300
Los Angeles, CA 90115, USA
ATTN: Accounting
```

DISCLAIMER. USC MAKES NO EXPRESS OR IMPLIED WARRANTIES, INCLUDING BUT NOT LIMITED TO MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE. THE SOFTWARE IS PROVIDED "AS IS" AND USC SHALL NOT BE LIABLE FOR ANY INCIDENTAL OR CONSEQUENTIAL DAMAGES.

For commercial license pricing and support please contact:

```
Rakesh Pandit
USC Stevens Institute for Innovation
University of Southern California
1150 S. Olive Street, Suite 2300
Los Angeles, CA 90115, USA
Tel: +1 213-821-3552
Fax: +1 213-821-5001
Email: rakeshvp@usc.edu (cc: accounting@stevens.usc.edu)
```

---

## Important Notice

1. **Training code, datasets and evaluation scripts are not released**. For training code or commercial usage please contact USC ISI. The released repository contains only the inference code.
2. **Pretrained architecture difference**. The model provided here uses 32 filters in the first block (IMC-VGG-W&D setting from Table 5 of the paper). This differs from the architecture described in the paper, which may lead to slightly different evaluation numbers.
