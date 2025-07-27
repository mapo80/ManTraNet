# Developer Notes

This repository has been updated to run with modern TensorFlow and Keras (tested with TensorFlow 2.16.1 and Keras 3.10 on Python 3.12). The legacy Keras backend functions have been replaced with TensorFlow equivalents.

## Converting pretrained weights to ONNX

Run `python convert_to_onnx.py` after installing the dependencies listed in the README. The script loads `pretrained_weights/ManTraNet_Ptrain4.h5` and produces `pretrained_weights/ManTraNet_Ptrain4.onnx`.

To quickly verify the ONNX model you can run `python test_onnx.py`. The script uses a few images from the `data` directory, prints the average forged and original scores and reports how long each inference took.

## Testing

No automated tests are provided. After modifying the code, ensure that `python convert_to_onnx.py` runs successfully and creates the ONNX file.
To test the ONNX runtime from .NET build the CLI:

```bash
dotnet run -c Release --project ManTraNetCLI
```

This prints mean scores for a few sample images.
