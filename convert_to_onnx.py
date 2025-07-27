import tensorflow as tf
from src.modelCore import load_pretrain_model_by_index
import tf2onnx

if __name__ == '__main__':
    model = load_pretrain_model_by_index(4, 'pretrained_weights', is_dynamic_shape=False)
    # Build the model with a dummy input to ensure shapes are set
    dummy = tf.zeros([1, 256, 256, 3], dtype=tf.float32)
    model(dummy)
    # Convert model with static input size to avoid shape issues
    spec = (tf.TensorSpec((1, 256, 256, 3), tf.float32, name="img_in"),)
    onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13)
    with open('pretrained_weights/ManTraNet_Ptrain4.onnx', 'wb') as f:
        f.write(onnx_model.SerializeToString())
    print('ONNX model saved to pretrained_weights/ManTraNet_Ptrain4.onnx')
