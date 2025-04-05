import tensorflow as tf

# resolution = (1920, 1080)
resolution = (1280, 720)
output_model_name = f'mobilenetv2_ssd_fixed_{resolution[0]}_{resolution[1]}.tflite'
model = tf.saved_model.load('./model')

concrete_func = model.signatures[tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
concrete_func.inputs[0].set_shape([None, resolution[0], resolution[1], 3])
converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])

try:
    tflite_model = converter.convert()

    with open(output_model_name, 'wb') as f:
        f.write(tflite_model)
    print("Model converted successfully.")
except Exception as e:
    with open('error.log', 'w') as f:
        f.write(str(e))
        f.write("\n")
    print("Failed: Error during conversion")
    raise
