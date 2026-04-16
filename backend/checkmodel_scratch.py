import tensorflow as tf

def check_model():
    print("Loading model.h5...")
    try:
        model = tf.keras.models.load_model('model.h5', compile=False) 
        model.summary()
        print("Input shape:", model.input_shape)
        print("Output shape:", model.output_shape)
    except Exception as e:
        print("Failed to load model:", e)

if __name__ == "__main__":
    check_model()
