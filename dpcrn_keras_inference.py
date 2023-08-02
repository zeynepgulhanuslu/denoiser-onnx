import keras
from keras.models import Model


def convert_functional_model(model):
    """
    Convert a Keras Functional model to a regular Keras model.

    Args:
        model: A Keras Functional model.

    Returns:
        A regular Keras model.
    """

    # Get the input and output tensors of the model.
    input_tensor = model.inputs[0]
    output_tensor = model.outputs[0]

    # Create a regular Keras model from the input and output tensors.
    model = Model(input_tensor, output_tensor)

    return model


if __name__ == '__main__':
    dpcrn_keras_func_model_file = "dpcrn_keras_model_func.h5"
    dpcrn_keras_model_file = "dpcrn_keras_model.h5"
    # Load the Keras model.
    model = keras.models.load_model(dpcrn_keras_func_model_file)
    model.reset_states()

    keras_model = convert_functional_model(model)
    print(keras_model.summary())
    keras_model.save(dpcrn_keras_model_file)
    print(type(keras_model))

