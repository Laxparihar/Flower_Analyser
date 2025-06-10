from tensorflow.keras.applications import VGG16, ResNet50, InceptionV3
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_preprocess
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.inception_v3 import preprocess_input as inception_preprocess
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd

# Resize images to 224x224 for VGG16 and ResNet50, 299x299 for Inception
def resize_dataset(x, size):
    return tf.image.resize(x, size).numpy()

# Preprocessing setup
input_shapes = {
    'VGG16': ((224, 224), vgg_preprocess),
    'ResNet50': ((224, 224), resnet_preprocess),
    'InceptionV3': ((299, 299), inception_preprocess)
}

def train_model(base_model_class, model_name):
    img_size, preprocess_func = input_shapes[model_name]
    
    # Resize and preprocess
    x_train_resized = resize_dataset(x_train, img_size)
    x_test_resized = resize_dataset(x_test, img_size)
    
    x_train_prep = preprocess_func(x_train_resized)
    x_test_prep = preprocess_func(x_test_resized)
    
    base_model = base_model_class(weights='imagenet', include_top=False, input_shape=img_size + (3,))
    base_model.trainable = False
    
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(64, activation='relu'),
        layers.Dense(4, activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train_prep, y_train, epochs=5, validation_data=(x_test_prep, y_test), verbose=1)
    
    y_pred = np.argmax(model.predict(x_test_prep), axis=1)
    
    report = classification_report(y_test, y_pred, output_dict=True)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    print(f"\n== {model_name} Report ==\n")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", conf_matrix)
    
    return report

# Run all models
vgg_report = train_model(VGG16, "VGG16")
resnet_report = train_model(ResNet50, "ResNet50")
inception_report = train_model(InceptionV3, "InceptionV3")
