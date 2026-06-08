from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_generator = datagen.flow_from_directory(
    "dataset",
    target_size=(224,224),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

val_generator = datagen.flow_from_directory(
    "dataset",
    target_size=(224,224),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(224,224,3)
)

base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)

predictions = Dense(7, activation='softmax')(x)

model = Model(
    inputs=base_model.input,
    outputs=predictions
)

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("Model Created Successfully")
model.summary()


history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10
)

model.save("cloud_model.keras")

print("Model Saved Successfully")