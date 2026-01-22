from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout

def build_model(num_classes):
    model = Sequential()

    model.add(Conv2D(32, (3,3), activation="relu", input_shape=(128,128,1)))
    model.add(MaxPooling2D(2,2))

    model.add(Conv2D(64, (3,3), activation="relu"))
    model.add(MaxPooling2D(2,2))

    model.add(Conv2D(128, (3,3), activation="relu"))
    model.add(MaxPooling2D(2,2))

    model.add(Flatten())
    model.add(Dense(256, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation="softmax"))

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model
