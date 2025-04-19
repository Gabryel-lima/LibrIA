import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # força uso de CPU

import warnings
warnings.filterwarnings("ignore")

import glob
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import string
from sklearn.model_selection import train_test_split


from conf import CFG, seed_everything

"""Api keras old"""
# from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
# from tensorflow.keras.applications import VGG16
# from tensorflow.keras.models import Model, load_model
# from tensorflow.keras.layers import Dense, Flatten, Dropout
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.callbacks import ModelCheckpoint

"""Api keras new"""
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.api.applications import VGG16
from keras.api.models import Model, load_model
from keras.api.layers import Dense, Flatten, Dropout
from keras.api.optimizers import Adam
from keras.api.callbacks import ModelCheckpoint

if __name__ == "__main__":
    # Labels
    labels = []
    alphabet = list(string.ascii_uppercase)
    labels.extend(alphabet)
    labels.extend(["del", "nothing", "space"])
    print(labels)

    def sample_images(labels):
        # Create Subplots
        y_size = 12
        if(len(labels)<10):
            y_size = y_size * len(labels) / 10
        fig, axs = plt.subplots(len(labels), 9, figsize=(y_size, 13))

        for i, label in enumerate(labels):
            axs[i, 0].text(0.5, 0.5, label, ha='center', va='center', fontsize=8)
            axs[i, 0].axis('off')

            label_path = os.path.join(CFG.TRAIN_PATH, label)
            list_files = os.listdir(label_path)

            for j in range(8):
                img_label = cv2.imread(os.path.join(label_path, list_files[j]))
                img_label = cv2.cvtColor(img_label, cv2.COLOR_BGR2RGB)
                axs[i, j+1].imshow(img_label)
                axs[i, j+1].axis("off")

        # Title
        plt.suptitle("Sample Images in ASL Alphabet Dataset", x=0.55, y=0.92)

        # Show
        plt.show()
        

    # Create Metadata
    list_path = []
    list_labels = []
    for label in labels:
        label_path = os.path.join(CFG.TRAIN_PATH, label, "*")
        image_files = glob.glob(label_path)
        
        sign_label = [label] * len(image_files)
        
        list_path.extend(image_files)
        list_labels.extend(sign_label)

    metadata = pd.DataFrame({
        "image_path": list_path,
        "label": list_labels
    })

    # Split Dataset to Train 0.7, Val 0.15, and Test 0.15
    X_train, X_test, y_train, y_test = train_test_split(
        metadata["image_path"], metadata["label"], 
        test_size=0.15, 
        random_state=2023, 
        shuffle=True, 
        stratify=metadata["label"]
    )
    data_train = pd.DataFrame({
        "image_path": X_train,
        "label": y_train
    })

    X_train, X_val, y_train, y_val = train_test_split(
        data_train["image_path"], data_train["label"],
        test_size=0.15/0.70,
        random_state=2023,
        shuffle=True,
        stratify=data_train["label"]
    )
    data_train = pd.DataFrame({
        "image_path": X_train,
        "label": y_train
    })
    data_val = pd.DataFrame({
        "image_path": X_val,
        "label": y_val
    })
    data_test = pd.DataFrame({
        "image_path": X_test,
        "label": y_test
    })

    # Data Augmentation (Just Rescale)
    def data_augmentation():
        datagen = ImageDataGenerator(rescale=1/255.,)
        # Training Dataset
        train_generator = datagen.flow_from_dataframe(
            data_train,
            directory=None,
            x_col="image_path",
            y_col="label",
            class_mode="categorical",
            batch_size=CFG.batch_size,
            target_size=(CFG.img_height, CFG.img_width),
        )

        # Validation Dataset
        validation_generator = datagen.flow_from_dataframe(
            data_val,
            directory=None,
            x_col="image_path",
            y_col="label",
            class_mode="categorical",
            batch_size=CFG.batch_size,
            target_size=(CFG.img_height, CFG.img_width),
        )
        
        # Testing Dataset
        test_generator = datagen.flow_from_dataframe(
            data_test,
            directory=None,
            x_col="image_path",
            y_col="label",
            class_mode="categorical",
            batch_size=1,
            target_size=(CFG.img_height, CFG.img_width),
            shuffle=False
        )
        
        return train_generator, validation_generator, test_generator

    seed_everything(2023)
    train_generator, validation_generator, test_generator = data_augmentation()

    # Load VGG16 model and modify for ASL recognition
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(CFG.img_height, CFG.img_width, CFG.img_channels))

    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(29, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    # Compile and train the model
    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

    # Callbacks
    checkpoint = ModelCheckpoint('./src/saved/asl_vgg16_full_model.keras', save_best_only=True, monitor='val_accuracy', mode='max')

    # Train the Model
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // CFG.batch_size,
        epochs=CFG.epochs,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // CFG.batch_size,
        callbacks=[checkpoint]
    )

    scores = model.evaluate(test_generator)
    print("%s: %.2f%%" % ("Evaluate Test Accuracy", scores[1]*100))
    
    # Save the Model
    model.save("src/saved/asl_vgg16.keras")
