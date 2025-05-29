# 1. Imports
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalMaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from google.colab import files

# 2. Load data and prepare labels
filenames = os.listdir("data/train")
categories = ["1" if f.startswith("dog") else "0" for f in filenames]
df = pd.DataFrame({"filename": filenames, "category": categories})

# # 3. Subsample to speed up training
# df = df.sample(frac=0.1, random_state=42)

# 4. Split into training and validation sets
train_df, validate_df = train_test_split(df, test_size=0.1, random_state=42)
train_df = train_df.reset_index(drop=True)
validate_df = validate_df.reset_index(drop=True)

# 5. Parameters
image_size = 224
batch_size = 16

# 6. Image generators
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    shear_range=0.2,
    zoom_range=0.1,
    horizontal_flip=True,
    width_shift_range=0.2,
    height_shift_range=0.2
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_dataframe(
    train_df,
    "data/train/",
    x_col='filename',
    y_col='category',
    target_size=(image_size, image_size),
    class_mode='binary',
    batch_size=batch_size
)

val_generator = val_datagen.flow_from_dataframe(
    validate_df,
    "data/train/",
    x_col='filename',
    y_col='category',
    target_size=(image_size, image_size),
    class_mode='binary',
    batch_size=batch_size
)

# 7. Load VGG16 model without top layers
vgg = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
for layer in vgg.layers[:15]:
    layer.trainable = False

x = GlobalMaxPooling2D()(vgg.output)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(1, activation='sigmoid')(x)
model = Model(inputs=vgg.input, outputs=output)

model.compile(
    loss='binary_crossentropy',
    optimizer=SGD(learning_rate=1e-4, momentum=0.9),
    metrics=['accuracy']
)

# 8. Callbacks
callbacks = [
    EarlyStopping(monitor='val_loss', patience=4),
    ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=2, verbose=1, min_lr=1e-5)
]

# 9. Training
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10,
    callbacks=callbacks
)

# 10. Evaluation and performance report
val_loss, val_acc = model.evaluate(val_generator)
print(f"Final validation accuracy: {val_acc:.4f}")

# Predictions on validation set
val_generator.reset()
pred_probs = model.predict(val_generator, steps=len(val_generator), verbose=1)
y_pred = (pred_probs > 0.5).astype(int).reshape(-1)
y_true = validate_df['category'].astype(int).values

# Classification report
print("\nClassification report:\n")
print(classification_report(y_true, y_pred, target_names=["cat", "dog"]))

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["cat", "dog"], yticklabels=["cat", "dog"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# 11. Predictions on test set
test_filenames = os.listdir("data/test1")
test_df = pd.DataFrame({"filename": test_filenames})

test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_dataframe(
    test_df,
    "data/test1/",
    x_col='filename',
    y_col=None,
    class_mode=None,
    target_size=(image_size, image_size),
    batch_size=batch_size,
    shuffle=False
)

preds = model.predict(test_generator, steps=int(np.ceil(len(test_df) / batch_size)))
test_df['label'] = np.where(preds > 0.5, 1, 0)
test_df['id'] = test_df['filename'].str.extract(r'(\d+)').astype(int)
test_df[['id', 'label']].sort_values("id").to_csv("submission.csv", index=False)

# 12. Download CSV file
files.download("submission.csv")
print("CSV file submission.csv generated and ready to download.")

# 13. Save model in Keras format (.keras)
model.save("model_vgg16.keras")
files.download("model_vgg16.keras")
print("Model saved as .keras and ready to download.")

