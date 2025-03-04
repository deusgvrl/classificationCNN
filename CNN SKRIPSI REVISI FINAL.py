# ------- IMPORT LIBRARY -------
import os
import cv2  
import numpy as np
import tensorflow as tf
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from keras.api.models import Sequential, Model
from keras.api.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from keras.api.optimizers import Adam
from keras.api.callbacks import EarlyStopping
from keras.api.utils import to_categorical
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import plotly.express as px
# ------------------------------
dataset_dir = r"/mnt/c/Users/trkb/Documents/162012233070/Dataset Kopi/Segmented"
# -- MERGE RGB AND HSV VALUES --
def rgbhsv_merge(image):
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    hsv_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
    
    return np.concatenate((rgb_image, hsv_image), axis=-1)
# -------------------------------
# ------- DEFINE LABELS ---------
categories = ['Segmented Light', 'Segmented Medium', 'Segmented Dark']
images = []
labels = []
sample_filenames = {}

# ------- CHECK GPU STATUS -------
print("GPU yang tersedia : ", len(tf.config.list_physical_devices('GPU')))

# ------ FIND IMAGE IN PATH ------
for idx, category in enumerate(categories):
    imagecategory_path = os.path.join(dataset_dir, category)
    
    image_filenames = []
    for root, dirs, files in os.walk(imagecategory_path):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_filenames.append(os.path.join(root, file))
    
    if image_filenames:
        sample_filenames[category] = os.path.basename(image_filenames[0])  
        print(f"Jumlah data yang diambil pada folder {imagecategory_path} = {len(image_filenames)}")
    else:
        print(f"Folder {category} tidak memiliki file gambar!")
        continue 
# ----- RESIZE IMAGE -----
    for img_path in image_filenames:
        img = cv2.imread(img_path)
        img = cv2.resize(img, (256, 256))  
        img_rgb_hsv = rgbhsv_merge(img)  
        images.append(img_rgb_hsv)
# ----- APPEND LABEL TO BEANS -----
        if "Light" in category:
            labels.append(0)
        elif "Medium" in category:
            labels.append(1)
        elif "Dark" in category:
            labels.append(2)

# ------ 3D PLOTTING ------
class_colors = {"Light": "red", "Medium": "green", "Dark": "blue"}

# Initialize empty dataframes for RGB and HSV data
rgb_data = []
hsv_data = []

# Loop over each class to extract RGB and HSV values
for classLabel, beanClass in enumerate(["Light", "Medium", "Dark"]):
    beanIndex = np.where(np.array(labels) == classLabel)[0]
    if len(beanIndex) == 0:
        print(f"No images in: {beanClass}")
        continue

    sample_image = images[beanIndex[0]]  
    rgbBean = sample_image[..., :3]  
    hsvBean = sample_image[..., 3:] 

    print(f"Class: {beanClass}")
    print(f"Sample Image Shape (Height x Width x Channels): {sample_image.shape}")
    print(f"RGB Points: {rgbBean.shape} | HSV Points: {hsvBean.shape}")

    r, g, b = rgbBean.reshape(-1, 3).T
    h, s, v = hsvBean.reshape(-1, 3).T
    print(f"Number of RGB Points: {len(r)} (R: {r[:5]}, G: {g[:5]}, B: {b[:5]})")
    print(f"Number of HSV Points: {len(h)} (H: {h[:5]}, S: {s[:5]}, V: {v[:5]})")

    rgb_data.extend(
        {"Red": r[i], "Green": g[i], "Blue": b[i], "Class": beanClass}
        for i in range(len(r))
    )

    hsv_data.extend(
        {"Hue": h[i], "Saturation": s[i], "Value": v[i], "Class": beanClass}
        for i in range(len(h))
    )


df_rgb = pd.DataFrame(rgb_data)
df_hsv = pd.DataFrame(hsv_data)

fig_rgb = px.scatter_3d(
    df_rgb,
    x="Red",
    y="Green",
    z="Blue",
    color="Class",
    title="RGB Color Space",
    opacity=0.7,
    color_discrete_map=class_colors,
)

fig_rgb.update_traces(marker=dict(size=0.1), selector=dict(mode="markers"))

for trace in fig_rgb.data:
    trace.update(
        marker=dict(size=4),  
        showlegend=True,  
    )
fig_rgb.update_layout(
    width=900,  
    height=700,  
    margin=dict(l=10, r=10, t=40, b=10),  
    scene=dict(
        xaxis=dict(title="Red"),
        yaxis=dict(title="Green"),
        zaxis=dict(title="Blue"),
        aspectmode="cube",  
    ),
    legend=dict(
        font=dict(size=18),  
        itemsizing="constant",  
        title=dict(font=dict(size=20)), 
    ),
)
fig_rgb.show()

fig_hsv = px.scatter_3d(
    df_hsv,
    x="Hue",
    y="Saturation",
    z="Value",
    color="Class",
    title="HSV Color Space",
    opacity=0.7,
    color_discrete_map=class_colors,
)

fig_hsv.update_traces(marker=dict(size=0.5), selector=dict(mode="markers"))

for trace in fig_hsv.data:
    trace.update(
        marker=dict(size=2),  
        showlegend=True, 
    )
fig_hsv.update_layout(
    width=900, 
    height=700, 
    margin=dict(l=10, r=10, t=40, b=10), 
    scene=dict(
        xaxis=dict(title="Hue"),
        yaxis=dict(title="Saturation"),
        zaxis=dict(title="Value"),
        aspectmode="cube",  
    ),
    legend=dict(
        font=dict(size=18),  
        itemsizing="constant",  
        title=dict(font=dict(size=20)), 
    ),
)
fig_hsv.show()

print("\nNama file gambar yang diambil dari tiap folder:")
for category, filename in sample_filenames.items():
    print(f"{category}: {filename}")

if len(images) == 0:
    raise ValueError("Tidak ada gambar pada folder dataset")

# ---- TRANSFORM IMAGES AND LABELS TO NUMPY ARRAY -----
images = np.array(images)
labels = np.array(labels)
print(images)

X_train, X_temp, y_train, y_temp = train_test_split(images, labels, test_size=0.4, stratify=labels, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

y_train = to_categorical(y_train, num_classes=3)
y_val = to_categorical(y_val, num_classes=3)
y_test = to_categorical(y_test, num_classes=3)

def plot_sampleImage(X_data, y_data, class_labels, titles):
    fig, ax = plt.subplots(len(class_labels), 5, figsize=(15, len(class_labels) * 5))
    fig.patch.set_facecolor('white')

    for row, class_label in enumerate(class_labels):
        class_indices = np.where(np.argmax(y_data, axis=1) == class_label)[0]
        images = X_data[class_indices[:5]]
        for col in range(5):
            ax[row, col].imshow(images[col][..., :3])
            ax[row, col].axis('off')
            if col == 0:  
                ax[row, col].set_ylabel(titles[row], fontsize=12, rotation=90, labelpad=10)
    
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.4, wspace=0.3)  
    plt.suptitle('Sample Images from Each Class', fontsize=16, y=1.02)
    plt.show(block=False)
    plt.savefig(r'D:\SKRIPSI\sampleImage.png', dpi=300, bbox_inches='tight')
#    plt.pause(3)
    plt.close()

plot_sampleImage(
    X_train, y_train,
    class_labels=[0, 1, 2],
    titles=['Light Class', 'Medium Class', 'Dark Class']
)

train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest',
    shear_range=0.1,
    )
val_test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow(X_train, y_train, batch_size=32, shuffle=True)
validation_generator = val_test_datagen.flow(X_val, y_val, batch_size=32, shuffle=False)
test_generator = val_test_datagen.flow(X_test, y_test, batch_size=32, shuffle=False)

model = Sequential()

# -- MODEL ARCHITECTURE --
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 6)))  
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.05))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(BatchNormalization()) 
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.05))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.05))

model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.05))

model.add(Conv2D(512, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.05))

# Flatten Layers
model.add(Flatten())

# Fully Connected 1
model.add(Dense(256, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

# Fully Connected 2
model.add(Dense(512, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

#Output Layer
model.add(Dense(3, activation='softmax')) 

# DISPLAY MODEL ARCHITECTURE
model.summary()

# COMPILE MODEL
model.compile(optimizer=Adam(learning_rate=0.00001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# ADD EARLY STOPPING WITH 10 PATIENCE
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history = model.fit(
    train_generator,
    epochs=100,
    validation_data=validation_generator,
    callbacks=[early_stopping]
)
# TEST DATA EVALUATION
test_loss, test_acc = model.evaluate(test_generator)
print(f"Akurasi pada data uji: {test_acc}")

y_pred = model.predict(test_generator)
y_pred_classes = np.argmax(y_pred, axis=1)  
y_true = np.argmax(y_test, axis=1)  

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

print(f"Akurasi data validasi: {val_acc[-1]:.4f}")

# DEFINE OUTPUT DIRECTORY
output_dir = r"/mnt/c/Users/trkb/Documents/162012233070/Hasil"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)  

# ---- SAVING PATH ----
cr_path = os.path.join(output_dir, "cr68.png")
cm_path = os.path.join(output_dir, "cm68.png")
graph_path = os.path.join(output_dir, "plot68.png")
log_path = os.path.join(output_dir, "log68.txt")
savemodel_path=r'C:\Users\trkb\Documents\162012233070\Saved Model\ModelCNN_41.h5'
# --------------------- #

# CONVERT CLASSIFICATION REPORT TO PANDAS DATAFRAME 
class_name = ['Light', 'Medium', 'Dark']
cr = classification_report(y_true, y_pred_classes, target_names=class_name)
print(cr)

y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

report = classification_report(y_true, y_pred_classes, target_names=class_name, output_dict=True)
report_df = pd.DataFrame(report).transpose()

class_accuracies = [cm[i, i] / cm[i].sum() for i in range(len(class_name))]
report_df["class_accuracy"] = [
    round(class_accuracies[i], 4) if i < len(class_name) else '' for i in range(len(report_df))
]
 
report_df = report_df.round(4)

try:
    fig, ax = plt.subplots(figsize=(10, len(report_df) * 0.6))  
    ax.axis('off')  
    table = ax.table(cellText=report_df.values, colLabels=report_df.columns, rowLabels=report_df.index, loc='center')
    table.auto_set_font_size(False) 
    table.set_fontsize(10)  
    plt.savefig(cr_path, dpi=300, bbox_inches='tight')  
    plt.show()
    print(f"Classification Report disimpan pada : {cr_path}")
except Exception as e:
    print(f"Error save classification report: {e}")

epochs_range = range(1, len(acc) + 1)

# INISIALISASI DAN VISUALISASI CONFUSION MATRIX
cm = confusion_matrix(y_true, y_pred_classes)
cm_normalized = cm / cm.sum(axis=1, keepdims=True) * 100

# Plot confusion matrix as heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap='Blues',
            xticklabels=['Light', 'Medium', 'Dark'], yticklabels=['Light', 'Medium', 'Dark'])
plt.xlabel('Prediksi')
plt.ylabel('Asli')
plt.title('Confusion Matrix')

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        percentage = f"{cm_normalized[i, j]:.2f}%"
        count = f"({cm[i, j]})"
        text_color = "white" if cm_normalized[i, j] > 50 else "black"  
        plt.text(j + 0.5, i + 0.4, percentage, ha='center', va='center', color=text_color, fontsize=10)
        plt.text(j + 0.5, i + 0.6, count, ha='center', va='center', color=text_color, fontsize=10)

plt.show()

try:
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix disimpan pada : {cm_path}")
except Exception as e:
    print("Error Save Confusion Matrix")

# INISIALISASI PLOT AKURASI 
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

# INISIALISASI PLOT LOSS
plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')

# SAVE ACCURACY AND LOSS GRAPH
try:
    plt.savefig(graph_path, dpi=300, bbox_inches='tight')
    print(f"Plot akurasi dan loss disimpan pada : {graph_path}")
except Exception as e:
    print("Error Save Graph Akurasi dan Loss")

plt.show()
plt.pause(3)
plt.close()

# PRINT MODEL CONFIG
print("\nFull Model Configuration:")
model_config = model.get_config()  
print(model_config)

# PRINT LAYER CONFIG 
print("\nLayer Config:")
for layer in model.layers:
    print(f"\nLayer: {layer.name}")
    print("Config:", layer.get_config())  
print("\nOptimizer Configuration:")
optimizer_config = model.optimizer.get_config() 
print(optimizer_config)

# WRITE ITERATION LOG
try:
    with open(log_path, 'w') as log_file:
        log_file.write("=== Training Iterations ===\n")
        log_file.write("Epoch | Training Loss | Training Accuracy | Validation Loss | Validation Accuracy\n")
        log_file.write("-" * 65 + "\n")
        
        for epoch in range(len(acc)):
            log_file.write(f"{epoch+1:5d} | {loss[epoch]:14.6f} | {acc[epoch]:17.6f} | "
                           f"{val_loss[epoch]:15.6f} | {val_acc[epoch]:19.6f}\n")
        
        log_file.write("\n")
        log_file.write(f"Total Epochs Trained: {len(acc)}\n")
        log_file.write("\n=== Konfigurasi Model ===\n")
        log_file.write(str(model_config))
        
        log_file.write("\n=== Konfigurasi Optimizer ===\n")
        log_file.write(str(optimizer_config))
        
        log_file.write("\n=== Hasil Pengujian ===\n")
        log_file.write(f"Test Loss: {test_loss:.6f}\n")
        log_file.write(f"Test Accuracy: {test_acc:.6f}\n")

    print(f"Log telah disimpan pada : {log_path}")
except Exception as e:
    print(f"Error save log: {e}")

y_true_classes = np.argmax(y_test, axis=1)

# TAMPILKAN FP DAN FN DAN SIMPAN
def showCMdata(indices_FN, indices_FP, X_data, class_name, output_dir, main_title):
    max_images = 8  
    cols = 4  
    rows = 2 

    # Create figure for FN and FP side-by-side subplots
    fig, axes = plt.subplots(1, 2, figsize=(cols * 3, rows * 3), gridspec_kw={'wspace': 0.8})  
    fig.suptitle(main_title, fontsize=16)

    # Subplot 1: False Negatives (FN)
    axes[0].set_title("False Negatives (FN)", fontsize=12)
    axes[0].axis('off') 
    for i, idx in enumerate(indices_FN[:max_images]):  
        row, col = divmod(i, cols)
        sub_ax = fig.add_axes([0.05 + col * 0.1, 0.55 - row * 0.25, 0.15, 0.2])
        sub_ax.imshow(X_data[idx][..., :3])  
        sub_ax.axis('off')  

    if not indices_FN:  
        axes[0].text(0.5, 0.5, "No FN", fontsize=12, ha='center', va='center')

    axes[1].set_title("False Positives (FP)", fontsize=12)
    axes[1].axis('off') 
    for i, idx in enumerate(indices_FP[:max_images]): 
        row, col = divmod(i, cols)
        sub_ax = fig.add_axes([0.55 + col * 0.1, 0.55 - row * 0.25, 0.15, 0.2])
        sub_ax.imshow(X_data[idx][..., :3])  
        sub_ax.axis('off')  

    if not indices_FP:  
        axes[1].text(0.5, 0.5, "No FP", fontsize=12, ha='center', va='center')

    fig.canvas.draw()  
    divider_x = (axes[0].get_position().x1 + axes[1].get_position().x0) / 2 
    fig.add_artist(plt.Line2D([divider_x, divider_x], [0.1, 0.9], transform=fig.transFigure, color='black', linestyle='--', linewidth=2))


    plt.tight_layout(rect=[0, 0, 1, 0.95])  
    rawCM = os.path.join(output_dir, f"{class_name}_Raw68.png")
    plt.savefig(rawCM, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Show CM Data disimpan pada : {rawCM}")

# Function to show TN without saving
def show_TN(indices_TN, X_data, title):
    total_images = len(indices_TN)
    cols = 5  # Fixed number of columns
    rows = (total_images + cols - 1) // cols 
    
    plt.figure(figsize=(cols * 3, rows * 3)) 
    plt.suptitle(title, fontsize=16)
    
    for i, idx in enumerate(indices_TN[:rows * cols]): 
        plt.subplot(rows, cols, i + 1)
        plt.imshow(X_data[idx][..., :3])  
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

# Loop over each class to process FP, FN, and TN
for class_label, class_name in enumerate(["Light", "Medium", "Dark"]):
    print(f"\nClass: {class_name}")
    
    # Calculate indices for FN, FP, and TN
    dataFN = np.where((y_true_classes == class_label) & (y_pred_classes != class_label))[0].tolist()
    dataFP = np.where((y_true_classes != class_label) & (y_pred_classes == class_label))[0].tolist()
    dataTN = np.where((y_true_classes != class_label) & (y_pred_classes != class_label))[0].tolist()
    
    print(f"Total FN: {len(dataFN)}")
    print(f"Total FP: {len(dataFP)}")
    print(f"Total TN: {len(dataTN)}")
    
    # Show and save FN + FP in separate subplots within one main plot
    if len(dataFN) > 0 or len(dataFP) > 0:
        showCMdata(dataFN, dataFP, X_test, class_name, output_dir, f"{class_name} - False Negative & False Positive")
    
    # Show TN in one plot (not saved)
    if len(dataTN) > 0:
        show_TN(dataTN, X_test, f"{class_name} - True Negative")


# SAVE MODEL FILE
try:
    model.save(savemodel_path)
    print(f"Model disimpan pada :  {savemodel_path}")
except Exception as e:
    print(f"Error save model {e}")

