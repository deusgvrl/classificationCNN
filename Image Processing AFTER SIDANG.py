import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

ROASTING_LEVELS = {
    "Light": {
        "inputFolder": "/mnt/c/Users/trkb/Documents/162012233070/Dataset Kopi/Light",
        "outputFolder": "/mnt/c/Users/trkb/Documents/162012233070/Dataset Kopi/Segmented Light",
        "cannyThresh": (50, 150),
    },
    "Medium": {
        "inputFolder": "/mnt/c/Users/trkb/Documents/162012233070/Dataset Kopi/Medium",
        "outputFolder": "/mnt/c/Users/trkb/Documents/162012233070/Dataset Kopi/Segmented Medium",
        "cannyThresh": (40, 175),
    },
    "Dark": {
        "inputFolder": "/mnt/c/Users/trkb/Documents/162012233070/Dataset Kopi/Dark",
        "outputFolder": "/mnt/c/Users/trkb/Documents/162012233070/Dataset Kopi/Segmented Dark",
        "cannyThresh": (30, 200),
    },
}

def imageprocessing(imgpath, cannyThresh):
    image = cv2.imread(imgpath)
    if image is None:
        print(f"Error membaca gambar pada : {imgpath}.")
        return None, None

    original = image.copy()
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(grayscale, (5, 5), 0)
    edges = cv2.Canny(blurred, cannyThresh[0], cannyThresh[1])

    plt.figure(figsize=(10, 5))
    
    # PLOT HISTOGRAM ORIGINAL
    plt.subplot(1, 2, 1)
    plt.hist(grayscale.ravel(), bins=256, range=(0, 256), color='blue', alpha=0.7)
    plt.title("Original", fontsize = 14)
    plt.xlabel("Intensitas Piksel", fontsize = 12)
    plt.ylabel("Frekuensi (Jumlah Piksel)", fontsize = 12)
    
    # PLOT HISTOGRAM GAUSSIAN BLUR
    plt.subplot(1, 2, 2)
    plt.hist(blurred.ravel(), bins=256, range=(0, 256), color='green', alpha=0.7)
    plt.title("Gaussian Blur", fontsize = 14)
    plt.xlabel("Intensitas Piksel", fontsize = 12)
    plt.ylabel("Frekuensi (Jumlah Piksel)", fontsize = 12)

    plt.tight_layout()
    plt.show()

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(grayscale)

    for cnt in contours:
        if cv2.contourArea(cnt) > 500:
            cv2.drawContours(mask, [cnt], -1, (255), thickness=cv2.FILLED)

    return mask, original, blurred

def thresholdValue(blurred, mask, stats):
    for i in range(1, stats.shape[0]):  
        x, y, w, h, area = stats[i]
        roi = blurred[y:y+h, x:x+w]
        maskedObject = mask[y:y+h, x:x+w]

        # Hanya memproses pixel hasil masking
        detectedPixel = roi[maskedObject > 0]  
        if detectedPixel.size > 0:
            lowerThresh = np.min(detectedPixel)
            upperThresh = np.max(detectedPixel)
            print(f"Biji {i}: x={x}, y={y}, w={w}, h={h}, Area={area}, "
                  f"Lower Threshold={lowerThresh}, Upper Threshold={upperThresh}")
        

def segment_images(mask, original, blurred, save_path, savedBeans, expectedBeans=400):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)

    print(f"Ditemukan {num_labels - 1} objek awal.")

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    thresholdValue(blurred, mask, stats)
    count = 0
    outputImage = original.copy()
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]

        if area > 500:
            cv2.rectangle(outputImage, (x, y), (x + w, y + h), (0, 255, 0), 2)
            bean = original[y:y+h, x:x+w]
            bean_name = os.path.join(save_path, f'bean_{savedBeans + count + 1}.jpg')
            cv2.imwrite(bean_name, bean)
            count += 1

            if savedBeans + count >= expectedBeans:
                break

    cv2.namedWindow('Bounding Box', cv2.WINDOW_NORMAL)
    cv2.imshow('Bounding Box', outputImage)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print(f"Disimpan {count} biji kopi ke : '{save_path}'.")
    return count

def main():
    for roastLevel, config in ROASTING_LEVELS.items():
        inputFolder = config["inputFolder"]
        outputFolder = config["outputFolder"]
        cannyThresh = config["cannyThresh"]

        savedBeans = 0

        print(f"Memproses Kategori: {roastLevel}")
        for i in range(1, 11): 
            if savedBeans >= 400:
                print(f"Tercapai 400 biji untuk kelas {roastLevel}.")
                break

            savename = f"{roastLevel}_{i}.jpg"
            imgpath = os.path.join(inputFolder, savename)  
            imgpath = os.path.normpath(imgpath)

            print(f"Cek file: {imgpath}")
            if not os.path.exists(imgpath):
                print(f"File tidak ada: {imgpath}")
                continue

            mask, original, blurred = imageprocessing(imgpath, cannyThresh)
            if mask is None:
                continue

            beans_saved = segment_images(mask, original, blurred, outputFolder, savedBeans, expectedBeans=400)
            savedBeans += beans_saved

        print(f"Selesai memproses {roastLevel} dengan {savedBeans} biji terdeteksi.")

if __name__ == "__main__":
    main()
