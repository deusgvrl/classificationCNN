import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

def imageProcessing(imagePath):
    image = cv2.imread(imagePath)
    if image is None:
        print(f"Error membaca gambar pada : {imagePath}.")
        return None, None

    original = image.copy()
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(grayscale, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    # Plot histograms
    plt.figure(figsize=(10, 5))
    
    # Left: Original Image Histogram
    plt.subplot(1, 2, 1)
    plt.hist(grayscale.ravel(), bins=256, range=(0, 256), color='blue', alpha=0.7)
    plt.title("Histogram of Original Image")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")
    
    # Right: Gaussian Blurred Image Histogram
    plt.subplot(1, 2, 2)
    plt.hist(blurred.ravel(), bins=256, range=(0, 256), color='green', alpha=0.7)
    plt.title("Histogram of Gaussian Blurred Image")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")

    plt.tight_layout()
    plt.show()  

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(grayscale)

    for cnt in contours:
        if cv2.contourArea(cnt) > 500:
            cv2.drawContours(mask, [cnt], -1, (255), thickness=cv2.FILLED)

    return mask, original

def segmentAndExport(mask, original, savePath, expected_count=40):
    output_image = original.copy()
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)

    print(f"Ditemukan {num_labels - 1} objek awal.") 

    if not os.path.exists(savePath):
        os.makedirs(savePath)

    count = 0
    for i in range(1, num_labels): 
        x = stats[i, cv2.CC_STAT_LEFT]      
        y = stats[i, cv2.CC_STAT_TOP]       
        w = stats[i, cv2.CC_STAT_WIDTH]    
        h = stats[i, cv2.CC_STAT_HEIGHT]    
        area = stats[i, cv2.CC_STAT_AREA]   

        if area > 500:  
            cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

            bean = original[y:y+h, x:x+w] 
            bean_filename = os.path.join(savePath, f'bean_{count+1}.jpg')  
            cv2.imwrite(bean_filename, bean)

            count += 1

            print(f"Bounding box drawn for label {i}, bean saved as: {bean_filename}")

            if count >= expected_count:
                break
        # Show the images and wait for key press
    cv2.namedWindow('Bounding Boxes', cv2.WINDOW_NORMAL)
    cv2.imshow("Bounding Boxes", output_image)
    cv2.namedWindow('Mask', cv2.WINDOW_NORMAL)
    cv2.imshow('Mask', mask)

    print(f"Disimpan {count} biji kopi ke direktori '{savePath}'.")

def main():
    imagePath = r'/mnt/c/Users/trkb/Documents/162012233070/Dataset Kopi/Light/Light_9.jpg' 
    savePath = r'/mnt/c/Users/trkb/Documents/162012233070/Dataset Kopi/Preprocessed Light'

    mask, original = imageProcessing(imagePath)
    if mask is None:
        return

    segmentAndExport(mask, original, savePath, expected_count=40)
    
    cv2.namedWindow('Original Image', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Mask', cv2.WINDOW_NORMAL)
    cv2.imshow('Original Image', original)
    cv2.imshow('Mask', mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
