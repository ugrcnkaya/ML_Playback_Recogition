from PIL import Image
import numpy as np
import cv2

def compare_images(image_path1, image_path2):
    image1 = Image.open(image_path1)
    image2 = Image.open(image_path2)

    image1_array = np.array(image1)
    image2_array = np.array(image2)

    diff = cv2.absdiff(image1_array, image2_array)

    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

   
    #Any pixel with a value above 30 is set to 255 (white), and any pixel with a value below 30 is set to 0 (black).

    _, threshold = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)

    #image to a 1D array
    flat_threshold = threshold.flatten()
    total_difference = np.count_nonzero(flat_threshold)
    difference_rate = total_difference / flat_threshold.size

    if total_difference > 0:
        print(f"There is a difference. Difference rate: {difference_rate * 100:.2f}%")
    else:
        print("There is no difference.")

    # convert to PIL 
    diff_image = Image.fromarray(threshold)
    diff_image.save('diff.png')
    diff_image.show()



compare_images('image1.png', 'image0.png')