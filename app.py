import cv2
import os

from objDetection import detect_objects

image_path = "/home/vaishnavishivade/Project/sample.jpg"

if os.path.exists(image_path):

    image = cv2.imread(image_path)

    if image is None:
        print("Cannot read image")

    else:
        print("Image uploaded successfully")

        detected_image = detect_objects(image)

        output_path = "detected_object.jpg"

        cv2.imwrite(output_path, detected_image)

        print("\nDetection completed!")
        print(f"Output image saved as: {output_path}")

else:
    print("Image not found.")
