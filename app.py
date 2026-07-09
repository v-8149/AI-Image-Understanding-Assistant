import cv2
from tkinter import Tk, filedialog
import os

# Hide tkinter window
root = Tk()
root.withdraw()

# Open file selection window
image_path = filedialog.askopenfilename(
    title="Select an Image",
    filetypes=[
        ("Image Files", "*.jpg *.jpeg *.png")
    ]
)

if image_path:

    # Read image using OpenCV
    image = cv2.imread(image_path)

    if image is None:
        print("Cannot read image")
    else:
        # Image dimensions
        height, width = image.shape[:2]

        print("Image uploaded successfully")
        print("--------------------------")
        print("File name:", os.path.basename(image_path))
        print("Width:", width)
        print("Height:", height)

        # Display image
        cv2.imshow("Uploaded Image", image)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

else:
    print("No image selected")