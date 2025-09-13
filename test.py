import cv2
import numpy as np
import os

def find_bounding_box(image_path):
    """
    Finds the bounding box around the non-white areas of the image.
    """
    # Check if file exists
    if not os.path.exists(image_path):
        print(f"Error: File not found at {image_path}")
        return None

    # Load the image (try different methods to ensure success)
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    if image is None:
        print(f"Error: Unable to load image from {image_path}")
        return None

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Threshold to detect non-white regions (adjust threshold if needed)
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print("No non-white regions found.")
        return None

    # Get bounding box around the largest contour (or all)
    x, y, w, h = cv2.boundingRect(np.vstack(contours))

    return x, y, w, h

def draw_bounding_box(image_path, output_path):
    """
    Draws a bounding box around non-white regions of an image and saves the output.
    """
    # Load the image
    image = cv2.imread(image_path)

    if image is None:
        print(f"Error: Unable to open image from {image_path}")
        return

    # Find bounding box
    bounding_box = find_bounding_box(image_path)

    if bounding_box:
        x, y, w, h = bounding_box
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green rectangle

        # Save output image
        cv2.imwrite(output_path, image)
        print(f"Output image saved to: {output_path}")

        # Display image and wait for keypress
        cv2.imshow("Bounding Box", image)
        
        # Wait for user to press 'q' or 'Esc' to close
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # 27 is the ASCII code for the 'Esc' key
                break

        # Destroy all OpenCV windows
        cv2.destroyAllWindows()
    else:
        print("No bounding box detected.")

if __name__ == "__main__":
    # Set input and output file paths
    input_image = "TEST IMAGE.jpeg"  # Change this to your actual image path
    output_image = "output.png"

    # Ensure input path is absolute
    input_image = os.path.abspath(input_image)
    output_image = os.path.abspath(output_image)

    # Debugging: Print file paths
    print(f"Input Image Path: {input_image}")
    print(f"Output Image Path: {output_image}")

    # Draw the bounding box
    draw_bounding_box(input_image, output_image)
