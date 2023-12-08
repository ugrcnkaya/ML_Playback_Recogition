import cv2
import numpy as np


class Constants:
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([255, 40, 255])  # Increased the saturation upper limit


def clean_image(image_path):
    # Load the image
    image = cv2.imread(image_path)

    # Convert the image to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define a range for white color
    lower_white = Constants.lower_white
    upper_white = Constants.upper_white

    # Create a mask for the white color
    mask = cv2.inRange(hsv, lower_white, upper_white)

    # Bitwise-AND the original image and the mask
    result = cv2.bitwise_and(image, image, mask=mask)

    # Save the result image
    cv2.imwrite('result.png', result)

    # Load the cleaned image
    cleaned_image = cv2.imread('result.png')

    # Display the cleaned image
    cv2.imshow('Cleaned Image', cleaned_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


class ImageProcessor:
    def __init__(self, image_path, template_path):
        self.image_path = image_path
        self.template_path = template_path
        self.matched_coordinates = []  # To store matched coordinates

    def load_image(self, path):
        return cv2.imread(path)

    def convert_to_hsv(self, image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    def create_mask(self, image, lower_range, upper_range):
        hsv = self.convert_to_hsv(image)
        return cv2.inRange(hsv, lower_range, upper_range)

    def apply_mask(self, image, mask):
        return cv2.bitwise_and(image, image, mask=mask)

    def convert_to_gray(self, image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    def template_matching(self, player_image, template_image, threshold=0.6):
        player_gray = self.convert_to_gray(player_image)
        template_gray = self.convert_to_gray(template_image)

        result = cv2.matchTemplate(player_gray, template_gray, cv2.TM_CCOEFF_NORMED)
        loc = np.where(result >= threshold)

        # Reset matched coordinates
        self.matched_coordinates = []

        for pt in zip(*loc[::-1]):
            x, y = pt[0], pt[1]
            self.matched_coordinates.append({'x': x, 'y': y})

            # Draw rectangle
            cv2.rectangle(player_image, pt, (x + template_image.shape[1], y + template_image.shape[0]), (0, 255, 0), 2)

            # Calculate center of the rectangle
            center_x = x + (template_image.shape[1] // 2)
            center_y = y + (template_image.shape[0] // 2)

            # Draw red dot at the center
            cv2.circle(player_image, (center_x, center_y), 5, (0, 0, 255), -1)

        return player_image, self.matched_coordinates[:1]  # Return only the first match

    def display_image(self, image, window_name='Image'):
        cv2.imshow(window_name, image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def process_image(self):
        # Load the original image
        original_image = self.load_image(self.image_path)

        # Define a range for white color
        lower_white = Constants.lower_white
        upper_white = Constants.upper_white

        # Create a mask for the white color
        mask = self.create_mask(original_image, lower_white, upper_white)

        # Bitwise-AND the original image and the mask
        result_image = self.apply_mask(original_image, mask)

        # Load the template image
        template_image = self.load_image(self.template_path)

        # Apply template matching
        result_with_template, matched_coordinates = self.template_matching(result_image.copy(), template_image)

        # Display the result with rectangles and red dots
        self.display_image(result_with_template, window_name='Result with Template Matching')

        # Print and return matched coordinates
        print("Matched Coordinates:", matched_coordinates)
        return matched_coordinates


image_paths = ['player.png', 'player2.png', 'player3.png', 'player4.png']
template_paths = ['next_video_icon.png', 'fullscreen.png', 'play_icon.png', 'settings_icon.png']

for image_path in image_paths:
    for template_path in template_paths:
        print(f"Processing image: {image_path} with template: {template_path}")
        
        # Create an instance of ImageProcessor for each image and template pair
        image_processor = ImageProcessor(image_path, template_path)
        
        # Process the image
        coordinates = image_processor.process_image()

#clean_image('player.png')
