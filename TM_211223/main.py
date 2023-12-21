import cv2
import numpy as np
import os

def match_template(template_img_path, icon_img_path, threshold=0.8):
    template_img = cv2.imread(template_img_path)
    icon_img = cv2.imread(icon_img_path)

    result = cv2.matchTemplate(template_img, icon_img, cv2.TM_CCOEFF_NORMED)
    _, _, _, max_loc = cv2.minMaxLoc(result)

    yloc, xloc = np.where(result >= threshold)
    w = icon_img.shape[1]
    h = icon_img.shape[0]

    max_loc = (max_loc[0] + w, max_loc[1] + h)
    cv2.rectangle(template_img, max_loc, (max_loc[0] - w, max_loc[1] - h), (0, 255, 0), 2)
    center_x = (max_loc[0] + max_loc[0] - w) // 2
    center_y = (max_loc[1] + max_loc[1] - h) // 2
    cv2.circle(template_img, (center_x, center_y), 5, (0, 0, 255), -1)
    title = f"Template: {os.path.basename(template_img_path)}, Icon: {os.path.basename(icon_img_path)}"

    cv2.imshow(title, template_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    
    return max_loc[0], max_loc[1], template_img, center_x, center_y

template_files = [
    "player1.png", "player2.png", "player3.png",
    "player4.png", "player.png"
]

icon_files = [
    "pause_icon.png", "fullscreen_icon.png",
    "next_video_icon.png", "previous_video.png"
]

for template_file in template_files:
    for icon_file in icon_files:
        template_path = os.path.join("", template_file)
        icon_path = os.path.join("", icon_file)

        x, y, result_img, center_x, center_y = match_template(template_path, icon_path)
        print(f"Max Matching Location (x, y) for {os.path.basename(icon_path)} on {os.path.basename(template_path)}:", x, y)
        print(f"Center Location (x, y):", center_x, center_y)

        result_file = f"result_{os.path.basename(template_path)}_{os.path.basename(icon_path)}.png"
        cv2.imwrite(result_file, result_img)