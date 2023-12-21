import cv2
import numpy as np

player_img=cv2.imread("player4.png")
icon_img=cv2.imread("previous_video.png")

result = cv2.matchTemplate(player_img, icon_img, cv2.TM_CCOEFF_NORMED)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)


threshold = 0.8
yloc, xloc = np.where(result >= threshold)
w = icon_img.shape[1]
h = icon_img.shape[0]

max_loc = (max_loc[0] + w, max_loc[1] + h)
cv2.rectangle(player_img, max_loc, (max_loc[0] - w, max_loc[1] - h), (0, 255, 0), 2)

cv2.imshow("player", player_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
print("Max Matching Location (x, y):", max_loc[0], max_loc[1])






