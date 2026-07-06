import cv2
 
# Match these to what you'll use in the hand-eye calibration script
SQUARES_X = 5
SQUARES_Y = 7
SQUARE_LENGTH = 0.030   # meters (30mm) -- adjust to your printable size
MARKER_LENGTH = 0.022   # meters (22mm), must be smaller than square length
 
ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_250)
BOARD = cv2.aruco.CharucoBoard(
    (SQUARES_X, SQUARES_Y), SQUARE_LENGTH, MARKER_LENGTH, ARUCO_DICT
)
 
# Render at high resolution for clean printing (300 DPI-ish)
PX_PER_METER = 4000  # tune so output image size is reasonable
img_size = (
    int(SQUARES_X * SQUARE_LENGTH * PX_PER_METER),
    int(SQUARES_Y * SQUARE_LENGTH * PX_PER_METER),
)
board_img = BOARD.generateImage(img_size, marginSize=40, borderBits=1)
 
cv2.imwrite("charuco_board.png", board_img)
print(f"Saved charuco_board.png at {img_size[0]}x{img_size[1]} px")
print(f"Print at actual size: each square = {SQUARE_LENGTH*1000:.0f}mm, "
      f"board = {SQUARES_X*SQUARE_LENGTH*1000:.0f}mm x {SQUARES_Y*SQUARE_LENGTH*1000:.0f}mm")
print("IMPORTANT: when printing, disable 'fit to page' / scaling -- print at 100% actual size,")
print("then measure a square with calipers to confirm SQUARE_LENGTH is accurate.")
