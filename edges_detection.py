import cv2
 
# Read the original image
img = cv2.imread('./input_data/egyptian_mau.png') 
# Display original image
cv2.imshow('Original', img)
cv2.waitKey(0)
 

# Convert to graycsale
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('converted to gray', img_gray)
cv2.waitKey(0)

# Apply thresholding to create a binary image
threshold_value = 128
_, binary_image = cv2.threshold(img_gray, threshold_value, 255, cv2.THRESH_BINARY)
cv2.imshow('converted to full black full white', binary_image)
cv2.waitKey(0)

# Blur the image for better edge detection
img_blur = cv2.GaussianBlur(binary_image, (3,3), 0) #(21,21) is the level of blurry, always use odd numbers, the higher more blurr, the lower less blur
cv2.imshow('blurred image', img_blur)
cv2.waitKey(0)
 
# Canny Edge Detection
edges = cv2.Canny(image=img_blur, threshold1=100, threshold2=200) # Canny Edge Detection
# Display Canny Edge Detection Image
cv2.imshow('Canny Edge Detection', edges)
cv2.imwrite("extracted.png", edges)
cv2.waitKey(0)
 
cv2.destroyAllWindows()

