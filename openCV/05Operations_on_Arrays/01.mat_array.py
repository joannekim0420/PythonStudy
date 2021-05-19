import cv2

image = cv2.imread("../images/Lenna.png", cv2.IMREAD_COLOR)
if image is None: raise Exception("error reading image")

x_axis = cv2.flip(image, 0)
y_axis = cv2.flip(image, 1)
xy_axis = cv2.flip(image, -1)
rep_image = cv2.repeat(image, 1, 2)
trans_image = cv2.transpose(image)

title = ['image','x_axis','y_axis','xy_axis','rep_image','trans_image']
for t in title:
    cv2.imshow(t, eval(t))
cv2.waitKey()
cv2.destroyAllWindows()