import numpy as np, cv2
image = cv2.imread("../images/minMax.jpg", cv2.IMREAD_GRAYSCALE)
if image is None: raise Exception("error reading image")

min_val, max_val, _, _ = cv2.minMaxLoc(image)

ratio = 255/ (max_val-min_val)
dst = np.round((image-min_val)*ratio).astype('uint8')
min_dst, max_dst, _,_ = cv2.minMaxLoc(dst)

print("original image min = %d, max = %d" %(min_val,max_val))
print("modified image min = %d, amx = %d" %(min_dst, max_dst))
cv2.imshow('image',image)
cv2.imshow('dst',dst)
cv2.waitKey()
cv2.destroyAllWindows()