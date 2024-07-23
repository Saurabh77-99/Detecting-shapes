import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('someshapes.jpg')
plt.imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGBA))
plt.title('Identify shapes');
plt.show()

gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
ret , thresh = cv2.threshold(gray,127,255,1)
contours , hierarchy = cv2.findContours(thresh.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
print('contours',contours)
print('hierarchy',hierarchy)

original_image = image.copy()

for cnt in contours:
    
    approx = cv2.approxPolyDP(cnt, 0.01*cv2.arcLength(cnt,True), True)
    
    M = cv2.moments(cnt)
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    
    if len(approx) == 3:
        shape_name = "Triangle"
        cv2.drawContours(original_image,[cnt],0,(0,255,0),-1)

        cv2.putText(original_image, shape_name, (cx-50, cy), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1)
    
    elif len(approx) == 4:
        x,y,w,h = cv2.boundingRect(cnt)
        
        if abs(w-h) <= 3:
            shape_name = "Square"
            
            cv2.drawContours(original_image, [cnt], 0, (0, 125 ,255), -1)
            cv2.putText(original_image, shape_name, (cx-50, cy), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1)
        else:
            shape_name = "Rectangle"
            
            cv2.drawContours(original_image, [cnt], 0, (0, 0, 255), -1)
            cv2.putText(original_image, shape_name, (cx-50, cy), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1)
            
    elif len(approx) == 10:
        shape_name = "Star"
        cv2.drawContours(original_image, [cnt], 0, (255, 255, 0), -1)

        cv2.putText(original_image, shape_name, (cx-50, cy), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1)
        
        
        
    elif len(approx) >= 15:
        shape_name = "Circle"
        cv2.drawContours(original_image, [cnt], 0, (0, 255, 255), -1)
        
        cv2.putText(original_image, shape_name, (cx-50, cy), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1)

    plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    plt.title('Identifying Shapes'); plt.show()
    