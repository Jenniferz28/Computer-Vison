import cv2
import numpy as np

# we define a metric to idenfity star
def ratio_error(contour, error_percent):
    area = cv2.contourArea(contour)
    length = cv2.arcLength(contour,True)
    ratio = np.sqrt(area/length)
    tolerance = ratio * error_percent # uncertainty of ratio
    return ratio, tolerance

def im_plot(title,data,wait):
    cv2.imshow(title, data)
    cv2.waitKey(wait)# waiting time (ms) to terminate the plot
    cv2.destroyAllWindows()

#find the mask for special stars like RED and BLACK
def find_color_mask(image,boundaries):
    lower = np.array(boundaries[0],dtype = "uint8")
    upper = np.array(boundaries[1],dtype = "uint8")
    target_mask = cv2.inRange(image, lower, upper)
    return target_mask


def main():
    mask = cv2.imread('mask.png',0)
    #finding contours = finding white object from black background
    _,thresh_mask = cv2.threshold(mask,127,255,0)
    _,contours,_ = cv2.findContours(thresh_mask,1,2)
    cnt = contours[1] #the contour of mask(star)
    error_percent = 0.1 # 10% uncertainty of the mask ratio
    mask_ratio, tolerance = ratio_error(cnt,error_percent)

    ### Look at contours of regular stars firstly---------
    img = cv2.imread('us_flag_color.png')
    img_hsv=cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    img_new = cv2.imread('us_flag_color.png',0)
    _,thresh0 = cv2.threshold(img_new,127,255,cv2.THRESH_BINARY)
    _,contours0,_ = cv2.findContours(thresh0,1,2)

    img_grey = cv2.cvtColor(img_new,cv2.COLOR_GRAY2RGB)
    for cnt in contours0:
        ratio, tolerance = ratio_error(cnt,error_percent)
        if ratio > mask_ratio - tolerance and ratio < mask_ratio + tolerance:
            cv2.drawContours(img_grey, [cnt], 0, (0,0,255), 2)
    print("Contours of Regular Stars --- Opps,the red and black stars are missing!")
    im_plot("Contours of Regular Stars --- Opps,the red and black stars are missing!", img_grey, 2000)


    ### Map the mask for the RED star and BLACK stars-----

    # red region in HSV
    boundaries_red = ([170,50,50],[180,255,255])
    mask_red = find_color_mask(img_hsv, boundaries_red)

    #black region in HSV
    boundaries_black = ([0,0,0],[180, 255, 30])
    mask_black = find_color_mask(img_hsv, boundaries_black)

    print("Finding the Mask for the Red and the Black Star.")
    im_plot("Mask for the Red Star", mask_red, 2000)
    im_plot("Mask for the Black Star", mask_black,  2000)

    _,contours_red,_ = cv2.findContours(mask_red,1,2)
    _,contours_black,_ = cv2.findContours(mask_black,1,2)


    ###Find contours for ALL STARS--------------
    ret,thresh = cv2.threshold(img_new,127,255,cv2.THRESH_BINARY)
    derp,contours,hierarchy = cv2.findContours(thresh,1,2)
    contours = contours + contours_red+contours_black

    for cnt in contours:
        ratio, tolerance = ratio_error(cnt,error_percent)
        if ratio > mask_ratio - tolerance and ratio < mask_ratio + tolerance:
            cv2.drawContours(img_grey, [cnt], 0, (0,0,255), 2)

    im_plot("Contours of All Stars", img_grey, round(2e4))


if __name__=="__main__":
    main()
