import cv2
import imutils
import numpy as np
from imutils import contours


roi_list = []


# simple preprocessing of image using different filters
def preprocessing(inp_image):
        gaussian_filter = cv2.GaussianBlur(inp_image,(5,5),0)
        grayscale_image = cv2.cvtColor(gaussian_filter,cv2.COLOR_BGR2GRAY)
        _,threshold_image = cv2.threshold(grayscale_image,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        return threshold_image

# red masking for private vehicles
# setting range of red color in hsv format..
def red_mask(inp_image):
    lower_red1 = np.array([0,50,50])
    upper_red1 = np.array([10,255,255])
    mask1 = cv2.inRange(inp_image,lower_red1,upper_red1)

    lower_red2 = np.array([170,50,50])
    upper_red2 = np.array([180,255,255])
    mask2 = cv2.inRange(inp_image,lower_red2,upper_red2)

    mask = mask1 | mask2
    out_image = cv2.bitwise_and(inp_image,inp_image,mask=mask)
    return out_image

def find_contours(inp_img):
    contours,hierarchy = cv2.findContours(inp_img,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    return contours

#Image parameters approximation / finding
def image_parameters(eps,contours):
     epsilon = eps*cv2.arcLength(contours,True)
     approx = cv2.approxPolyDP(contours,epsilon,True)
     try:
         if len(approx)==4:
                x,y,w,h = cv2.boundingRect(contours)
                return x,y,w,h
     except Exception as e:
         pass


# locaization of license plate from input image
def localization(inp_image):
    final_img = inp_image.copy()
    hsv_image = cv2.cvtColor(inp_image,cv2.COLOR_BGR2HSV)
    masked_image = red_mask(hsv_image)
    #cv2.imshow("masked",masked_image)
    preprocess_image = preprocessing(masked_image)
    #cv2.imshow("preprocess_img",preprocess_image)
    gradient_filter = cv2.morphologyEx(preprocess_image,cv2.MORPH_GRADIENT,cv2.getStructuringElement(cv2.MORPH_RECT,(5,5)))
    #cv2.imshow("gradient",gradient_filter)
    cannyedge_image = cv2.Canny(gradient_filter,100,300)
    #cv2.imshow("canny",cannyedge_image)
    av_test= average_area_filter(cannyedge_image)
    contours = find_contours(av_test)

    for cnts in contours:
        param = image_parameters(0.04,cnts)
        if param != None:
            x = param[0]
            y = param[1]
            w = param[2]
            h = param[3]
            print(param)
            ar = w/h
            aspect = 4.7272
            minArea = 15 * aspect * 15
            maxArea = 125 * aspect * 125
            area = w * h

            if ar < 1:
                ar = 1 / ar

            if ar >= 0.95 and ar <1.05:
                print("square")

            if (ar >= 1.33 and ar <= 2.0) and (area >= minArea and area <= maxArea):
                cv2.rectangle(inp_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # selecting region of interest of th image
                roiImage = final_img[y:y+h,x:x+w]
                cv2.imshow("localized plate",inp_image)
                return roiImage



# Cleans noise from the border areas
def clean_border(inp_img,radius):
    contours_list = []
    img_rows = inp_img.shape[0]
    img_cols = inp_img.shape[1]
    print(img_rows)

    contours = find_contours(inp_img)

    for index in np.arange(len(contours)):
        cnts = contours[index]
        for points in cnts:
            row_cnts = points[0][1]
            col_cnts = points[0][0]
            check_row = (row_cnts >= 0 and row_cnts < radius) or (row_cnts >= img_rows-1-radius and row_cnts < img_rows)
            check_cols = (col_cnts >= 0 and col_cnts < radius) or (col_cnts >= img_cols-1-radius and col_cnts < img_cols)
            if check_row or check_cols:
                contours_list.append(index)
    #print(contours_list)
    for cnt_index in contours_list:
        cv2.drawContours(inp_img,contours,cnt_index,(0,0,0),-1)
    return inp_img

def average_area_filter(inp_image):
    area_list = []
    contours_list = []
    contours = find_contours(inp_image)

    for cnts in contours:
        area = cv2.contourArea(cnts)
        area_list.append(area)
    average_area = np.average(area_list)
    print("Avg:",average_area)
    print("area_list",area_list)

    for cnts_idx in contours:
        area_temp = cv2.contourArea(cnts_idx)
        if area_temp < 0.5*average_area :
            contours_list.append(cnts_idx)
    cv2.drawContours(inp_image,contours_list,-1,(0,0,0),-1)

    return inp_image
# average width filter
def average_width(contours):
    w_list = []
    for cnts in contours:
       x,y,w,h = cv2.boundingRect(cnts)
       w_list.append(w)
    avg_width = np.average(w_list)
    return avg_width

# Erosion and Dilation of selected image
def erode_dilate(inp_image):
    out_image = cv2.morphologyEx(inp_image,cv2.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_RECT,(3,3)))
    return out_image

# Character Segmentation
def character_segmentation(inp_image,fin_image):
    cnts,hierarchy = cv2.findContours(inp_image,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    avg_width = average_width(cnts)
    print(avg_width)
    roi_list.clear()


    cnt_list = contours.sort_contours(cnts, "top-to-bottom")[0]
    sorted_list = []
    for (q, i) in enumerate(np.arange(0, len(cnt_list),4)):
        cnt = contours.sort_contours(cnt_list[i:i + 4], method="left-to-right")[0]
        sorted_list.extend(cnt)
        print(len(sorted_list))

    for e_cnts in sorted_list:
        x,y,w,h = cv2.boundingRect(e_cnts)
        if not w < 0.5*avg_width:
            cv2.rectangle(inp_image,(x,y),(x+w,y+h),(255,255,255),3)
            cv2.rectangle(fin_image,(x,y),(x+w,y+h),(0,255,0),3)
            cv2.imshow("segment",fin_image)
            roi_image = fin_image[y:y+h,x:x+w]
            roi_list.append(roi_image)




    return roi_list















def main():
    input = cv2.imread("test1.jpg")
    resize = imutils.resize(input,500,500,cv2.INTER_AREA)
    #print(resize.shape)
    localized_plate =localization(resize)
    preprocessed_img = preprocessing(localized_plate)
    cv2.imshow("preprocessed image",preprocessed_img)
    cleaned_border_img = clean_border(preprocessed_img,10)
    cv2.imshow("border cleaned image",cleaned_border_img)
    area_filter = average_area_filter(cleaned_border_img)
    cv2.imshow("area filtered image",area_filter)
    er_dil = erode_dilate(area_filter)
    cv2.imshow("erode dilate image",er_dil)
    character_segmentation(er_dil,localized_plate)

    k = cv2.waitKey(0)
    if k == 27:
        cv2.destroyAllWindows()




if __name__ == '__main__':
        main()
