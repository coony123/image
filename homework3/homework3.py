import cv2
import numpy as np
import skimage
from skimage.transform import rotate
from skimage.feature import local_binary_pattern
from skimage import data, io,data_dir,filters, feature
from skimage.color import label2rgb
#-------------------------------------
def Variance_compare(LBP_1,LBP_2,cluster_num):
    lbp_cluster_piece = (np.power(2,n_points)-1) / cluster_num
    LBP_Hist_1 = np.zeros((cluster_num),dtype=float)
    LBP_Hist_2 = np.zeros((cluster_num),dtype=float)

    for i in range(cluster_num):
        LBP_Hist_1[i] = len(LBP_1[(LBP_1 >= i * lbp_cluster_piece) & (LBP_1 < (i+1) * lbp_cluster_piece)])
        LBP_Hist_2[i] = len(LBP_2[(LBP_2 >= i * lbp_cluster_piece) & (LBP_2 < (i+1) * lbp_cluster_piece)])
    
    LBP_Hist_1 /= len(LBP_1)
    LBP_Hist_2 /= len(LBP_2)

    D0 = LBP_Hist_1 - LBP_Hist_2
    D0 = np.power(D0,2)
    D0 = np.sqrt(np.sum(D0))
    return D0
#-------------------------------------
# height = gray.shape[0]
# width = gray.shape[1]
# hist=[0]*256
# for x in range(0, width):
#     for y in range(0, height):     
#         hist[int(gray[y,x])]+=1
# hist[0]=0
# for x in range(0, 256):
#     img[255-int(hist[x]*256/(max(hist)+0.00001)):255,x,:]=255
#-------------------------------------
width  = 1600
height = 1200
img  = cv2.imread("test.jpg")
img_drow  = np.copy(img)

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# ret,thresh_img = cv2.threshold(gray,128,255,cv2.THRESH_BINARY)
#-------------------------------------
radius = 1 # LBP算法中范围半径的取值
n_points = 8*radius # 领域像素点
lbp = local_binary_pattern(gray, n_points, radius)
#-------------------------------------
for S in range (0,1,1):
    kernel_size = 33+(S*2) #15,17
    kernel_range = kernel_size // 2 #取商 向下取整 
    marker = np.zeros((height,width),np.uint8)

    for w in range((kernel_size*2),(width-kernel_size),kernel_range):
        for h in range((kernel_size*2),(height-kernel_size),kernel_range):
            
            Diff1=Variance_compare(lbp[h-kernel_range:h+kernel_range,w-kernel_range:w+kernel_range],lbp[h-kernel_range:h+kernel_range,w:w+(kernel_range*2)],32)
            #print("Diff1=",Diff1)
            Diff2=Variance_compare(lbp[h-kernel_range:h+kernel_range,w-kernel_range:w+kernel_range],lbp[h:h+(kernel_range*2),w-kernel_range:w+kernel_range],32)
            #print("Diff2=",Diff2)
            Diff3=Variance_compare(lbp[h-kernel_range:h+kernel_range,w-kernel_range:w+kernel_range],lbp[h-kernel_range:h+kernel_range,w-(kernel_range*2):w],32)
            #print("Diff3=",Diff3)
            Diff4=Variance_compare(lbp[h-kernel_range:h+kernel_range,w-kernel_range:w+kernel_range],lbp[h-(kernel_range*2):h,w-kernel_range:w+kernel_range],32)
            #print("Diff4=",Diff4)
            
            # print("Diff_total = ",(Diff1+Diff2+Diff3+Diff4))
            if ((Diff1+Diff2+Diff3+Diff4)/4)<0.9:
                marker[h-kernel_range:h+kernel_range,w-kernel_range:w+kernel_range] = 255
#-------------------------------------
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11,11))
    marker = cv2.morphologyEx(marker, cv2.MORPH_CLOSE, kernel,iterations=2)
    sure_fg = np.uint8(marker)
    ret, markers = cv2.connectedComponents(sure_fg)
    markers = cv2.watershed(img,markers) 
    cut = np.zeros((height,width),np.uint8)
    cut[markers == -1] = [255]
#-------------------------------------
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    dilate = np.zeros((height,width),np.uint8)
    dilate = cv2.dilate(cut,kernel)
    dilate = 255 - dilate
#-------------------------------------
    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(dilate) #編號
    max_size = 0 
    road_number = -1
    roadbase = np.zeros((height,width),np.uint8)

    for i in range(0,nlabels-1,1):
        x,y = centroids[i]
        if  (y>(height/2))  and np.all(dilate[labels==i] == 255 ):
            if(dilate[labels==i].size > max_size):
                max_size =  dilate[labels==i].size 
                road_number = i
    print ("road_number : ",road_number) #第幾剛當標準
    roadbase[labels==road_number] = (255)
#-------------------------------------
    for i in range(0,nlabels,1):
            x,y = centroids[i]
            Diff=Variance_compare(lbp[ labels==road_number],lbp[ (labels==i) & (dilate==255) ],32)      
            cv2.putText(cut, str(Diff), (int(x),int(y)), cv2.FONT_HERSHEY_SIMPLEX,  0.5, (255, 255,255), 1, cv2.LINE_AA)
            if (y>(height/2-200))  and np.all(dilate[labels==i] == 255 ) :     
                if  ((0.1 < Diff) and (Diff < 0.12)) or ((0.124 < Diff) and (Diff < 0.126)) or ((0.0173 < Diff) and (Diff < 0.0175)) :
                    roadbase[ (labels==i)] = (255)

    roadbase = cv2.morphologyEx(roadbase, cv2.MORPH_CLOSE, kernel,iterations=2)
    roadbase = np.uint8(roadbase)
    cnt, hierarchy  = cv2.findContours(roadbase.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for c in cnt:
        cv2.drawContours( img_drow, [c], -1, ( 160, 102, 211), -1) 
#-------------------------------------
cv2.namedWindow("img", cv2.WINDOW_NORMAL)
cv2.imshow("img",img)

cv2.namedWindow("lbp", cv2.WINDOW_NORMAL)
cv2.imshow("lbp",lbp)

cv2.namedWindow("cut", cv2.WINDOW_NORMAL)
cv2.imshow("cut",cut)

cv2.namedWindow("dilate", cv2.WINDOW_NORMAL)
cv2.imshow("dilate",dilate)

cv2.namedWindow("roadbase", cv2.WINDOW_NORMAL)
cv2.imshow("roadbase",roadbase)

cv2.namedWindow("img_drow", cv2.WINDOW_NORMAL)
cv2.imshow("img_drow",img_drow)
#-------------------------------------
cv2.imwrite('img.jpg',img)
cv2.imwrite('lbp.jpg',lbp)
cv2.imwrite('cut.jpg',cut)
cv2.imwrite('dilate.jpg',dilate)
cv2.imwrite('roadbase.jpg',roadbase)
cv2.imwrite('img_drow.jpg',img_drow)
#-------------------------------------
cv2.waitKey(0)
cv2.destroyAllWindows()
