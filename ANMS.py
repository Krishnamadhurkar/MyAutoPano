
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm


def ANMS (image, N_best):
    Gray_image = cv.cvtColor(image,cv.COLOR_BGR2GRAY)

    # find corners using shi-tomasi
    corners = cv.goodFeaturesToTrack(Gray_image,5000,0.01,10)
    # get rid of unwanted dimensions
    corners = np.squeeze(corners)
    total_corners = corners.shape[0]
    Corner_score = cv.cornerHarris(Gray_image,8,3,0.1)
    r = np.inf*np.ones((total_corners,3))
    ED = 0  # Euclidean distance
    # we have (x,y) of all the corners in variable corners x=j and y =i
    for i in tqdm(range(total_corners)):
        xi,yi = int(corners[i][0]),int(corners[i][1])
        for j in range(total_corners):
            xj,yj = int(corners[j][0]),int(corners[j][1])
            if  Corner_score[yj][xj] > Corner_score[yi][xi]:
                ED = (xj-xi)**2+(yj-yi)**2
            if ED<r[i][2]:
            # find points that are evenly spread in the image
                r[i][0] = xi
                r[i][1] = yi
                r[i][2] = ED  # r = [xi,yi,Largest euclidean distance (point farther from original point)]
    # Sort points in descending order of ED
    r = r[r[:,2].argsort()] # ascending order
    corner_best  = np.flip(r,axis=0)  # descending order

    k = np.ones((N_best,2))
    count = 0
    i = 0
    while (count<N_best):
        if (corner_best[i][0] > 21 and corner_best[i][0]< image.shape[1]-21 and corner_best[i][1]>21 and corner_best[-i][1]< image.shape[0]-21) :
            # remove corners at the corners of the image (less than 20 or 20+image.shape) to match to the dim of patch (40x40)
            k[count][0] = corner_best[i][0]
            k[count][1] = corner_best[i][1]
            count+=1
        i+=1

    features = k   # feature point locations
    for i in range(N_best):
        image = cv.circle(image,(int(k[i][0]),int(k[i][1])),1,(0,0,255),2)
    
    Image = cv.resize(image,(1080,630))
    cv.imshow('image',Image)
    cv.waitKey(0)
    cv.destroyAllWindows
    return features


def featurevector(image,corners):
    image = cv.cvtColor(image,cv.COLOR_BGR2GRAY)

    # Apply Gaussian blur for all the corner points
    Num_corners = corners.shape[0]
    std_featurevector = {}  # dictionary collect vector corresponding to each corners
    
    for i in tqdm(range(Num_corners)):
        row = int(corners[i][1])
        col = int(corners[i][0])
        #40X40 Patch around the center of corner
        patch_i = image[row-20:row+20,col-20:col+20]
        patch = cv.GaussianBlur(patch_i,(5,5),cv.BORDER_DEFAULT) # Kernel size is 10x10
        res_patch = cv.resize(patch,(8,8))
        vector = np.resize(res_patch,(64,1))  # feature vector 64x1
        # Standardization of the vector: subtract the vector by its mean, and then divide the result by the vector's standard deviation.
        std_featurevector[(col,row)] = (vector-np.mean(vector))/np.std(vector)
   
    return std_featurevector



def FeatureMatching(features1,features2,threshold):
    correspondence_list = [] # tuple of keypoints of image 1 and 2 that are a match
    # get the coordinate values and norm value of the vector if they are under a threshold
    for i in features1:
        norm_list = []
        for j in features2:
            norm_list.append((j, np.linalg.norm(features1[i]-features2[j])))
        norm_list.sort(key = lambda X:X[1])
        if ((norm_list[0][1])/norm_list[1][1])<threshold:
            correspondence_list.append([i,norm_list[0][0]])
        #    # edge case : On next step check if the list has more than 4 matches for random sampling during RANSAC
    # correspondence_list = False
    return correspondence_list


def Drawmatches(image1, image2, loc1,loc2):
    # color image
    dim = (max(image1.shape[0], image2.shape[0]), image1.shape[1]+image2.shape[1], image1.shape[2]) # dimensions of new image that has bothe im1 and im2 joined horizontally
    match_image = np.zeros(dim,type(image1.flat[0]))
    # join the images horizontally
    match_image[0:image1.shape[0],0:image1.shape[1]] = image1
    match_image[0:image2.shape[0],image1.shape[1]:image1.shape[1]+image2.shape[1]] = image2
    # and draw lines between the matching pairs (x1,y1)  to (x2,y2)
    for i in  range(len(loc1)):
        x1 = loc1[i][0]
        y1 = loc1[i][1]
        x2 = loc2[i][0] + int(image1.shape[1]) # horizontal shift
        y2 = loc2[i][1]
        cv.line(match_image,(x1,y1),(x2,y2),(0,255,255),1)
    cv.imshow('image',match_image)
    cv.waitKey(0)
    cv.destroyAllWindows
    return None

def Homography(keypoints1,keypoints2, threshold):
    pass



Image1 = cv.imread('1.jpg') 
Image2 = cv.imread('2.jpg')
x1 = ANMS(Image1,500)
x2 = ANMS(Image2,500)
f1 = featurevector(Image1,x1)  # key: feature point location and values: 64x1 std vector
f2 = featurevector(Image2,x2)  # features of image 2
z = FeatureMatching(f1,f2,threshold =0.5)
Loc1 = []
Loc2 = []
for i in z:
    Loc1.append(i[0])  # Location of keypoints in image 1
    Loc2.append(i[1])
Image1 = cv.imread('1.jpg') 
Image2 = cv.imread('2.jpg')
k =  Drawmatches(Image1,Image2,Loc1,Loc2)

