import os
from os import listdir
from os.path import isfile, join
import csv
import random
import cv2
import shutil
import matplotlib.pyplot as plt
import os.path
import sys
from os import path
import numpy as np
from scipy.spatial.distance import euclidean
from skimage.feature import local_binary_pattern
from sklearn.metrics import auc
import pandas as pd


def compAvgFaceTrue():
    csv_file = csv.reader(open('TruePairs.csv', "r"), delimiter=",")
    allDistancesLBP, allDistancesHOG = [], []

    for row in csv_file:
        if (path.exists("C:/Users/User/Downloads/biom3/" + row[0].split('.')[0]) and path.exists("C:/Users/User/Downloads/biom3/" + row[1].split('.')[0])):
            file1 = "C:/Users/User/Downloads/biom3/averageFaces/" + row[0].split('.')[0] + ".jpg"
            file2 = "C:/Users/User/Downloads/biom3/averageFaces/" + row[1].split('.')[0] + ".jpg"
            image1 = cv2.imread(file1)
            image2 = cv2.imread(file2)
            allDistancesLBP.append(compareTwoImagesWithLBP(lbp_features(image1), lbp_features(image2)))
            allDistancesHOG.append(compareTwoImagesWithHOG(getHOG(image1), getHOG(image2)))

    return fromDistancesToDivided(allDistancesLBP), fromDistancesToDivided(allDistancesHOG)

def compAvgFaceFalse():
    csv_file = csv.reader(open('FalsePairs.csv', "r"), delimiter=",")
    allDistancesLBP, allDistancesHOG = [], []

    for row in csv_file:
        if (path.exists("C:/Users/User/Downloads/biom3/" + row[0].split('.')[0]) and path.exists("C:/Users/User/Downloads/biom3/" + row[1].split('.')[0])):
            file1 = "C:/Users/User/Downloads/biom3/averageFaces/" + row[0].split('.')[0] + ".jpg"
            file2 = "C:/Users/User/Downloads/biom3/averageFaces/" + row[1].split('.')[0] + ".jpg"
            image1 = cv2.imread(file1)
            image2 = cv2.imread(file2)
            allDistancesLBP.append(compareTwoImagesWithLBP(lbp_features(image1), lbp_features(image2)))
            allDistancesHOG.append(compareTwoImagesWithHOG(getHOG(image1), getHOG(image2)))

    return fromDistancesToDivided(allDistancesLBP), fromDistancesToDivided(allDistancesHOG)

def getAverageImage(folder, nameOfTheFile):
    image_data = []
    for filename in os.listdir(folder):
        img = cv2.imread(folder + filename)
        if img is not None:
            image_data.append(img)

    dst = image_data[0]

    for i in range(len(image_data)):
        if i == 0:
            pass
        else:
            alpha = 1.0 / (i + 1)
            beta = 1.0 - alpha
            dst = cv2.addWeighted(image_data[i], alpha, dst, beta, 0.0)

    cv2.imwrite("C:/Users/User/Downloads/biom3/averageFaces/"+nameOfTheFile+".jpg", dst)


def lbp_features(img, radius=1, sampling_pixels=8):
    # LBP operates in single channel images so if RGB images are provided
    # we have to convert it to grayscale
    if (len(img.shape) > 2):
        img = img.astype(float)
        # RGB to grayscale convertion using Luminance
        img = img[:, :, 0] * 0.3 + img[:, :, 1] * 0.59 + img[:, :, 2] * 0.11

    # converting to uint8 type for 256 graylevels
    img = img.astype(np.uint8)

    # normalize values can also help improving description
    i_min = np.min(img)
    i_max = np.max(img)
    if (i_max - i_min != 0):
        img = (img - i_min) / (i_max - i_min)

    # compute LBP
    lbp = local_binary_pattern(img, sampling_pixels, radius, method="uniform")

    # LBP returns a matrix with the codes, so we compute the histogram
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, sampling_pixels + 3), range=(0, sampling_pixels + 2))

    # normalization
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)
    # return the histogram of Local Binary Patterns
    return hist

#def getHOG(color_image):
#    fd = hog(color_image, orientations=9, pixels_per_cell=(8, 8),
#                    cells_per_block=(2, 2), transform_sqrt=True, block_norm="L1", channel_axis=-1)
#    return fd

def getHOG(color_image):
    winSize = (20,20)
    blockSize = (10,10)
    blockStride = (5,5)
    cellSize = (10,10)
    nbins = 9
    derivAperture = 1
    winSigma = -1.
    histogramNormType = 0
    L2HysThreshold = 0.2
    gammaCorrection = 1
    nlevels = 64
    signedGradient = True

    hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,histogramNormType,L2HysThreshold,gammaCorrection,nlevels, signedGradient)
    return hog.compute(color_image)


def compareTwoImagesWithHOG(x, y):
    dst = euclidean(x, y)
    print(dst)
    return dst

def compareTwoImagesWithLBP(x, y):
    dst = euclidean(x, y)
    return dst

def plotROC(x,y,name,x1,y1,name1,x2,y2,name2):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(x, y, label=name)
    ax.plot(x1, y1, label=name1)
    ax.plot(x2, y2, label=name2)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.legend()
    plt.show()

def align(image, leftEyePts, rightEyePts, desiredLeftEye=(0.35, 0.35),desiredFaceWidth=100, desiredFaceHeight=100):
    # compute the center of mass for each eye
    leftEyeCenter = leftEyePts.mean(axis=0).astype("int")
    rightEyeCenter = rightEyePts.mean(axis=0).astype("int")

    # compute the angle between the eye centroids
    dY = rightEyeCenter[1] - leftEyeCenter[1]
    dX = rightEyeCenter[0] - leftEyeCenter[0]
    angle = np.degrees(np.arctan2(dY, dX))

    # compute the desired right eye x-coordinate based on the
    # desired x-coordinate of the left eye
    desiredRightEyeX = 1.0 - desiredLeftEye[0]

    # determine the scale of the new resulting image by taking
    # the ratio of the distance between eyes in the *current*
    # image to the ratio of distance between eyes in the
    # *desired* image
    dist = np.sqrt((dX ** 2) + (dY ** 2))
    desiredDist = (desiredRightEyeX - desiredLeftEye[0])
    desiredDist *= desiredFaceWidth
    scale = desiredDist / dist

    # compute center (x, y)-coordinates (i.e., the median point)
    # between the two eyes in the input image
    eyesCenter = (float((leftEyeCenter[0] + rightEyeCenter[0]) // 2),
                  float((leftEyeCenter[1] + rightEyeCenter[1]) // 2))

    # grab the rotation matrix for rotating and scaling the face
    m = cv2.getRotationMatrix2D(eyesCenter, angle, scale)

    # update the translation component of the matrix
    tX = desiredFaceWidth * 0.5
    tY = desiredFaceHeight * desiredLeftEye[1]
    m[0, 2] += (tX - eyesCenter[0])
    m[1, 2] += (tY - eyesCenter[1])

    # apply the affine transformation
    (w, h) = (desiredFaceWidth, desiredFaceHeight)
    output = cv2.warpAffine(image, m, (w, h), flags=cv2.INTER_CUBIC)

    # return the aligned face
    return output


def get_tresholds(method):
    if method == 'hog':
        return [number / 100 for number in range(0, 100, 1)]
    else:
        return []


def fromDistancesToDivided(distances):
    normalized = [float(i) / max(distances) for i in distances]
    rounded = [round(num, 2) for num in normalized]
    rounded.sort()
    boxes = np.linspace(0, 1, num=50)
    roundedBoxes = [round(num, 2) for num in boxes]

    numberOfNumbers = [0] * 50
    for index, number in enumerate(roundedBoxes):
        for percent in rounded:
            if percent <= number:
                numberOfNumbers[index] += 1

    divided = [x / len(rounded) for x in numberOfNumbers]
    return divided

def files_to_videos():
    folder = "video_input/"
    filenames = [f for f in listdir(folder) if isfile(join(folder, f))]

    videoNumber = 0
    for filename in filenames:
        filepath = (os.path.join(folder, filename))
        videoFile = np.load(filepath)
        colorImages = videoFile['colorImages']
        landmarks2D = videoFile['landmarks2D']
        videoNumber += 1
        fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
        h = colorImages.shape[0]
        w = colorImages.shape[1]
        frameNumber=0
        out = cv2.VideoWriter("C:/Users/User/Downloads/biom3/videa/video" + str(videoNumber) + ".avi", fourcc, 10, (w, h), isColor=True)
        shutil.rmtree(filename.split(".")[0], ignore_errors=True)
        os.mkdir(filename.split(".")[0])
        for i in range(0, colorImages.shape[-1], 3):
            left_eye = []
            left_eye.append(landmarks2D[37, :, i])
            left_eye.append(landmarks2D[38, :, i])
            left_eye.append(landmarks2D[39, :, i])
            left_eye.append(landmarks2D[40, :, i])
            left_eye.append(landmarks2D[41, :, i])
            left_eye.append(landmarks2D[41, :, i])
            right_eye = []
            right_eye.append(landmarks2D[43, :, i])
            right_eye.append(landmarks2D[44, :, i])
            right_eye.append(landmarks2D[45, :, i])
            right_eye.append(landmarks2D[46, :, i])
            right_eye.append(landmarks2D[47, :, i])
            right_eye.append(landmarks2D[48, :, i])
            left_eye = np.array(left_eye)
            right_eye = np.array(right_eye)
            img1 = colorImages[:, :, :, i]
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
            img1 = align(img1, left_eye, right_eye)
            cv2.imwrite("C:/Users/User/Downloads/biom3/" + str(filename.split(".")[0]) + "/" + str(frameNumber) +".jpg", img1)
            out.write(img1)
            frameNumber+=1
        out.release()
        getAverageImage("C:/Users/User/Downloads/biom3/" + str(filename.split(".")[0])+ "/", filename.split(".")[0])


def compRandomFaceFalse():
    csv_file = csv.reader(open('FalsePairs.csv', "r"), delimiter=",")
    allDistancesLBP, allDistancesHOG = [], []

    for row in csv_file:
        if (path.exists("C:/Users/User/Downloads/biom3/" + row[0].split('.')[0]) and path.exists("C:/Users/User/Downloads/biom3/" + row[1].split('.')[0])):
            file1 = "C:/Users/User/Downloads/biom3/" + row[0].split('.')[0] + "/" + str(random.randint(0, 12)) + ".jpg"
            file2 = "C:/Users/User/Downloads/biom3/" + row[1].split('.')[0] + "/" + str(random.randint(0, 12)) + ".jpg"
            image1 = cv2.imread(file1)
            image2 = cv2.imread(file2)
            allDistancesLBP.append(compareTwoImagesWithLBP(lbp_features(image1), lbp_features(image2)))
            allDistancesHOG.append(compareTwoImagesWithHOG(getHOG(image1), getHOG(image2)))

    return fromDistancesToDivided(allDistancesLBP), fromDistancesToDivided(allDistancesHOG)

def compRandomFaceTrue():
    csv_file = csv.reader(open('TruePairs.csv', "r"), delimiter=",")
    allDistancesLBP, allDistancesHOG = [], []

    for row in csv_file:
        if (path.exists("C:/Users/User/Downloads/biom3/" + row[0].split('.')[0]) and path.exists("C:/Users/User/Downloads/biom3/" + row[1].split('.')[0])):
            file1 = "C:/Users/User/Downloads/biom3/" + row[0].split('.')[0] + "/" + str(random.randint(0, 12)) + ".jpg"
            file2 = "C:/Users/User/Downloads/biom3/" + row[1].split('.')[0] + "/" + str(random.randint(0, 12)) + ".jpg"
            image1 = cv2.imread(file1)
            image2 = cv2.imread(file2)
            allDistancesLBP.append(compareTwoImagesWithLBP(lbp_features(image1), lbp_features(image2)))
            allDistancesHOG.append(compareTwoImagesWithHOG(getHOG(image1), getHOG(image2)))

    return fromDistancesToDivided(allDistancesLBP), fromDistancesToDivided(allDistancesHOG)

def compAllFramesTrue():
    csv_file = csv.reader(open('TruePairs.csv', "r"), delimiter=",")
    allDistancesLBP, allDistancesHOG  = [], []

    for row in csv_file:
        if (path.exists("C:/Users/User/Downloads/biom3/" + row[0].split('.')[0]) and path.exists("C:/Users/User/Downloads/biom3/" + row[1].split('.')[0])):
            onlyfilesDirOne = next(os.walk("C:/Users/User/Downloads/biom3/" + row[0].split('.')[0]))[2]
            onlyfilesDirTwo = next(os.walk("C:/Users/User/Downloads/biom3/" + row[1].split('.')[0]))[2]
            if(len(onlyfilesDirOne)>len(onlyfilesDirTwo)):
                onlyfiles=onlyfilesDirTwo
            else:
                onlyfiles = onlyfilesDirOne
            for x in range(len(onlyfiles)):
                file1 = "C:/Users/User/Downloads/biom3/" + row[0].split('.')[0] + "/" + str(x) + ".jpg"
                file2 = "C:/Users/User/Downloads/biom3/" + row[1].split('.')[0] + "/" + str(x) + ".jpg"
                image1 = cv2.imread(file1)
                image2 = cv2.imread(file2)
                allFaceFalseLBP = pd.DataFrame(
                    [[file1, file2, 'TP', 'LBP', compareTwoImagesWithLBP(lbp_features(image1), lbp_features(image2))]])
                allFaceFalseLBP.to_csv('allFace.csv', mode='a', header=False)
                allFaceFalseHOG = pd.DataFrame(
                    [[file1, file2, 'TP', 'HOG', compareTwoImagesWithHOG(getHOG(image1), getHOG(image2))]])
                allFaceFalseHOG.to_csv('allFace.csv', mode='a', header=False)
                allDistancesLBP.append(compareTwoImagesWithLBP(lbp_features(image1), lbp_features(image2)))
                allDistancesHOG.append(compareTwoImagesWithHOG(getHOG(image1), getHOG(image2)))

    return fromDistancesToDivided(allDistancesLBP), fromDistancesToDivided(allDistancesHOG)

def compAllFramesFalse():
    csv_file = csv.reader(open('FalsePairs.csv', "r"), delimiter=",")
    allDistancesLBP, allDistancesHOG  = [], []

    headers = pd.DataFrame([], columns=['Video 1', 'Video 2', 'Pair', 'Method', 'Distance'])
    headers.to_csv('allFace.csv')

    for row in csv_file:
        if (path.exists("C:/Users/User/Downloads/biom3/" + row[0].split('.')[0]) and path.exists("C:/Users/User/Downloads/biom3/" + row[1].split('.')[0])):
            onlyfilesDirOne = next(os.walk("C:/Users/User/Downloads/biom3/" + row[0].split('.')[0]))[2]
            onlyfilesDirTwo = next(os.walk("C:/Users/User/Downloads/biom3/" + row[1].split('.')[0]))[2]
            if(len(onlyfilesDirOne)>len(onlyfilesDirTwo)):
                onlyfiles=onlyfilesDirTwo
            else:
                onlyfiles = onlyfilesDirOne
            for x in range(len(onlyfiles)):
                file1 = "C:/Users/User/Downloads/biom3/" + row[0].split('.')[0] + "/" + str(x) + ".jpg"
                file2 = "C:/Users/User/Downloads/biom3/" + row[1].split('.')[0] + "/" + str(x) + ".jpg"
                image1 = cv2.imread(file1)
                image2 = cv2.imread(file2)
                allFaceFalseLBP = pd.DataFrame([[file1, file2, 'FP', 'LBP', compareTwoImagesWithLBP(lbp_features(image1), lbp_features(image2))]])
                allFaceFalseLBP.to_csv('allFace.csv', mode='a', header=False)
                allFaceFalseHOG = pd.DataFrame([[file1, file2, 'FP', 'HOG', compareTwoImagesWithHOG(getHOG(image1), getHOG(image2))]])
                allFaceFalseHOG.to_csv('allFace.csv', mode='a', header=False)
                allDistancesLBP.append(compareTwoImagesWithLBP(lbp_features(image1), lbp_features(image2)))
                allDistancesHOG.append(compareTwoImagesWithHOG(getHOG(image1), getHOG(image2)))

    return fromDistancesToDivided(allDistancesLBP), fromDistancesToDivided(allDistancesHOG)


#files_to_videos()

avg_TP_Distances_LBP, avg_TP_Distances_HOG = compAvgFaceTrue()
avg_IP_Distances_LBP, avg_IP_Distances_HOG = compAvgFaceFalse()

random_IP_Distances_LBP, random_IP_Distances_HOG = compRandomFaceFalse()
random_TP_Distances_LBP, random_TP_Distances_HOG = compRandomFaceTrue()

all_IP_Distances_LBP, all_IP_Distances_HOG = compAllFramesFalse()
all_TP_Distances_LBP, all_TP_Distances_HOG = compAllFramesTrue()


plotROC(all_TP_Distances_LBP, all_IP_Distances_LBP, 'LBP - ROC CURVE ALL', random_TP_Distances_LBP, random_IP_Distances_LBP, 'LBP - ROC CURVE RANDOM', avg_TP_Distances_LBP, avg_IP_Distances_LBP, 'LBP - ROC CURVE AVG')
print('AUC OF LBP')
AUClbp = f'ROC Curve (AUC={auc(all_TP_Distances_LBP, all_IP_Distances_LBP):.4f})'
print(AUClbp)
plotROC(all_TP_Distances_HOG, all_IP_Distances_HOG,'HOG - ROC CURVE ALL', random_TP_Distances_HOG, random_IP_Distances_HOG, 'HOG - ROC CURVE RANDOM', avg_TP_Distances_HOG, avg_IP_Distances_HOG, 'HOG - ROC CURVE AVG')
print('AUC OF HOG')
AUChog = f'ROC Curve (AUC={auc(all_TP_Distances_HOG, all_IP_Distances_HOG):.4f})'
print(AUChog)
