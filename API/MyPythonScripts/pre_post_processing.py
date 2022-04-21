# Imported libraries
import tensorflow as tf
import numpy as np
import cv2
from skimage import measure
from skimage.color import label2rgb, rgb2gray
from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
import math
import pandas as pd
from statistics import median
from MyPythonScripts import smooth_tiled_predictions as sp

########################################################################################################################
######      PREPROCESS IMAGE TO APPROPRIATE DETECT PATCH SIZE BASED ON THE DIAMETER OF THE BIGGEST CELL       #####################    
########################################################################################################################

def getRatio(image,debugging=False):

    #scale = 0.6 #microns/pixel
    
    # The input image.
    img = rgb2gray(image)
    #plt.imshow(image)
    
    #Generate thresholded image
    thresholded_img = (img < threshold_otsu(img)).astype(np.uint8)
    #plt.imshow(thresholded_img)
    
    #Remove edge touching regions
    edge_touching_removed = clear_border(thresholded_img)
    #plt.imshow(edge_touching_removed)
    
    #fill holes in image
    kernel = np.ones((4,4),np.uint8)
    filled_holes = cv2.dilate(edge_touching_removed,kernel,iterations=1)
    cv2.floodFill(filled_holes, None, (0, 0), 255)
    #plt.imshow(filled_holes)
    
    # invert binary image
    filled_holes = cv2.bitwise_not(filled_holes)
    #plt.imshow(filled_holes)
    
    kernel = np.ones((3,3),np.uint8)
    # Morphological operations to remove small noise and holes 
    erosion = cv2.erode(filled_holes,kernel,iterations=3)
    #plt.imshow(erosion)
    
    #Watershed should find this area for us. 
    dilation = cv2.dilate(erosion,kernel,iterations=3)
    #plt.imshow(dilation)
    
    dist_transform = cv2.distanceTransform(erosion,cv2.DIST_L2,3)
    ret2, sure_fg = cv2.threshold(dist_transform,0.01*dist_transform.max(),255,0)
    sure_fg = np.uint8(sure_fg)
    #plt.imshow(sure_fg)
    
    unknown = cv2.subtract(dilation,sure_fg)
    #plt.imshow(unknown)
    
    #Now we create a marker and label the regions inside. 
    ret3, markers = cv2.connectedComponents(sure_fg)
    markers = markers+10
    # Now, mark the region of unknown with zero
    markers[unknown==1] = 0
    #plt.imshow(markers)
    
    #Now we are ready for watershed filling. 
    markers = cv2.watershed(image,markers)
    markers[markers>10] = 255
    markers[markers<=10] = 0
    #plt.imshow(markers)
    
    # extract labels from markers
    labels = measure.label(markers, connectivity=markers.ndim)
    #plt.imshow(labels)
    
    # extract feature properties from resulting image
    props=measure.regionprops_table(labels, img,properties=['equivalent_diameter'])
    df = pd.DataFrame(props)
    
    #print(df.head())

    # used in debugging mode to  generate dummy data for faster processing when testing
    if debugging==True:
        return filled_holes,df
    else:
        # calculate the appropriate ratio for cropping depending on the largest diameter
        diameter = math.floor(median(df['equivalent_diameter']))
        return diameter*2

##########################################################################################################   
#########################             Predict using smooth blending                   ####################  
##########################################################################################################  

def smoothBlending(img):
    baseDir='C:\\Users\\1999m\\Desktop\\lira competition\\Model\\'
    model = tf.keras.models.load_model(baseDir+'FINAL_pre_trained_unfrozen_unet_model_100epochs.h5', compile=False)
    ratio = 256/getRatio(img)
    #print("the ratio is\t",ratio)
    #plt.imshow(img)
    
    dim = (int(ratio * img.shape[1]), int(ratio * img.shape[0]))
    #print("img:\t",img.shape)
    #print("dim:\t",dim)
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA )/255.
    #plt.imshow(resized)
    
    predictions_smooth = sp.predict_img_with_smooth_windowing(
        resized,
        window_size=256,
        subdivisions=2,  # Minimal amount of overlap for windowing. Must be an even number.
        nb_classes=1,
        pred_func=(
            lambda img_batch_subdiv: model.predict((img_batch_subdiv))
        )
    )
    
    predictions_smooth = cv2.resize(predictions_smooth, (img.shape[1],img.shape[0]), interpolation = cv2.INTER_AREA )
    #plt.imshow(predictions_smooth)
    
    # post processing and thresholding of prediction
    predictions_smooth=(predictions_smooth > 0.8).astype(np.uint8)*255
    
    
    # Data extraction from the smoothed prediction
    df, image_labels_overlay = extractInfo(predictions_smooth,img)
    image_labels_overlay=(image_labels_overlay*255).astype(np.uint8)
    #print("before deletion\n",df,"\n")
    
    # Data Cleaning
    # To delete small regions less than 70% of median diameter
    df = df[df['equivalent_diameter'] > 256/(ratio*3)]
    
    return predictions_smooth,image_labels_overlay, df
        
    #print(predictions_smooth.shape)
    #plt.imshow(predictions_smooth[:,:,0]>0.5)
    
    """
    #used for multiclass prediction
    final_prediction = np.argmax(predictions_smooth, axis=2)
    plt.imshow(final_prediction)
    """

########################################################################################################################
############           POSTPROCESS PREDICTION TO EXTRACT INFORMATION AFTER PREDICTION         #####################    
########################################################################################################################

def extractInfo(thresholded_img,img):
    
    #plt.imshow(thresholded_img)
    
    #Remove edge touching regions
    edge_touching_removed = clear_border(thresholded_img)
    #plt.imshow(edge_touching_removed)
    
    kernel = np.ones((3,3),np.uint8)
    # Morphological operations to remove small noise and holes 
    erosion = cv2.erode(edge_touching_removed,kernel,iterations=4)
    #plt.imshow(erosion)
    
    #Watershed should find this area for us. 
    dilation = cv2.dilate(erosion,kernel,iterations=4)
    #plt.imshow(dilation)
    
    #Now we create a marker and label the regions inside. 
    ret3, markers = cv2.connectedComponents(dilation)
    #plt.imshow(markers)
    
    labels = measure.label(markers, connectivity=markers.ndim)
    #plt.imshow(labels)
    
    image_labels_overlay = label2rgb(labels, image=img)
    #plt.imshow(image_labels_overlay)
    
    props=measure.regionprops_table(labels, img,properties=['area', 'equivalent_diameter','perimeter','eccentricity','moments_hu','orientation'])
    df = pd.DataFrame(props)

    return df,image_labels_overlay