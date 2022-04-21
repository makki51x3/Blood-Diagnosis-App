
from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
import base64
import io
from io import BytesIO
import requests
from flask_cors import CORS,cross_origin
import scipy.signal
from tqdm import tqdm
import gc
#from patchify import patchify, unpatchify
import cv2
from skimage import measure#, img_as_ubyte, io
from skimage.color import label2rgb, rgb2gray
from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
import pandas as pd
import math
import matplotlib.pyplot as plt
from statistics import median

"""
if __name__ == '__main__':
    PLOT_PROGRESS = False
    # See end of file for the rest of the __main__.
else:
    PLOT_PROGRESS = False
"""
def _spline_window(window_size, power=2):
    """
    Squared spline (power=2) window function:
    https://www.wolframalpha.com/input/?i=y%3Dx**2,+y%3D-(x-2)**2+%2B2,+y%3D(x-4)**2,+from+y+%3D+0+to+2
    """
    intersection = int(window_size/4)
    wind_outer = (abs(2*(scipy.signal.triang(window_size))) ** power)/2
    wind_outer[intersection:-intersection] = 0

    wind_inner = 1 - (abs(2*(scipy.signal.triang(window_size) - 1)) ** power)/2
    wind_inner[:intersection] = 0
    wind_inner[-intersection:] = 0

    wind = wind_inner + wind_outer
    wind = wind / np.average(wind)
    return wind


cached_2d_windows = dict()
def _window_2D(window_size, power=2):
    """
    Make a 1D window function, then infer and return a 2D window function.
    Done with an augmentation, and self multiplication with its transpose.
    Could be generalized to more dimensions.
    """
    # Memoization
    global cached_2d_windows
    key = "{}_{}".format(window_size, power)
    if key in cached_2d_windows:
        wind = cached_2d_windows[key]
    else:
        wind = _spline_window(window_size, power)
        wind = np.expand_dims(np.expand_dims(wind, 1), 1)      #Changed from 3, 3, to 1, 1 
        wind = wind * wind.transpose(1, 0, 2)
        """
        if PLOT_PROGRESS:
            # For demo purpose, let's look once at the window:
            plt.imshow(wind[:, :, 0], cmap="viridis")
            plt.title("2D Windowing Function for a Smooth Blending of "
                      "Overlapping Patches")
            plt.show()
        """
        cached_2d_windows[key] = wind
    return wind


def _pad_img(img, window_size, subdivisions):
    """
    Add borders to img for a "valid" border pattern according to "window_size" and
    "subdivisions".
    Image is an np array of shape (x, y, nb_channels).
    """
    aug = int(round(window_size * (1 - 1.0/subdivisions)))
    more_borders = ((aug, aug), (aug, aug), (0, 0))
    ret = np.pad(img, pad_width=more_borders, mode='reflect')
    # gc.collect()
    """
    if PLOT_PROGRESS:
        # let's look once at the window:
        plt.imshow(ret)
        plt.title("Padded Image for Using Tiled Prediction Patches\n"
                  "(notice the reflection effect on the padded borders)")
        plt.show()
        """
    return ret


def _unpad_img(padded_img, window_size, subdivisions):
    """
    Undo what's done in the `_pad_img` function.
    Image is an np array of shape (x, y, nb_channels).
    """
    aug = int(round(window_size * (1 - 1.0/subdivisions)))
    ret = padded_img[
        aug:-aug,
        aug:-aug,
        :
    ]
    # gc.collect()
    return ret


def _rotate_mirror_do(im):
    """
    Duplicate an np array (image) of shape (x, y, nb_channels) 8 times, in order
    to have all the possible rotations and mirrors of that image that fits the
    possible 90 degrees rotations.
    It is the D_4 (D4) Dihedral group:
    https://en.wikipedia.org/wiki/Dihedral_group
    """
    mirrs = []
    mirrs.append(np.array(im))
    mirrs.append(np.rot90(np.array(im), axes=(0, 1), k=1))
    mirrs.append(np.rot90(np.array(im), axes=(0, 1), k=2))
    mirrs.append(np.rot90(np.array(im), axes=(0, 1), k=3))
    im = np.array(im)[:, ::-1]
    mirrs.append(np.array(im))
    mirrs.append(np.rot90(np.array(im), axes=(0, 1), k=1))
    mirrs.append(np.rot90(np.array(im), axes=(0, 1), k=2))
    mirrs.append(np.rot90(np.array(im), axes=(0, 1), k=3))
    return mirrs


def _rotate_mirror_undo(im_mirrs):
    """
    merges a list of 8 np arrays (images) of shape (x, y, nb_channels) generated
    from the `_rotate_mirror_do` function. Each images might have changed and
    merging them implies to rotated them back in order and average things out.
    It is the D_4 (D4) Dihedral group:
    https://en.wikipedia.org/wiki/Dihedral_group
    """
    origs = []
    origs.append(np.array(im_mirrs[0]))
    origs.append(np.rot90(np.array(im_mirrs[1]), axes=(0, 1), k=3))
    origs.append(np.rot90(np.array(im_mirrs[2]), axes=(0, 1), k=2))
    origs.append(np.rot90(np.array(im_mirrs[3]), axes=(0, 1), k=1))
    origs.append(np.array(im_mirrs[4])[:, ::-1])
    origs.append(np.rot90(np.array(im_mirrs[5]), axes=(0, 1), k=3)[:, ::-1])
    origs.append(np.rot90(np.array(im_mirrs[6]), axes=(0, 1), k=2)[:, ::-1])
    origs.append(np.rot90(np.array(im_mirrs[7]), axes=(0, 1), k=1)[:, ::-1])
    return np.mean(origs, axis=0)


def _windowed_subdivs(padded_img, window_size, subdivisions, nb_classes, pred_func):
    """
    Create tiled overlapping patches.
    Returns:
        5D numpy array of shape = (
            nb_patches_along_X,
            nb_patches_along_Y,
            patches_resolution_along_X,
            patches_resolution_along_Y,
            nb_output_channels
        )
    Note:
        patches_resolution_along_X == patches_resolution_along_Y == window_size
    """
    WINDOW_SPLINE_2D = _window_2D(window_size=window_size, power=2)

    step = int(window_size/subdivisions)
    padx_len = padded_img.shape[0]
    pady_len = padded_img.shape[1]
    subdivs = []

    for i in range(0, padx_len-window_size+1, step):
        subdivs.append([])
        for j in range(0, pady_len-window_size+1, step):            #Changed padx to pady (Bug in original code)
            patch = padded_img[i:i+window_size, j:j+window_size, :]
            subdivs[-1].append(patch)

    # Here, `gc.collect()` clears RAM between operations.
    # It should run faster if they are removed, if enough memory is available.
    gc.collect()
    subdivs = np.array(subdivs)
    gc.collect()
    a, b, c, d, e = subdivs.shape
    subdivs = subdivs.reshape(a * b, c, d, e)
    gc.collect()

    subdivs = pred_func(subdivs)
    gc.collect()
    subdivs = np.array([patch * WINDOW_SPLINE_2D for patch in subdivs])
    gc.collect()

    # Such 5D array:
    subdivs = subdivs.reshape(a, b, c, d, nb_classes)
    gc.collect()

    return subdivs


def _recreate_from_subdivs(subdivs, window_size, subdivisions, padded_out_shape):
    """
    Merge tiled overlapping patches smoothly.
    """
    step = int(window_size/subdivisions)
    padx_len = padded_out_shape[0]
    pady_len = padded_out_shape[1]

    y = np.zeros(padded_out_shape)

    a = 0
    for i in range(0, padx_len-window_size+1, step):
        b = 0
        for j in range(0, pady_len-window_size+1, step):                #Changed padx to pady (Bug in original code)
            windowed_patch = subdivs[a, b]
            y[i:i+window_size, j:j+window_size] = y[i:i+window_size, j:j+window_size] + windowed_patch
            b += 1
        a += 1
    return y / (subdivisions ** 2)


def predict_img_with_smooth_windowing(input_img, window_size, subdivisions, nb_classes, pred_func):
    """
    Apply the `pred_func` function to square patches of the image, and overlap
    the predictions to merge them smoothly.
    See 6th, 7th and 8th idea here:
    http://blog.kaggle.com/2017/05/09/dstl-satellite-imagery-competition-3rd-place-winners-interview-vladimir-sergey/
    """
    pad = _pad_img(input_img, window_size, subdivisions)
    pads = _rotate_mirror_do(pad)

    # Note that the implementation could be more memory-efficient by merging
    # the behavior of `_windowed_subdivs` and `_recreate_from_subdivs` into
    # one loop doing in-place assignments to the new image matrix, rather than
    # using a temporary 5D array.

    # It would also be possible to allow different (and impure) window functions
    # that might not tile well. Adding their weighting to another matrix could
    # be done to later normalize the predictions correctly by dividing the whole
    # reconstructed thing by this matrix of weightings - to normalize things
    # back from an impure windowing function that would have badly weighted
    # windows.

    # For example, since the U-net of Kaggle's DSTL satellite imagery feature
    # prediction challenge's 3rd place winners use a different window size for
    # the input and output of the neural net's patches predictions, it would be
    # possible to fake a full-size window which would in fact just have a narrow
    # non-zero dommain. This may require to augment the `subdivisions` argument
    # to 4 rather than 2.

    res = []
    for pad in tqdm(pads):
        # For every rotation:
        sd = _windowed_subdivs(pad, window_size, subdivisions, nb_classes, pred_func)
        one_padded_result = _recreate_from_subdivs(
            sd, window_size, subdivisions,
            padded_out_shape=list(pad.shape[:-1])+[nb_classes])

        res.append(one_padded_result)

    # Merge after rotations:
    padded_results = _rotate_mirror_undo(res)

    prd = _unpad_img(padded_results, window_size, subdivisions)

    prd = prd[:input_img.shape[0], :input_img.shape[1], :]
    """
    if PLOT_PROGRESS:
        print(prd.shape)
        
        plt.imshow(prd)
        plt.title("Smoothly Merged Patches that were Tiled Tighter")
        plt.show()
    """
        
    return prd

# prediction time augmentation for more accurate results
def PTA(model,image):
    
    p0 = model.predict(np.expand_dims(image, axis=0))[0][:, :, 0]
    
    p1 = model.predict(np.expand_dims(np.fliplr(image), axis=0))[0][:, :, 0]
    p1 = np.fliplr(p1)
    
    p2 = model.predict(np.expand_dims(np.flipud(image), axis=0))[0][:, :, 0]
    p2 = np.flipud(p2)
    
    p3 = model.predict(np.expand_dims(np.fliplr(np.flipud(image)), axis=0))[0][:, :, 0]
    p3 = np.fliplr(np.flipud(p3))
    
    thresh = 0.5
    p_thresh_avg = ((p0 + p1 + p2 + p3) / 4) > thresh
    p = (p_thresh_avg).astype(np.uint8)
    p[p > 0 ] = 255
    return p

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

###################################################################################################   
##################                Predict using smooth blending                      ##############  
###################################################################################################  

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
    
    predictions_smooth = predict_img_with_smooth_windowing(
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

"""
###################################################################################################   
##################     Predict patch by patch with no smooth blending (NOT USED )    ##############  
###################################################################################################  

baseDir='C:\\Users\\1999m\\Desktop\\lira competition\\LIRA\\Model\\'
unet_model = tf.keras.models.load_model(baseDir+'FINAL_pre_trained_unfrozen_unet_model_100epochs.h5', compile=False)
    
img = cv2.imread(path+img_name, cv2.IMREAD_COLOR)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# size of patches
patch_size = diameter
# Number of classes 
n_classes = 1

SIZE_X = (img.shape[1]//patch_size)*patch_size #Nearest size divisible by our patch size
SIZE_Y = (img.shape[0]//patch_size)*patch_size #Nearest size divisible by our patch size
large_img = Image.fromarray(img)
large_img = large_img.crop((0 ,0, SIZE_X, SIZE_Y))  #Crop from top left corner
large_img = np.array(large_img)     

patches_img = patchify(large_img, (patch_size, patch_size, 3), step=patch_size)  #Step=256 for 256 patches means no overlap
later=patches_img.shape[:2]+patches_img.shape[3:-1]

patches_img = patches_img[:,:,0,:,:,:]

patched_prediction = []
for i in range(patches_img.shape[0]):
    for j in range(patches_img.shape[1]):
        single_patch_img = patches_img[i,j,:,:,:]
        single_patch_img = cv2.resize(single_patch_img, dsize=(256, 256), interpolation=cv2.INTER_CUBIC)/255.
        single_patch_img = single_patch_img
        #pred = unet_model.predict(single_patch_img)    
        pred = PTA(unet_model,single_patch_img)                                                             
        patched_prediction.append(cv2.resize(pred, dsize=(patch_size, patch_size), interpolation=cv2.INTER_CUBIC))

#print(np.unique(patched_prediction))
#plt.imshow(pred)

patched_prediction = np.array(patched_prediction)
patched_prediction = np.reshape(patched_prediction, later)
unpatched_prediction = unpatchify(patched_prediction, large_img.shape[:-1])

plt.imshow(unpatched_prediction)
plt.imshow(large_img)

"""

########################################################################################################################
####################################                              SERVER AREA                              #####################    
########################################################################################################################

app = Flask(__name__)

@app.post("/reactImage")
def add_reactImage():
    if request.is_json:
        imageJson = request.get_json()
        return imageJson, 201
    return {"error": "Request must be JSON"}, 415
 
def pil_base64(imageDir):
    with open(imageDir, "rb") as img_file:
        my_string = base64.b64encode(img_file.read())
    return str(my_string)

def image_base64(array):
    img = Image.fromarray(array)
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()
    img_str = base64.b64encode(img_byte_arr)
    return str(img_str)

CORS(app, support_credentials=True)

@app.route("/send-image/", methods=['POST'])
def image_check():
    content = request.get_json()
    url = content['selectedImg']   
    
    try:
        # Base64 DATA
        if "data:image/jpeg;base64," in url:
            base_string = url.replace("data:image/jpeg;base64,", "")
            decoded_img = base64.b64decode(base_string)
            img = Image.open(BytesIO(decoded_img)).convert('RGB')
            
        elif "data:image/jpg;base64," in url:
            base_string = url.replace("data:image/jpg;base64,", "")
            decoded_img = base64.b64decode(base_string)
            img = Image.open(BytesIO(decoded_img)).convert('RGB')
            
        # Base64 DATA
        elif "data:image/png;base64," in url:
            base_string = url.replace("data:image/png;base64,", "")
            decoded_img = base64.b64decode(base_string)
            img = Image.open(BytesIO(decoded_img)).convert('RGB')

        # Regular URL From DATA
        else:
            response = requests.get(url)
            img = Image.open(BytesIO(response.content)).convert("RGB")
            
    except Exception as e:
        return {"error": str(e)}, 415
    
    
    debugging=False
    
    if debugging == True:
        prediction, df = getRatio(np.array(img),debugging)
        respond = {"info": df.to_json()}
        respond['maskBase64'] = "data:image/png;base64,"+image_base64(prediction)[2:-1]
        return jsonify(respond), 200 
        
    else:
        prediction, image_labels_overlay, df = smoothBlending(np.array(img))

        respond = {"info": df.to_json()}
        respond['maskBase64'] = "data:image/png;base64,"+image_base64(prediction)[2:-1]
        respond['labelBase64'] = "data:image/png;base64,"+image_base64(image_labels_overlay)[2:-1]
        
        return jsonify(respond), 200 

if __name__ == '__main__':
   app.run()

"""
# can be used later to add saving mechanism
#File naming process for nameless base64 data.
#We are using the timestamp as a file_name.
from datetime import datetime
dateTimeObj = datetime.now()
file_name_for_base64_data = dateTimeObj.strftime("%d-%b-%Y--(%H-%M-%S)")
file_name = file_name_for_base64_data + ".jpeg"
img.save("./static/recievedImages/"+file_name, "jpeg")
"""    
    
"""
    #Image preprocessing
    img = img.resize((256, 256))
    image = np.array(img)  
    image = image/255.
    
    #Prediction time augmentation
    baseDir='C:\\Users\\1999m\\Desktop\\lira competition\\LIRA\\Model\\'
    model = tf.keras.models.load_model(baseDir+'FINAL_pre_trained_unfrozen_unet_model_100epochs.h5', compile=False)

    p0 = model.predict(np.expand_dims(image, axis=0))[0][:, :, 0]
    
    p1 = model.predict(np.expand_dims(np.fliplr(image), axis=0))[0][:, :, 0]
    p1 = np.fliplr(p1)
    
    p2 = model.predict(np.expand_dims(np.flipud(image), axis=0))[0][:, :, 0]
    p2 = np.flipud(p2)
    
    p3 = model.predict(np.expand_dims(np.fliplr(np.flipud(image)), axis=0))[0][:, :, 0]
    p3 = np.fliplr(np.flipud(p3))
    
    thresh = 0.5
    p_thresh_avg = ((p0 + p1 + p2 + p3) / 4) > thresh
    p = (p_thresh_avg).astype(np.uint8)
    p[p > 0 ] = 255
"""
########################################################################################################################
####################################                              SCRATCH AREA                              #####################    
########################################################################################################################
"""
#Return an RGB image where color-coded labels are painted over the image.
#Using label2rgb
labels = measure.label(edge_touching_removed, connectivity=img.ndim)
plt.imshow(labels)

image_labels_overlay = label2rgb(labels, image=img)
plt.imshow(image_labels_overlay)

#plt.imsave("labeled_result.jpg", image_labels_overlay) 
    
#Compute image properties and return them as a pandas-compatible table.
#Available regionprops: area, bbox, centroid, convex_area, coords, eccentricity,
# equivalent diameter, euler number, label, intensity image, major axis length, 
#max intensity, mean intensity, moments, orientation, perimeter, solidity, and many more

all_props=measure.regionprops(labels, img)
#Can print various parameters for all objects
for prop in all_props:
    print('{}: Area: {}, Perimeter: {}, Diameter {}'.format(prop.label, prop.area, prop.perimeter, prop.equivalent_diameter))

import pandas as pd
props=measure.regionprops_table(labels, img,properties=['label','area', 'equivalent_diameter','perimeter', 'equivalent_diameter'])

df = pd.DataFrame(props)
print(df.head())

#To delete small regions...
df = df[df['equivalent_diameter'] > 80]
print(df.head())

#######################################################
#Convert to micron scale
df['area_sq_microns'] = df['area'] * (scale**2)
df['equivalent_diameter_microns'] = df['equivalent_diameter'] * (scale)
print(df.head())

df.to_csv('data/cast_iron_measurements.csv')

"""