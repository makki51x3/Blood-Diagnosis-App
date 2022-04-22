
"""
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
"""

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