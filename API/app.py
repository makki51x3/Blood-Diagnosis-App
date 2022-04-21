
# Imported libraries
from flask import Flask, request, jsonify
import numpy as np
from PIL import Image
import base64
from io import BytesIO
import requests
from flask_cors import CORS
from MyPythonScripts.pre_post_processing import getRatio, smoothBlending


app = Flask(__name__)

# Converts an array to a PNG Base64 encoded image 
def image_base64(array):
    img = Image.fromarray(array)
    img_byte_arr = BytesIO()
    img.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()
    img_str = base64.b64encode(img_byte_arr)
    return str(img_str)

# adds cross origin support for the app
CORS(app, support_credentials=True)

# Send-image route recieves an image in base64 or a URL 
# Returns AI prediction and labeled prediction with a list of features
@app.route("/send-image/", methods=['POST'])
def image_check(): 
    content = request.get_json()
    url = content['selectedImg']   
    
    # Convert JSON to an RGB array
    try:
        # Base64 jpeg 
        if "data:image/jpeg;base64," in url:
            base_string = url.replace("data:image/jpeg;base64,", "")
            decoded_img = base64.b64decode(base_string)
            img = Image.open(BytesIO(decoded_img)).convert('RGB')
        # Base64 jpg     
        elif "data:image/jpg;base64," in url:
            base_string = url.replace("data:image/jpg;base64,", "")
            decoded_img = base64.b64decode(base_string)
            img = Image.open(BytesIO(decoded_img)).convert('RGB')
            
        # Base64 png 
        elif "data:image/png;base64," in url:
            base_string = url.replace("data:image/png;base64,", "")
            decoded_img = base64.b64decode(base_string)
            img = Image.open(BytesIO(decoded_img)).convert('RGB')

        # Regular URL 
        else:
            response = requests.get(url)
            img = Image.open(BytesIO(response.content)).convert("RGB")
            
    except Exception as e:
        return {"error": "The image is not recognized"+str(e)}, 415
    
    # Set to true to skip waiting for the AI to finish processing and provide dummy data instead
    debugging=False
    
    if debugging == True:
        prediction, df = getRatio(np.array(img),debugging)
        respond = {"info": df.to_json()}
        respond['maskBase64'] = "data:image/png;base64,"+image_base64(prediction)[2:-1]
        return jsonify(respond), 200 
        
    else:
        # Perform smooth blending on image
        prediction, image_labels_overlay, df = smoothBlending(np.array(img))

        # Convert returned resuts to JSON
        respond = {"info": df.to_json()}
        respond['maskBase64'] = "data:image/png;base64,"+image_base64(prediction)[2:-1]
        respond['labelBase64'] = "data:image/png;base64,"+image_base64(image_labels_overlay)[2:-1]
        
        return jsonify(respond), 200 

if __name__ == '__main__':
   app.run()