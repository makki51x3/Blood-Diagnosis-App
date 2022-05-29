import * as ImagePicker from 'expo-image-picker';
   
const openImagePicker = async () => {

  let mimeType="";
  let Image_URI="";

  //Ask for permission to access the media library
  let permissionResult = await ImagePicker.requestMediaLibraryPermissionsAsync();
  if (permissionResult.granted == false) {
    alert("Permission to access camera roll is required!");
    return;
  }

  //Wait for user to select an image
  let pickerResult = await ImagePicker.launchImageLibraryAsync({base64:true});

  //Handle the image if selected
  if(pickerResult.cancelled == false){
    //configure the mimetype
    if((pickerResult.uri).includes("png")){
      mimeType="png";
    }
    else if((pickerResult.uri).includes("jpeg")){
      mimeType="jpeg";
    }
    else{
      mimeType="jpg";
    }
    //reform selected image to base64 encoding
    if(typeof(pickerResult.base64) == "undefined"){
      Image_URI=pickerResult.uri;
    }
    else{
      Image_URI='data:image/'+mimeType+';base64,' + pickerResult.base64;
    }
  }

  return ({mimeType,Image_URI});
}

export default openImagePicker;