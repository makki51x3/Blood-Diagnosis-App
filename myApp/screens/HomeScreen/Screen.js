import { useEffect } from 'react';
import IsVisibleBtn from './Components/IsVisibleBtn';
import IsVisibleTxt from './Components/IsVisibleTxt';
import openImagePicker from './Components/openImagePicker';
import React from 'react';
import { Dimensions, StyleSheet,TouchableOpacity, Text, ImageBackground,  View, Image, Platform } from "react-native";
import axios from "axios";

import { useSelector, useDispatch } from "react-redux";
import { updateMask, updateLabel, updateInfo } from "../../redux/slices/resultSlice";
import { updateLoading, updateImage } from "../../redux/slices/homePageSlice";

const ScreenHeight = Dimensions.get("window").height;
const ScreenWidth = Dimensions.get("window").width;

const HomeScreen = ({navigation}) => {

    // Get data from the redux store
    const dispatch = useDispatch();
    
    // Results Slice Reducer
    const mask = useSelector((state) => state.resultReducer.mask);

    // Home Page Slice
    const image = useSelector((state) => state.homePageReducer.image);
    const loading = useSelector((state) => state.homePageReducer.loading);
  
    useEffect(
      () => {
        if(mask){
          navigation.navigate('Result');
        }
      }, [mask,loading]
    );
      
    const uploadImage = async () => {
      var config = { headers: {'Content-Type': 'application/json'}}
      
      //Check if any file is selected or not
      if (image) {
        dispatch(updateLoading(true));
        axios.post("https://cdab-94-187-11-155.ngrok.io/"+"/send-image/",{selectedImg:image},  config)
        .then((res) => {
          if (res.status >= 200 && res.status <= 299) {
            dispatch(updateInfo(res.data.info));
            dispatch(updateMask(res.data.maskBase64));
            dispatch(updateLabel(res.data.labelBase64));
          } 
          else {
            alert('Server Error\n Try re-uploading the image');
          }
          dispatch(updateLoading(false));
        }, (error) => {
          // no Timeout
          console.log(error);
          alert("Failed to load resource.\n Try re-uploading the image");
          dispatch(updateLoading(false));
        });
      }
      else {
        alert('Select an Image First');
      }
    }
  
    return (
    <ImageBackground source= {require('../../assets/background.png')}  resizeMode="cover" style={styles.imageBG}>
      <View style={styles.container}>
        <View style={styles.row}>
          <Image source={!image?require('../../assets/transparent.png'):{ uri: image }} style = {styles.image} />
        </View>
        <View style={styles.row}>
          <IsVisibleTxt state={loading}></IsVisibleTxt>
        </View>
        <View style={styles.row}>
          <TouchableOpacity
            style={styles.btn}
            onPress={()=>{
              dispatch(updateMask(""));
              uploadImage();
            }}
            underlayColor='#fff'>
            <Text style={styles.btnText}>Run Diagnosis</Text>
          </TouchableOpacity>
          <TouchableOpacity
              style={styles.btn}
              onPress={()=>{
                  openImagePicker().then((res)=>{
                  dispatch(updateImage(res.Image_URI));
                });
              }}
              underlayColor='#fff'>
              <Text style={styles.btnText}>Select Image</Text>
          </TouchableOpacity>
          <IsVisibleBtn navigation={navigation} state={mask} ></IsVisibleBtn>
        </View>
      </View>
    </ImageBackground>
    );
  }

const styles = StyleSheet.create({
  container:{
    justifyContent: "center",
    flex:1,
    backgroundColor:"rgba(0, 0, 0,0.1)",
    width:"100%"
  },
  image: {
    height: ScreenHeight*0.4,
    width: (Platform.OS == "ios"|| Platform.OS =="android")?ScreenWidth*0.7:ScreenHeight*0.4, 
    resizeMode:"cover",
    borderRadius:10,
  },
  imageBG: {
    flex: 1,
    justifyContent: (Platform.OS == "ios"||Platform.OS =="android")? "space-evenly":"space-around",
    alignItems: "center",
    paddingTop: (Platform.OS == "ios"||Platform.OS =="android")? 30:0,
    paddingBottom: (Platform.OS == "ios"||Platform.OS =="android")? 30:0,
  },
  row: {
    flexDirection: 'row',
    justifyContent:"center",
    marginTop: ScreenHeight*0.07,
    },
  btn:{
    marginRight:40,
    marginLeft:40,
    marginTop:10,
    paddingTop:10,
    paddingBottom:10,
    backgroundColor:'#112031',
    borderRadius:10,
    borderWidth: 1,
    borderColor: '#152D35'
  },
  btnText:{
    color:'white',
    textAlign:'center',
    paddingLeft : 10,
    paddingRight : 10,
    fontWeight: "500",
}
});

export default HomeScreen;
