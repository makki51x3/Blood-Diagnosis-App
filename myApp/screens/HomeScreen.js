import { useState, useEffect } from 'react';
import Br from '../components/Br';
import IsVisibleBtn from '../components/HomeScreen/IsVisibleBtn';
import IsVisibleTxt from '../components/HomeScreen/IsVisibleTxt';
import openImagePicker from '../components/HomeScreen/openImagePicker';

import React from 'react';
import { Dimensions, StyleSheet,TouchableOpacity, Text, ImageBackground,  View, Image, Platform } from "react-native";
import axios from "axios";

const ScreenHeight = Dimensions.get("window").height;
const ScreenWidth = Dimensions.get("window").width;

const HomeScreen = ({navigation}) => {

    const [Image_URI, setImage_URI] = useState(false);
    const [mask, setMask] = useState(false);
    const [label, setLabel] = useState(false);
    const [info, setInfo] = useState(false);
    const [loading, setLoading] = useState(false);
  
    useEffect(() => {if(mask!==false){navigation.navigate('Result',{info: info, msk: mask, img: label});}}, [mask,loading]);
      
    const uploadImage = async () => {
      var config = { headers: {'Content-Type': 'application/json'}}

      //Check if any file is selected or not
      if (Image_URI !== false) {
        setLoading(true);
        axios.post("https://b231-94-187-0-34.ngrok.io"+"/send-image/",{selectedImg:Image_URI},  config)
        .then((res) => {
          //console.log(res);
          if (res.status >= 200 && res.status <= 299) {
            setInfo(res.data.info);
            setMask(res.data.maskBase64);
            setLabel(res.data.labelBase64);
          } 
          else {
            alert('Server Error\n Try reuploading the image');
          }
          setLoading(false);
        }, (error) => {
          // no Timeout
          console.log(error);
          alert("Failed to load resource.\n Try reuploading the image");
          setLoading(false);
        });
      }
      else {
        alert('Select an Image First');
      }
    }
  
    return (
    <ImageBackground source= {require('../assets/background.png')}  resizeMode="cover" style={styles.imageBG}>
      <View style={{justifyContent: "space-around",flex:1}}>
        <Br num={2} />
        <Image source={Image_URI===false?require('../assets/transparent.png'):{ uri: Image_URI }} style = {styles.image} />
        <IsVisibleTxt state={loading!==false}></IsVisibleTxt>
        <Br/>
        <View style={(Platform.OS == "ios"|| Platform.OS =="android")?{justifyContent:"flex-end"}:[styles.row,{justifyContent:"center"}]}>
          <TouchableOpacity
            style={styles.btn}
            onPress={()=>{
              setMask(false);
              uploadImage();
            }}
            underlayColor='#fff'>
            <Text style={styles.btnText}>Run Diagnosis</Text>
          </TouchableOpacity>
          <Br/>
          <TouchableOpacity
              style={styles.btn}
              onPress={()=>{
                openImagePicker().then((res)=>{
                  setImage_URI(res.Image_URI);
                });
              }}
              underlayColor='#fff'>
              <Text style={styles.btnText}>Select Image</Text>
          </TouchableOpacity>
          <Br/>
          <IsVisibleBtn navigation={navigation} state={mask!==false} info={info} mask={mask} label={label}></IsVisibleBtn>
        </View>
      </View>
    </ImageBackground>
    );
  }

const styles = StyleSheet.create({
  item: {
    flex: 1,
    backgroundColor: "#D4ECDD",
    padding: 5,
    fontWeight: "700",
    alignItems: "center",
  },
  header: {
    color: "#FAFFAF",
    backgroundColor: "#152D35",
    fontWeight: "700",
    fontSize: 25,
    textAlign: 'center',
    padding: 5,
  },
  image: {
    height: (Platform.OS == "ios"|| Platform.OS =="android")?ScreenHeight*0.4:ScreenHeight*0.7, 
    width: (Platform.OS == "ios"|| Platform.OS =="android")?ScreenWidth*0.7:ScreenWidth*0.4, 
    margin: 5, 
    resizeMode:"cover",
    borderRadius:10 
  },
  imageBG: {
    flex: 1,
    justifyContent: (Platform.OS == "ios"||Platform.OS =="android")? "space-evenly":"space-around",
    alignItems: "center",
    paddingTop: (Platform.OS == "ios"||Platform.OS =="android")? 30:0,
    paddingBottom: (Platform.OS == "ios"||Platform.OS =="android")? 30:10,
  },
  result: {
    justifyContent: "center",
    backgroundColor: "#5584AC",
    alignItems: "center",
  },
  row: {
    flexDirection: 'row'
    },
  text:{
    color:"white",
    fontWeight: "700",
    fontSize: 25,
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

