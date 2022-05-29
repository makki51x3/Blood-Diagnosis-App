
import React from 'react';
import { Dimensions, StyleSheet, TouchableOpacity, Text, ImageBackground,  View, ScrollView, Platform, Button } from "react-native";

import * as FileSystem from 'expo-file-system';
import * as MediaLibrary from 'expo-media-library';
import { useSelector, useDispatch } from "react-redux";
import { updateIsShownImg, updateIsShownMsk } from "../../redux/slices/resultPageSlice";

const ScreenHeight = Dimensions.get("window").height;
const ScreenWidth = Dimensions.get("window").width;


const ResultScreen = () => {

    // Get data from the redux store
    const dispatch = useDispatch();

    // Result Page Slice
    const isShownImg = useSelector((state) => state.resultPageReducer.isShownImg);
    const isShownMsk = useSelector((state) => state.resultPageReducer.isShownMsk);

    // Results Slice Reducer
    const msk = useSelector((state) => state.resultReducer.mask);
    const img = useSelector((state) => state.resultReducer.label);
    const info = useSelector((state) => state.resultReducer.info);
    
    async function downloadFile(uri, filename=Date.now().toString()){ //if not specified default filename is a timestamp
      const fileUri= `${FileSystem.documentDirectory}${filename}.png`; // destination
      const Base64Code = uri.split("base64,");
      FileSystem.writeAsStringAsync(fileUri, Base64Code[1],  { encoding:FileSystem.EncodingType.Base64 })
      .then(
        (res) => {console.log("File: ", res);}, 
        (error) => {console.log("error of write as string async:\t",error);}
        );
      let perm = await  MediaLibrary.requestPermissionsAsync();
      if (perm.granted == false) {
        alert("Permission to access camera roll is required!");
        return;
      }
      try {
        const asset = await MediaLibrary.createAssetAsync(fileUri);
        const album = await MediaLibrary.getAlbumAsync('Results of Diagnosis');
        if (album == null) {
          await MediaLibrary.createAlbumAsync('Results of Diagnosis', asset, false);
        } else {
          await MediaLibrary.addAssetsToAlbumAsync([asset], album, false);
        }
        alert("Download Successful!");
      } catch (e) {
        console.log("error e:\n",e);
      }
    }
  
    function ExtractedInfo(){
      var df = JSON.parse(info);
      var result = [];
      var index = 0;
      for (var feature in df){
        var items = [];
        var indx = 0;
        for(var element in df[feature]){
          indx = items.push(
            <Text key ={indx}  style={styles.item}>{indx+1}{') '}{df[feature][element]}</Text> 
          );          
        }
        indx = indx + 1;
        index = result.push(
          <View key={index} style={styles.sectionList}>
            <Text key ={indx} style={styles.header}>{feature.charAt(0).toUpperCase() + feature.slice(1)}</Text>
            {items}
          </View>
        );
      }
      return <View>{result}</View>;
    }
    
    return (
      <ScrollView contentContainerStyle={styles.result}>
        <View style={(Platform.OS == "ios"|| Platform.OS =="android")?null:styles.row}>
          <TouchableOpacity onPress={()=>{dispatch(updateIsShownImg(!isShownImg));}}>
            <ImageBackground source={{ uri: img }} style = {styles.image}>
              {isShownImg && (
              <View style={styles.downloadContainer}>
                <Text style={[styles.text,{opacity:1}]}>Download Image{"\n\n"}</Text>
                <Button title="download" color='#112031' onPress={()=>{(Platform.OS == "ios"|| Platform.OS =="android")?downloadFile(img,"blood_smear_image"):alert("Downloading...");}} />
              </View> )}
            </ImageBackground>
          </TouchableOpacity>
          <TouchableOpacity  onPress={()=>{dispatch(updateIsShownMsk(!isShownMsk));}}>
            <ImageBackground source={{uri: msk}} style = {styles.image} >
              {isShownMsk && (
              <View style={styles.downloadContainer}>
                <Text style={[styles.text,{opacity:1}]}>Download Mask{"\n\n"}</Text>
                <Button title="download" color='#112031' onPress={()=>{(Platform.OS == "ios"|| Platform.OS =="android")?downloadFile(msk,"labeled_mask"):alert("Downloading...");}} />
              </View> )
              }
            </ImageBackground>
          </TouchableOpacity>
        </View>
        <View style={{justifyContent:"flex-end"}}>
          <ExtractedInfo/>
        </View>
      </ScrollView>
    );
  }

const styles = StyleSheet.create({
    downloadContainer: {
      justifyContent: "space-around",
      position: 'absolute', 
      top: 0, 
      left: 0, 
      right: 0, 
      bottom: 0, 
      justifyContent: 'center', 
      alignItems: 'center', 
      backgroundColor:'rgba(0,0,0,0.7)', 
    },
    sectionList: {
      width: (Platform.OS == "ios"|| Platform.OS =="android")?ScreenWidth*0.7:ScreenWidth*0.3, 
      backgroundColor: "#E5EFC1",
      margin:15,
    },
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
    container: {
      flex: 1,
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
    waitingTxt:{
      fontStyle: "italic",
      color:'white',
      textAlign:'center',
      paddingLeft : 10,
      paddingRight : 10,
      fontWeight: "400",
      fontSize:(Platform.OS == "ios"|| Platform.OS =="android")?17:21
  },
    btnText:{
      color:'white',
      textAlign:'center',
      paddingLeft : 10,
      paddingRight : 10,
      fontWeight: "500",
  }
});

export default ResultScreen;
