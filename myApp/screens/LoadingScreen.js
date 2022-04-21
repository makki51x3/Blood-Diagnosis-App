import { useState, useEffect } from 'react';
import { ImageBackground,  StyleSheet, Platform, ActivityIndicator, Animated} from "react-native";
import React from 'react';

const LoadingScreen = ({navigation})=>{
    const [appIsReady, setAppIsReady] = useState(false);
    const listOfTexts = ["Preparing Tests","Cleaning Lab Equipment","Calling Dr. AI", "Sanitizing the Microscope","Ironing the Lab Coat","Putting on Safety Goggles","Wearing Gloves"];
    const [loadingText, setLoadingText] = useState(listOfTexts[0]);
    const Fade= new Animated.Value(1);
  
    setInterval(() => {
      setAppIsReady(true);
    }, 7000);
  
    useEffect(() => {if(appIsReady==true){navigation.navigate('Home');}}, [appIsReady]);
  
    useEffect(() => {
      if(appIsReady==false){
        const interval = setInterval(() => {
          setLoadingText(listOfTexts[Math.floor(Math.random() * (listOfTexts.length-1))])
          }, 3000);
        Animated.sequence(
          [
            Animated.timing(Fade,{
              toValue:1,
              duration:1000, 
              useNativeDriver: true 
            }), //FadeOut
            Animated.timing(Fade,{
              delay:700,
              toValue:0,
              duration:1000, 
              useNativeDriver: true 
            }), //FadeIn
          ]).start();
          return () => clearInterval(interval);
      }
    }, [Fade]);
  
    return (
    <ImageBackground source= {require('../assets/background.png')}  resizeMode="cover" style={styles.loadingContainer}>
        < Animated.Text style={[ styles.text,{opacity: Fade} ]}>{loadingText}</Animated.Text>
        <ActivityIndicator size="large" color="white" /> 
    </ImageBackground>
    );
  }
  
const styles = StyleSheet.create({
    loadingContainer: {
      flex: 1,
      justifyContent: (Platform.OS == "ios"||Platform.OS =="android")? "space-evenly":"space-around",
      alignItems: "center",
      paddingTop: (Platform.OS == "ios"||Platform.OS =="android")? 30:0,
      paddingBottom: (Platform.OS == "ios"||Platform.OS =="android")? 30:10,
    },
    text:{
      color:"white",
      fontWeight: "700",
      fontSize: 25,
    }
  });

  export default LoadingScreen;