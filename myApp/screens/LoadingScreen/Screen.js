import { useEffect } from 'react';
import { useSelector, useDispatch } from "react-redux";
import { ImageBackground, View, StyleSheet, Platform, ActivityIndicator, Animated} from "react-native";
import React from 'react';
import { updateLoadingText, updateAppIsReady } from "../../redux/slices/loadingPageSlice";

const LoadingScreen = ({navigation})=>{

    // Get data from the redux store
    const dispatch = useDispatch();

    // Loading Page Reducer
    const appIsReady = useSelector((state) => state.loadingPageReducer.appIsReady);
    const listOfTexts = useSelector((state) => state.loadingPageReducer.listOfTexts);
    const loadingText = useSelector((state) => state.loadingPageReducer.loadingText);

    // animation variables
    const Fade= new Animated.Value(1);

    useEffect(() => {
      setTimeout(
        () => {
        dispatch(updateAppIsReady(true));
        }, 7000
      );
    }, []);

    useEffect(() => {
      if(appIsReady){
        navigation.navigate("Home");
      }
    }, [appIsReady]);

    useEffect(() => {
      const textInterval = setTimeout(
        () => {
          dispatch(updateLoadingText(listOfTexts[Math.floor(Math.random() * (listOfTexts.length-1))]));
          }, 3000
      );

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
        return () => {
          clearInterval(textInterval);
        }
    }, [loadingText]);
  
    return (
    <ImageBackground source= {require('../../assets/background.png')}  resizeMode="cover" style={styles.AppContainer}>
        <View style={styles.loadingContainer}>
          <Animated.Text style={[ styles.text,{opacity: Fade} ]}>{loadingText}</Animated.Text>
          <ActivityIndicator size="large" color="white" /> 
        </View>
    </ImageBackground>
    );
  }
  
const styles = StyleSheet.create({
    AppContainer: {
      flex: 1,
      // paddingTop: (Platform.OS == "ios"||Platform.OS =="android")? 30:0,
      // paddingBottom: (Platform.OS == "ios"||Platform.OS =="android")? 30:0,
    },
    text:{
      color:"white",
      fontWeight: "700",
      fontSize: 25,
    },
    loadingContainer:{
      width: "100%", 
      flex:1,
      backgroundColor:"rgba(0, 0, 0,0.1)",
      justifyContent: (Platform.OS == "ios"||Platform.OS =="android")? "space-evenly":"space-around",
      alignItems: "center",
    },
  });

  export default LoadingScreen;