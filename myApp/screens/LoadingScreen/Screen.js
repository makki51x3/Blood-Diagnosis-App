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

    // Animation variables
    const Fade= new Animated.Value(1);

    useEffect(() => { // on mount start an exit timeout of 7 seconds
      setTimeout(
        () => {
        dispatch(updateAppIsReady(true));
        navigation.navigate("Home");
        }, 7430
      );
    }, []);

    useEffect(() => {  // Animate text fade in and fade out then change it randomly every 3 seconds
      if(!appIsReady){
        const textInterval = setTimeout(
          () => {
            const num = Math.floor((Math.random()*100)) % (listOfTexts.length-1);
            dispatch(updateLoadingText(listOfTexts[num]));
          }, 3000
        );

        Animated.sequence(
          [
            Animated.timing(Fade,{ // Fade Out
              toValue:1,
              duration:1000, 
              useNativeDriver: true 
            }), 
            Animated.timing(Fade,{ // Fade In
              delay:700,
              toValue:0,
              duration:1000, 
              useNativeDriver: true 
            }), 
          ]).start();

        return () => {
          clearInterval(textInterval);  // Clearing the timeout to use it again
        }
      }
    }, [Fade]);
  
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