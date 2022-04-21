import { Text, View, Platform, StyleSheet, ActivityIndicator } from "react-native";
import React from 'react';

const IsVisibleTxt = ({state})=>{
    if(state){
        return (
        <View style={styles.container}>
            <ActivityIndicator size="small" color="white" /> 
            <Text style={styles.waitingTxt}>Pending result,{(Platform.OS == "ios"||Platform.OS =="android")?"\n\t\t":""} this may take a while...</Text> 
        </View>
    );}
    else{
        return <></>;
    }
}

const styles = StyleSheet.create({
    container:{
        flexDirection: 'row', 
        justifyContent: "center", 
        alignItems: "center", 
        paddingTop: 30, 
        paddingBottom: 5
    },
    waitingTxt:{
        fontStyle: "italic",
        color:'white',
        textAlign:'center',
        paddingLeft : 10,
        paddingRight : 10,
        fontWeight: "400",
        fontSize:(Platform.OS == "ios"|| Platform.OS =="android")?17:21
    }
  });

export default IsVisibleTxt;
