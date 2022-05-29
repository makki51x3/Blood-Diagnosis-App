import {StyleSheet, TouchableOpacity, Text} from "react-native";
import React from 'react';

const IsVisibleBtn = ({navigation, state}) => {

    if(state){
      return (
      <TouchableOpacity
      style={styles.btn}
      onPress={()=>{navigation.navigate('Result');}}
      underlayColor='#fff'>
        <Text style={styles.btnText}>Results</Text>
      </TouchableOpacity>
      );
    }
    else{
      return <></>;
    }
  }

const styles = StyleSheet.create({
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

export default IsVisibleBtn;
