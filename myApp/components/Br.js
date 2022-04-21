import {Text} from "react-native";
import React from 'react';

const Br = (num=1)=>{
    let indentation = "";
    for (let i = 0; i<num; i++) {
        indentation = indentation + "\n";
    }
    return <Text>{indentation}</Text>;
}

export default Br;