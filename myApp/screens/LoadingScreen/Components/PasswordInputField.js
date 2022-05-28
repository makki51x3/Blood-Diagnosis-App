import { TextInput } from 'react-native-paper';
import {StyleSheet} from 'react-native';
import {updatePasswordVisible} from "../../../redux/slices/loginPageSlice"
import { updatePassword } from "../../../redux/slices/authenticationSlice"
import { useSelector, useDispatch } from "react-redux";


export const PasswordInputField = ()=>{

    // Get data from the redux store
    const dispatch = useDispatch();
    const passwordVisible =  useSelector((state) => state.loginPageReducer.passwordVisible);
  
    return(
        <TextInput
            onChangeText={(pass) => {dispatch(updatePassword(pass))}}
            placeholder={"Password"}
            secureTextEntry={!passwordVisible}
            style={styles.input}
            right={
            <TextInput.Icon 
                style={styles.icon} 
                name={passwordVisible ? "eye-off":"eye"} 
                size={20} 
                onPress={() => dispatch(updatePasswordVisible(!passwordVisible))} 
            />}
      />
    );
}

const styles = StyleSheet.create({
    icon: {
        marginHorizontal:"auto", 
        backgroundColor:"#e8f0fe",    
    },
    input: {
        width: 200,
        height: 40,
        marginBottom: 20,
        backgroundColor:  "#e8f0fe"
    },
});

export default PasswordInputField;
