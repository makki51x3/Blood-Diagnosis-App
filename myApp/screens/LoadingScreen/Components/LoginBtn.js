import { ActivityIndicator, StyleSheet, View, TouchableOpacity, Text } from 'react-native';
import {useSelector,useDispatch} from "react-redux";
import {handleLogin} from "../handlers/handleLogin"

export const LoginBtn = ({navigation})=>{

    // Get data from the redux store
    const dispatch = useDispatch();

    // Log In Page Reducer
    const loginFailed = useSelector((state) => state.loginPageReducer.loginFailed);
    const loading = useSelector((state) => state.loginPageReducer.loading);

    // Authentication Reducer
    const userName = useSelector((state) => state.authenticationReducer.credentials.user);
    const password = useSelector((state) => state.authenticationReducer.credentials.pass);
    const warningText = useSelector((state) => state.authenticationReducer.warningText);

    return (
        <View>
            {  // upon failed login display warning text
              loginFailed && <Text style={styles.warning}>{warningText}</Text>
            }

            <View style={{flexDirection:"row",justifyContent: "center"}}>
                <TouchableOpacity
                    // disable button if loading or user name and password are empty
                    disabled={userName=="" || password=="" || loading}
                    style={styles.btn}
                    onPress={()=>{ handleLogin( navigation, dispatch, userName, password )}}
                    underlayColor='#fff'>
                    <Text style={styles.text}>Login</Text>
                </TouchableOpacity>

                {  // display spinner while loading
                    loading &&
                    <ActivityIndicator 
                    size="small" 
                    color="white" 
                    style={styles.spinner} 
                    />
                }
            </View>
        </View>
    )
}

// Style sheet for login button component
const styles = StyleSheet.create({
    warning:{
      color:"red", 
      fontWeight:"700"
    },
    spinner:{
      marginLeft:15,
    },
    btn:{
      opacity:1,
      marginVertical:10,
      paddingVertical:7,
      paddingHorizontal:10,
      backgroundColor:"rgb(31, 20, 99)",
      borderRadius:10,
      borderWidth: 1,
      borderColor: 'black'
    },
    text:{
      color:"white",
      fontSize: 14,
      textAlign: 'center',
    },
  });

export default LoginBtn; 