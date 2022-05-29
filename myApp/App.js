import LoadingScreen from './screens/LoadingScreen/Screen';
import HomeScreen from './screens/HomeScreen/Screen';
import ResultScreen from './screens/ResultScreen/Screen';

import { Provider } from 'react-redux';
import React from 'react';
import { NavigationContainer } from '@react-navigation/native';
import { createNativeStackNavigator } from '@react-navigation/native-stack';
import { store } from './redux/store';
import { StyleSheet, Platform } from "react-native";

// import * as NavigationBar from "expo-navigation-bar";
// import { useEffect, useState } from 'react';

function App() {
  
  const Stack = createNativeStackNavigator();

  // // Hide default bottom navigation bar on android only
  // if (Platform.OS=="android"){
  //   const [barVisibility, setBarVisibility] = useState();
  //   NavigationBar.addVisibilityListener(({ visibility }) => {
  //     setBarVisibility(visibility);
  //   });
  //   useEffect(() => {
  //     NavigationBar.setVisibilityAsync("hidden"); // Hide it
  //   }, [barVisibility]);
  // }

  return (
    <Provider store={store}>
      <NavigationContainer>
        <Stack.Navigator initialRouteName="Loading">
          <Stack.Screen name="Result" component={ResultScreen}   options={{headerTintColor: "white", headerStyle: styles.header}}/>
          <Stack.Screen name="Home" component={HomeScreen}   options={{headerShown: false}}/>
          <Stack.Screen name="Loading" component={LoadingScreen}   options={{headerShown: false}}/>
        </Stack.Navigator>
      </NavigationContainer>
    </Provider>
  );
}

const styles = StyleSheet.create({
  header: {
    backgroundColor: "#22577E",
    borderBottomWidth:0
  }
});

export default App;

