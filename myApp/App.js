import LoadingScreen from './screens/LoadingScreen';
import HomeScreen from './screens/HomeScreen';
import ResultScreen from './screens/ResultScreen';

import React from 'react';
import { NavigationContainer } from '@react-navigation/native';
import { createNativeStackNavigator } from '@react-navigation/native-stack';

const Stack = createNativeStackNavigator();

function App() {
  return (
    <NavigationContainer>
      <Stack.Navigator initialRouteName="Loading">
         <Stack.Screen name="Result" component={ResultScreen}   options={{headerShown: true, headerTintColor: "white", headerStyle: {backgroundColor: "#22577E",borderBottomWidth:0}}}/>
         <Stack.Screen name="Home" component={HomeScreen}   options={{headerShown: false}}/>
         <Stack.Screen name="Loading" component={LoadingScreen}   options={{headerShown: false}}/>
      </Stack.Navigator>
    </NavigationContainer>
  );
}
export default App;