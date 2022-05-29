import { createSlice } from "@reduxjs/toolkit";
  
  const initialState = {
    appIsReady: false,
    listOfTexts: [
      "Preparing Tests",
      "Cleaning Lab Equipment",
      "Calling Dr. AI", 
      "Sanitizing the Microscope",
      "Ironing the Lab Coat",
      "Putting on Safety Goggles",
      "Wearing Gloves"
    ],
    loadingText: "Preparing Tests"
  };

  export const loadingPageSlice = createSlice({
    name: "loadingPage",
    initialState,
    reducers: {
      updateLoadingText:(state, action) => {
        state.loadingText=action.payload;
      },
      updateListOfTexts:(state, action) => {
        state.listOfTexts=action.payload;
      },
      updateAppIsReady:(state, action) => {
        state.appIsReady=action.payload;
      }
    },
  });
  
  export const { 
    updateLoadingText, 
    updateAppIsReady } = loadingPageSlice.actions;
  
  export default loadingPageSlice.reducer;