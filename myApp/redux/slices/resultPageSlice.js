import { createSlice } from "@reduxjs/toolkit";
  
  const initialState = {
    isShownImg: false,
    isShownMsk: false,
  };

  export const resultPageSlice = createSlice({
    name: "resultPage",
    initialState,
    reducers: {
      updateIsShownImg:(state, action) => {
        state.isShownImg=action.payload;
      },
      updateIsShownMsk:(state, action) => {
        state.isShownMsk=action.payload;
      },
    },
  });
  
  export const { 
    updateIsShownImg, 
    updateIsShownMsk } = resultPageSlice.actions;
  
  export default resultPageSlice.reducer;