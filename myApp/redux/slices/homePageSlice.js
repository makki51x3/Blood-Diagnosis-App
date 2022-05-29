import { createSlice } from "@reduxjs/toolkit";
  
  const initialState = {
    loading: false,
    image: ""
  };

  export const homePageSlice = createSlice({
    name: "homePage",
    initialState,
    reducers: {
      updateLoading:(state, action) => {
        state.loading=action.payload;
      },
      updateImage:(state, action) => {
        state.image=action.payload;
      },
    },
  });
  
  export const { 
    updateLoading, 
    updateImage } = homePageSlice.actions;
  
  export default homePageSlice.reducer;