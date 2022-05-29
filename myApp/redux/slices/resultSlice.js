import { createSlice } from "@reduxjs/toolkit";
  
  const initialState = {
    mask: "",
    label: "",
    info: ""
  };

  export const ResultSlice = createSlice({
    name: "Result",
    initialState,
    reducers: {
      updateMask:(state, action) => {
        state.mask=action.payload;
      },
      updateLabel:(state, action) => {
        state.label=action.payload;
      },
      updateInfo:(state, action) => {
        state.info=action.payload;
      }
    },
  });
  
  export const { 
    updateMask, 
    updateLabel,
    updateInfo } = ResultSlice.actions;
  
  export default ResultSlice.reducer;