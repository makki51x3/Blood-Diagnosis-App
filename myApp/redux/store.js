import { configureStore} from "@reduxjs/toolkit";
// import authenticationReducer from "./slices/authenticationSlice";
// import articlesReducer from "./slices/articlesSlice";
import loadingPageReducer from "./slices/loadingPage";
// import dashBoardPageReducer from "./slices/dashBoardPageSlice";
// import speechReducer from "./slices/speechSlice";
// import searchReducer from "./slices/searchSlice";

export const store = configureStore({
    reducer:{
        // authenticationReducer,
        // articlesReducer,
        loadingPageReducer,
        // dashBoardPageReducer,
        // speechReducer,
        // searchReducer,
    },
    // middleware: (getDefaultMiddleware) => getDefaultMiddleware({
    //     immutableCheck: false,
    //     serializableCheck: false,
    //   }) // disable middleware warnings
});
