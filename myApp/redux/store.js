import { configureStore} from "@reduxjs/toolkit";

import loadingPageReducer from "./slices/loadingPageSlice";
import resultReducer from "./slices/resultSlice";
import homePageReducer from "./slices/homePageSlice";
import resultPageReducer from "./slices/resultPageSlice";

export const store = configureStore({
    reducer:{
        resultReducer,
        loadingPageReducer,
        homePageReducer,
        resultPageReducer
    },
    // middleware: (getDefaultMiddleware) => getDefaultMiddleware({
    //     immutableCheck: false,
    //     serializableCheck: false,
    //   }) // disable middleware warnings
});
