# Blood Diagnosis App

## Block Diagram

![block diagram updated](https://user-images.githubusercontent.com/96151955/162471753-b238b2e5-6fdb-4b1a-b962-7bc3d17341e9.png)

## Project Workflow

The application is built in expo react native to provide cross platform functionality across Android, IOS and Web platforms.
Upon starting the application the user is presented with an animated loading screen. Then they can select a blood smear image from their local file system and choose from the available red blood cell morphologies for diagnosis ( Codocyte, Dacrocyte, Acanthocyte, Stomatoctye, Elliptocyte, Sickle Cell ). Afterwards the user should confirm to upload the image to the server for processing. Once that is done, The application communicates with the REST API hosting the Artificial Intelligence through a secure tunnel provided by Ngrok. The REST API is built in Flask. Once it recieves an image it applies different image preprocessing techniques to identify the appropriate patch size to crop the image into patches suitable for the AI to process. These patches are then passed to the chosen AI models to predict on them. Currently there are six AI models each specific to a certain morphology.  each AI is a binary semantic image segmentation model with Residual and Attention Unet architecture. The models are ensembled to perform as a single multiclass model, this process pipeline enables customization, helps in maintaining high accuracy rates for each class and provides an efficient way for scaling up the system. Prediction time augmentation is applied to get a more accurate prediction. Then the resulting prediction for each patch are reassembled into the full sized image by applying smooth blending. The server then proceeds to extract features from the prediction (Area, Equivalent Diameter, Perimeter, Eccentricity, Orientation and Hu moments up to the sixth order ). Finally the API packs the results in a JSON format and sends it back to the application. Upon receiving the results, the application opens up a new page displaying them. It also provides an option to download the images as PNG image and extracted features as a CSV file to the local file system. Lastly, The application can interact with an arduino system if available via bluetooth. 

## Project outcomes

The diagnosis of blood-based diseases often involves identifying and characterizing patient blood samples. Automated methods to detect and classify blood cell subtypes have important medical applications. Automated biomedical image processing and analysis offers a powerful tool for medical diagnosis. In this work we tackle the problem of abnormal red blood cell classification based on the morphological characteristics of the cell. The work explores a set of preprocessing algorithms and Deep Learning Convolutional Neural Networks with a set of feature extraction techniques that are able to recognize, measure, count and classify the region of interest. The outcome of this work can be used as a framework by medical doctors or researchers for automated blood cell classification and feature extraction, it can aid in the diagnosis and treatment process of related blood diseases. The framework can be accessed from any smart device with access to an internet connection. It is cross-platform and available for use on both mobile and web alike. The framework is lightweight as the heavy processing and AI models are hosted on a RESTful API located on the cloud. It has an easy to learn user friendly interface. It allows customisation and proper storing of the output results on your local file system. Lastly, It can interact with an embededded system with via a bluetooth connection.

## Video Demo
### Find it on Youtube: https://youtu.be/2Ir9hSbU31g

https://user-images.githubusercontent.com/96151955/162437024-caa94880-9417-4198-a6f7-f79c377817d6.mp4

