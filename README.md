# Data Clinic demo v1

First working version of data clinic integrated with the compute service.
To run this demo, start the followings services (in the order):  
-	mlex\_computing\_api
-  mlex\_dash\_segmentation\_demo

Then build the images of the models inside the folder `models` using the command `make build_docker`.

It supports asynchronous job submissions and results showing: choose which (completed) training results from the list to use for testing. For testing, choose the results you'd like to visualize and the reconstructed image will be shown in the frontend of the app.

Notebook examples: model/notebooks
