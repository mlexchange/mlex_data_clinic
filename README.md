# Data Clinic demo v1

First working version of data clinic integrated with the compute service.
To run this demo, start the followings services (in the order):  
-	mlex\_computing\_api
- mlex\_data\_clinic

Note: Modify the data directories in the docker compose file. If your file corresponds to an "npz" file, make sure to add it's name in src/frontend.py

Then build the images of the models inside the folder `models` using the command `make build_docker`.

It supports asynchronous job submissions and results showing: choose which (completed) training results from the list to use for testing. For testing, choose the results you'd like to visualize and the reconstructed image will be shown in the frontend of the app.

Notebook examples: model/notebooks
