# waste_mgmt


This project is to train a classifier for Recyclable VS Organic waste.

URL: https://www.kaggle.com/datasets/techsash/waste-classification-data/code

# Overview
This Project uses AWS and streamlit to create a demo on hosting CNN models on cloud. The model weights are saved on Amazon S3 and retrieved by the dockerized container and used to predict if images are organic or recyclable. 

## Step 1 
Run the jupyter notebook once you have downloaded the dataset. Two models are provided, a self defined CNN and VGG16. The model weights are then saved locally. 

## Step 2
Set up an AWS account and an S3 bucket, and an IAM policy that gives the access to read/write into the specific bucket. Create an IAM role with this policy access and create an access and secret token. Save them in a ```.env``` file 
```commandline
AWS_ACCESS_KEY_ID=your_access_key_id
AWS_SECRET_ACCESS_KEY=your_secret_key_id
```
Use the access key to save the model weights into the S3 bucket. 
## Step 3
Dockerize the entire application using ```docker compose up --build``` Test if the image is working locally. ```clf_test.py``` retrieves self defined CNN model weights from S3, changing to ```clf.py``` is to use the model weights locally. 

## Step 4
Download and install AWS CLI. Configure the credentials (I just used the same IAM role with same access key and secret key, but give an additional permission to read and write to ECR), upload the image to ECR. 

## Step  5
Use Amazon ECS to run this image. Create a cluster and a task, give a TCP rule that port 8501 can connect to all incoming connection (0.0.0.0), and choose the option of auto generating a public IP (in subnet option or if using fargate can choose this directly).
Add the permission to retrieve the same S3 bucket for your cluster/task, and run the task in the cluster. Access your website by going to the public IP:8501

# Additional notes
## Install ipykernel (add conda env to jupyter notebook)
```
conda install -c anaconda ipykernel
```
```
python -m ipykernel install --user --name={name_of_env}
```


# To run streamlit 
```
steamlit run stream_main.py
```

