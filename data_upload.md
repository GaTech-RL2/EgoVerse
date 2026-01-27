# S3 setup
Download AWS CLI

```
curl "https://awscli.amazonaws.com/AWSCLIV2.pkg" -o "AWSCLIV2.pkg"
sudo installer -pkg AWSCLIV2.pkg -target /
```

Configure the console.  First we need some info
1. Go to https://aws.amazon.com/ and click log in
2. Our account id is aws-scs-d2i.  Fill in the username and pw Simar gave you.
3. Click IAM -> your username -> security credentials -> create access key.  This will give both `Access Key ID` and `Secret Access Key`

Then you can run
```
aws configure
AWS Access Key ID [None]: <fill this>
AWS Secret Access Key [None]: <fill this>
Default region name [None]: us-east-2
Default output format [None]:
```

# Data Uploading
Embodiment specific uploaders inherit from [``abstract_upload.py``](./egomimic/scripts/abstract_upload.py), which handles the key logic of generating an episode hash + collecting user specified metadata, then uploading to aws.  We have embodiemnt specific uploaders which inherit from this
- [``eva_uploader.py``](./egomimic/scripts/eva_uploader.py)
- [``aria_uploader.py``](./egomimic/scripts/aria_uploader.py)


Lab Name List
- rl2
- wang
- song
- eth
