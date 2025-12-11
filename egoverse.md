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

To test that this works, you can run.
```
aws s3 mv example.txt s3://rldb/raw/example<yourname>.txt 
```

Our standard will be
for objInBowl, the vrs should be named
`<lab-name>-scene<id>-<toy-color>toy-<bowl-color>bowl-<number>.vrs`

for smallShirtFold, the vrs should be named
`<lab-name>-scene<id>-<shirt-color>shirt-<number>.vrs`

Task Name list
- objInBowl
- smallShirtFold

Lab Name List
- rl2
- wang
- song
- eth
