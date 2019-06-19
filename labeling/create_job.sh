#!/bin/bash
# Creates labeling job

aws sagemaker create-labeling-job --labeling-job-name black6-mouse-42 --label-attribute-name keypoint --input-config DataSource={S3DataSource={ManifestS3Uri=s3://ratception-tolabel/black6_mouse_42/dataset.manifest}} --output-config S3OutputPath=s3://ratception-tolabel/black6_mouse_42/ --role-arn arn:aws:iam::458691804448:role/service-role/AmazonSageMaker-ExecutionRole-20190207T125408 --human-task-config '{ 
  "WorkteamArn": "arn:aws:sagemaker:us-east-1:458691804448:workteam/private-crowd/Mashunn", 
  "UiConfig": { 
    "UiTemplateS3Uri": "s3://ratception-tolabel/black6_mouse_42/mouse.template" 
  }, 
  "PreHumanTaskLambdaArn": "arn:aws:lambda:us-east-1:458691804448:function:SageMaker", 
  "TaskKeywords": ["mouse"], 
  "TaskTitle": "Mouse labeling", 
  "TaskDescription": "Select unoccluded keypoints", 
  "NumberOfHumanWorkersPerDataObject": 1, 
  "TaskTimeLimitInSeconds": 28800, 
  "TaskAvailabilityLifetimeInSeconds": 28800, 
  "MaxConcurrentTaskCount": 1, 
  "AnnotationConsolidationConfig": { 
    "AnnotationConsolidationLambdaArn": "arn:aws:lambda:us-east-1:458691804448:function:SageMaker_out" 
  } 
}'