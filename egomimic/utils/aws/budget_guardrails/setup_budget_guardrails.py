import argparse
import io
import json
import time
import zipfile

import boto3
from botocore.exceptions import ClientError


def _zip_lambda_source(lambda_path):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(lambda_path, arcname="stop_ec2_lambda.py")
    buffer.seek(0)
    return buffer.read()


def _ensure_role(iam, role_name):
    try:
        return iam.get_role(RoleName=role_name)["Role"]["Arn"]
    except ClientError as exc:
        if exc.response["Error"]["Code"] != "NoSuchEntity":
            raise

    trust_policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {"Service": "lambda.amazonaws.com"},
                "Action": "sts:AssumeRole",
            }
        ],
    }
    role = iam.create_role(
        RoleName=role_name,
        AssumeRolePolicyDocument=json.dumps(trust_policy),
        Description="Stops EC2 instances when budget threshold is hit.",
    )

    iam.attach_role_policy(
        RoleName=role_name,
        PolicyArn="arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole",
    )

    policy_doc = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Action": ["ec2:DescribeInstances", "ec2:StopInstances"],
                "Resource": "*",
            }
        ],
    }
    iam.put_role_policy(
        RoleName=role_name,
        PolicyName="BudgetGuardrailsStopEc2",
        PolicyDocument=json.dumps(policy_doc),
    )

    # IAM propagation
    time.sleep(10)
    return role["Role"]["Arn"]


def _ensure_lambda(lambda_client, lambda_name, role_arn, region, env, zip_bytes):
    try:
        lambda_client.get_function(FunctionName=lambda_name)
        lambda_client.update_function_code(
            FunctionName=lambda_name, ZipFile=zip_bytes, Publish=True
        )
        lambda_client.update_function_configuration(
            FunctionName=lambda_name,
            Runtime="python3.11",
            Handler="stop_ec2_lambda.lambda_handler",
            Role=role_arn,
            Timeout=60,
            Environment={"Variables": env},
        )
        return lambda_client.get_function(FunctionName=lambda_name)["Configuration"]["FunctionArn"]
    except ClientError as exc:
        if exc.response["Error"]["Code"] != "ResourceNotFoundException":
            raise

    response = lambda_client.create_function(
        FunctionName=lambda_name,
        Runtime="python3.11",
        Role=role_arn,
        Handler="stop_ec2_lambda.lambda_handler",
        Code={"ZipFile": zip_bytes},
        Timeout=60,
        Environment={"Variables": env},
        Publish=True,
    )
    return response["FunctionArn"]


def _ensure_sns_topic(sns, topic_name, emails):
    topic_arn = sns.create_topic(Name=topic_name)["TopicArn"]
    for email in emails:
        sns.subscribe(TopicArn=topic_arn, Protocol="email", Endpoint=email)
    return topic_arn


def _ensure_sns_lambda_subscription(sns, topic_arn, lambda_arn):
    existing = sns.list_subscriptions_by_topic(TopicArn=topic_arn).get(
        "Subscriptions", []
    )
    for sub in existing:
        if sub.get("Endpoint") == lambda_arn and sub.get("Protocol") == "lambda":
            return sub.get("SubscriptionArn")
    return sns.subscribe(TopicArn=topic_arn, Protocol="lambda", Endpoint=lambda_arn)[
        "SubscriptionArn"
    ]


def _allow_sns_invoke(lambda_client, lambda_name, topic_arn):
    statement_id = "AllowSnsInvokeBudgetGuardrails"
    try:
        lambda_client.add_permission(
            FunctionName=lambda_name,
            StatementId=statement_id,
            Action="lambda:InvokeFunction",
            Principal="sns.amazonaws.com",
            SourceArn=topic_arn,
        )
    except ClientError as exc:
        if exc.response["Error"]["Code"] != "ResourceConflictException":
            raise


def _ensure_budget(budgets, account_id, budget_name, limit, topic_arn, service_filter):
    budget = {
        "BudgetName": budget_name,
        "BudgetLimit": {"Amount": str(limit), "Unit": "USD"},
        "BudgetType": "COST",
        "TimeUnit": "DAILY",
        "CostFilters": {"Service": [service_filter]} if service_filter else {},
    }
    notification = {
        "NotificationType": "ACTUAL",
        "ComparisonOperator": "GREATER_THAN",
        "Threshold": float(limit),
        "ThresholdType": "ABSOLUTE_VALUE",
    }
    subscribers = [{"SubscriptionType": "SNS", "Address": topic_arn}]

    try:
        budgets.create_budget(
            AccountId=account_id,
            Budget=budget,
            NotificationsWithSubscribers=[
                {"Notification": notification, "Subscribers": subscribers}
            ],
        )
        return
    except ClientError as exc:
        if exc.response["Error"]["Code"] != "DuplicateRecordException":
            raise

    budgets.update_budget(AccountId=account_id, NewBudget=budget)
    budgets.update_notification(
        AccountId=account_id, BudgetName=budget_name, OldNotification=notification, NewNotification=notification
    )
    budgets.update_subscriber(
        AccountId=account_id,
        BudgetName=budget_name,
        Notification=notification,
        OldSubscriber=subscribers[0],
        NewSubscriber=subscribers[0],
    )


# Run: python3 setup_budget_guardrails.py --emails skareer6@gatech.edu rpunamiya6@gatech.edu danfei@gatech.edu
def main():
    parser = argparse.ArgumentParser(description="Set up EC2 daily budget guardrails.")
    parser.add_argument("--region", default="us-east-2", help="EC2 region to stop.")
    parser.add_argument("--daily-limit", type=float, default=150.0)
    parser.add_argument("--emails", nargs="+", required=True)
    parser.add_argument("--budget-name", default="ec2-daily-budget-guardrails")
    parser.add_argument("--topic-name", default="ec2-daily-budget-alerts")
    parser.add_argument("--lambda-name", default="stop-ec2-on-budget")
    parser.add_argument("--role-name", default="budget-guardrails-stop-ec2")
    parser.add_argument("--stop-tag-key", default="")
    parser.add_argument("--stop-tag-value", default="")
    parser.add_argument(
        "--service-filter",
        default="Amazon Elastic Compute Cloud - Compute",
        help="Cost Explorer service filter; empty to disable.",
    )
    parser.add_argument(
        "--lambda-path",
        default="egomimic/utils/aws/budget_guardrails/stop_ec2_lambda.py",
    )
    args = parser.parse_args()

    sts = boto3.client("sts")
    account_id = sts.get_caller_identity()["Account"]

    iam = boto3.client("iam")
    lambda_client = boto3.client("lambda", region_name=args.region)
    sns = boto3.client("sns", region_name=args.region)
    budgets = boto3.client("budgets", region_name="us-east-1")

    role_arn = _ensure_role(iam, args.role_name)
    zip_bytes = _zip_lambda_source(args.lambda_path)
    env = {
        "REGION": args.region,
        "STOP_TAG_KEY": args.stop_tag_key,
        "STOP_TAG_VALUE": args.stop_tag_value,
    }
    lambda_arn = _ensure_lambda(
        lambda_client, args.lambda_name, role_arn, args.region, env, zip_bytes
    )
    topic_arn = _ensure_sns_topic(sns, args.topic_name, args.emails)
    _allow_sns_invoke(lambda_client, args.lambda_name, topic_arn)
    _ensure_sns_lambda_subscription(sns, topic_arn, lambda_arn)
    _ensure_budget(
        budgets,
        account_id,
        args.budget_name,
        args.daily_limit,
        topic_arn,
        args.service_filter if args.service_filter else "",
    )

    print("Budget guardrails configured.")
    print(f"Account: {account_id}")
    print(f"Region: {args.region}")
    print(f"SNS topic: {topic_arn}")
    print(f"Lambda: {lambda_arn}")


if __name__ == "__main__":
    main()
