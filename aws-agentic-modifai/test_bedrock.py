import boto3

bedrock = boto3.client(
    "bedrock-runtime",
    region_name="ap-south-1"
)

response = bedrock.converse(
    modelId="apac.amazon.nova-micro-v1:0",
    messages=[
        {
            "role": "user",
            "content": [{"text": "Hello"}]
        }
    ],
    inferenceConfig={
        "maxTokens": 50
    }
)

print(response["output"]["message"]["content"][0]["text"])