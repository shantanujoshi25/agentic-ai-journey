Session 1 (Day 1):
- AWS account hardened (MFA on root + IAM user)
- Budget alerts set at $25 / $100
- Bedrock model access enabled in us-east-1
- Anthropic use case form submitted
- Models confirmed available: Claude Sonnet 4.5, Haiku 4.5, Nova Micro, Titan Embed V2
Stuck on: nothing yet


Session 2 (Day 2):
- boto3 installed, .env and .gitignore done
- aws sts get-caller-identity confirms identity
- 01_hello_bedrock.py runs against Claude Sonnet 4.5 via us. inference profile
- ~139 tokens used, will check cost dashboard tomorrow