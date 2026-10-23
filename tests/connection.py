import boto3

client = boto3.client(
    "bedrock-runtime",
    region_name="eu-west-3"
)

models = {
    "Nova 2 Lite": "eu.amazon.nova-2-lite-v1:0",
    "Nova Lite": "eu.amazon.nova-lite-v1:0",

    "Claude 3 Haiku": "eu.anthropic.claude-3-haiku-20240307-v1:0",
    "Claude 3 Sonnet": "eu.anthropic.claude-3-sonnet-20240229-v1:0",

    "Claude Haiku 4.5": "eu.anthropic.claude-haiku-4-5-20251001-v1:0",
    "Claude Haiku 5.5": "eu.anthropic.claude-haiku-5-5",

    "Claude Sonnet 4.5": "eu.anthropic.claude-sonnet-4-5-20250929-v1:0",
    "Claude Sonnet 4.6": "eu.anthropic.claude-sonnet-4-6",
    "Claude Sonnet 5": "eu.anthropic.claude-sonnet-5",
    "Claude Sonnet 5.5": "eu.anthropic.claude-sonnet-5-5",

    "Claude Opus 4.5": "eu.anthropic.claude-opus-4-5-20251101-v1:0",
    "Claude Opus 4.6": "eu.anthropic.claude-opus-4-6-v1",
    "Claude Opus 4.7": "eu.anthropic.claude-opus-4-7",
    "Claude Opus 4.8": "eu.anthropic.claude-opus-4-8",
    "Claude Opus 5": "eu.anthropic.claude-opus-5",
    "Claude Opus 5.5": "eu.anthropic.claude-opus-5-5",

    "Llama 3.2 1B": "eu.meta.llama3-2-1b-instruct-v1:0",
    "Llama 3.2 3B": "eu.meta.llama3-2-3b-instruct-v1:0",

    "Pixtral Large": "eu.mistral.pixtral-large-2502-v1:0",
}

for name, model_id in models.items():
    try:
        client.converse(
            modelId=model_id,
            messages=[{
                "role": "user",
                "content": [{"text": "OK"}]
            }]
        )

        print(f"OK|{name}|{model_id}")

    except Exception as e:
        print(f"KO|{name}|{model_id}|{type(e).__name__}")