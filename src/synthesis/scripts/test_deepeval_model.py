from src.settings import settings
from loguru import logger
import os
import asyncio
from .bedrock_model import AmazonBedrockModel

logger.info(
    f"Testing DeepEval model: model {settings.critique_model_name} region {settings.critique_model_region_name}"
)

aws_access_key_id = os.environ["AWS_ACCESS_KEY_ID"]
aws_secret_access_key = os.environ["AWS_SECRET_ACCESS_KEY"]
aws_session_token = os.environ.get("AWS_SESSION_TOKEN")

logger.info(f"AWS Access Key ID: {aws_access_key_id}")
logger.info(f"AWS Secret Access Key: {aws_secret_access_key}")
logger.info(f"AWS Session Token: {'Present' if aws_session_token else 'Not found'}")


def test_model():
    model = AmazonBedrockModel(
        model=settings.critique_model_name,
        region=settings.critique_model_region_name,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        aws_session_token=aws_session_token,
        cost_per_input_token=0.3 * 10**-6,
        cost_per_output_token=2.5 * 10**-6,
        generation_kwargs={"max_tokens": 3000, "temperature": 0.1, "top_p": 0.9},
    )

    try:
        response = model.generate(
            "What is the capital of France?",
        )
        logger.info(f"Response: {response}")
    except Exception as e:
        logger.error(f"Error: {e}")
        return None
    finally:
        asyncio.run(model.close())


if __name__ == "__main__":
    test_model()
