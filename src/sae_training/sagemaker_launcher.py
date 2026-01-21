import boto3
import sagemaker
from sagemaker.pytorch import PyTorch
import os
from dotenv import load_dotenv

load_dotenv()

ROLE = "arn:aws:iam::115801844135:role/service-role/AmazonSageMaker-ExecutionRole-20260114T144694"
BUCKET = "saltandpurple-mllab"
REGION = "eu-central-1"

def launch_training(training_tokens: int = None):
    boto_session = boto3.Session(region_name=REGION)
    sm_session = sagemaker.Session(boto_session=boto_session)

    env = {"WANDB_API_KEY": os.environ.get("WANDB_API_KEY")}
    if training_tokens:
        env["TRAINING_TOKENS"] = str(training_tokens)

    estimator = PyTorch(
        sagemaker_session=sm_session,
        entry_point="train.py",
        source_dir="src/sae_training",
        role=ROLE,
        instance_type="ml.g4dn.xlarge",
        instance_count=1,
        framework_version="2.1",
        py_version="py310",
        output_path=f"s3://{BUCKET}/sae-training/output",
        environment=env,
        max_run=14400,  # 4 hours for 70M tokens
        volume_size=50,
    )

    estimator.fit(wait=True)
    print(f"Training job: {estimator.latest_training_job.name}")
    return estimator

if __name__ == "__main__":
    launch_training()
