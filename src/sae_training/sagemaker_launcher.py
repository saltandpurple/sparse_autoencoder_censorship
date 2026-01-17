import sagemaker
from sagemaker.pytorch import PyTorch
import os
from dotenv import load_dotenv

load_dotenv()

ROLE = "arn:aws:iam::115801844135:role/service-role/AmazonSageMaker-ExecutionRole-20260114T144694"
BUCKET = "saltandpurple-mllab"

def launch_training():
    session = sagemaker.Session()

    estimator = PyTorch(
        entry_point="sagemaker_train.py",
        source_dir="src/sae_training",
        role=ROLE,
        instance_type="ml.g4dn.xlarge",
        instance_count=1,
        framework_version="2.1",
        py_version="py310",
        output_path=f"s3://{BUCKET}/sae-training/output",
        environment={
            "WANDB_API_KEY": os.environ.get("WANDB_API_KEY"),
        },
        max_run=7200,  # 2 hours
        volume_size=50,
    )

    estimator.fit(wait=True)
    print(f"Training job: {estimator.latest_training_job.name}")
    return estimator

if __name__ == "__main__":
    launch_training()
