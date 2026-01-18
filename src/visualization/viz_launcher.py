import sagemaker
from sagemaker.pytorch import PyTorch
import os
from dotenv import load_dotenv

load_dotenv()

ROLE = "arn:aws:iam::115801844135:role/service-role/AmazonSageMaker-ExecutionRole-20260114T144694"
BUCKET = "saltandpurple-mllab"
# SAE model from training job
SAE_MODEL_PATH = "s3://saltandpurple-mllab/sae-training/output/pytorch-training-2026-01-17-15-57-09-245/output/model.tar.gz"

def launch_visualization():
    wandb_api_key = os.environ.get("WANDB_API_KEY")
    if not wandb_api_key:
        raise ValueError("WANDB_API_KEY environment variable must be set")

    estimator = PyTorch(
        entry_point="sagemaker_viz.py",
        source_dir="src/visualization",
        role=ROLE,
        instance_type="ml.g5.xlarge",
        instance_count=1,
        framework_version="2.1",
        py_version="py310",
        output_path=f"s3://{BUCKET}/sae-visualization/output",
        environment={
            "WANDB_API_KEY": wandb_api_key,
        },
        max_run=7200,  # 2 hours
        volume_size=50,
    )

    # Pass the trained SAE model as input
    estimator.fit(
        inputs={"sae": SAE_MODEL_PATH},
        wait=False
    )
    
    print(f"Visualization job launched: {estimator.latest_training_job.name}")
    return estimator

if __name__ == "__main__":
    launch_visualization()
