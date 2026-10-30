# CPU-only twin of the Kaggle environment for local checks: Triton kernels run under
# TRITON_INTERPRET=1 and Soup's regression tests run with solution.apply(). Timing and memory
# numbers only come from a Kaggle T4.
FROM python:3.13-slim

RUN pip install --no-cache-dir torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir triton==3.6.0 numpy pytest \
    "soup-cli[train]==0.77.0" "transformers==5.16.1" "peft==0.20.0" "trl==0.29.1" \
    "bitsandbytes==0.50.2" "accelerate==1.14.0"

WORKDIR /repo
