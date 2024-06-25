FROM quay.io/fabiendupont/pytorch-devel:nvidia-24.04-py3-ubi9

RUN python3.11 -m ensurepip --upgrade
COPY ./entrypoint.py .

ARG TRAINING_VERSION
RUN python3.11 -m pip install instructlab-training$([ -z "${TRAINING_VERSION}" ] && echo "" || echo "==${TRAINING_VERSION}")

ENTRYPOINT [ "python3.11", "entrypoint.py" ]