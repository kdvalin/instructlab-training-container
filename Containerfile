FROM quay.io/fabiendupont/pytorch-devel:nvidia-24.04-py3-ubi9

ENV PATH="/opt/python3.11/venv/bin:$PATH"

RUN python3 -m ensurepip --upgrade
COPY ./entrypoint.py .

ARG TRAINING_VERSION
RUN python3 -m pip install instructlab-training$([ -z "${TRAINING_VERSION}" ] && echo "" || echo "==${TRAINING_VERSION}")

ENTRYPOINT [ "python3", "entrypoint.py" ]