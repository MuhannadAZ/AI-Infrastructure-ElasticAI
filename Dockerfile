
FROM python:3.12-slim

# Allow the same image to work on AWS Lambda
COPY --from=public.ecr.aws/awsguru/aws-lambda-adapter:1.0.1 /lambda-adapter /opt/extensions/lambda-adapter

WORKDIR /var/task

ENV PORT=8000
ENV PYTHONUNBUFFERED=1
ENV AWS_LWA_READINESS_CHECK_PATH=/health

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY train_model.py .

# Train the model inside the container image
RUN python train_model.py

EXPOSE 8000

CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
ENV AWS_LWA_PORT=8000