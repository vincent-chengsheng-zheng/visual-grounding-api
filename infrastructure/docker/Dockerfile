FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

WORKDIR /app
COPY . .

RUN pip install -r requirements.txt --no-cache-dir
RUN pip install flash-attn --no-cache-dir --no-build-isolation

EXPOSE 8000
CMD ["uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "8000"]
