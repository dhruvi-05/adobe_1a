FROM --platform=linux/amd64 python:3.10-slim

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

RUN mkdir -p /app/input /app/output

CMD ["python", "process_pdfs.py", "--input_dir", "/app/input", "--output_dir", "/app/output"]
