FROM python:3-alpine

WORKDIR /usr/src/flask_uuid

COPY . .
RUN pip install --no-cache-dir -r requirements.txt

CMD ["python","app.py"]
