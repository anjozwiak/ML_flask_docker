FROM python:3.8-slim
WORKDIR /app

COPY app.py	.
COPY requirements.txt	.
COPY train.py . 
COPY Customertravel.csv .

RUN pip install -r requirements.txt
RUN python3 train.py

ENTRYPOINT ["python3"]
CMD ["app.py"]
