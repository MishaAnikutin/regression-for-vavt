FROM python:3.11.9-slim

COPY requirements.txt .
RUN pip install --upgrade -r requirements.txt

COPY app app

CMD ["uvicorn", "app.main:forecast_service", "--host", "0.0.0.0", "--port", "5051"]