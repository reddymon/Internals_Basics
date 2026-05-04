FROM python:3.12-slim
WORKDIR /app
COPY . .
RUN pip install pandas scikit-learn
CMD ["python", "src/predict_cli.py"]