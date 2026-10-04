# Container image of the dashboard, the front end in compose.yaml. It holds no model: every description is
# sent to the back end (API_URL), so the image needs only Streamlit and pandas.
#   build: docker build -t tariff-dashboard dashboard/
FROM python:3.12-slim
RUN pip install --no-cache-dir "streamlit>=1.36" "pandas>=2.2"
WORKDIR /app
COPY streamlit_app.py .
RUN useradd --create-home --uid 10001 appuser
USER appuser
EXPOSE 8501
CMD ["streamlit", "run", "streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501"]
