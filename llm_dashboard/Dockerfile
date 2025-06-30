# Use official Python base image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set work directory
WORKDIR /app

# RUN apt-get update && apt-get install -y libpq-dev gcc

# Install dependencies
COPY requirements.txt /app/
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy project code
COPY . /app/

# Expose port (default Django port)
EXPOSE 8000

# Run Django server
# CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]

# CMD ["gunicorn", "--workers", "3", "--bind", "0.0.0.0:8000", "llm_dashboard.wsgi:application"]
CMD ["gunicorn", "--workers", "3", "--timeout", "960", "--bind", "0.0.0.0:8000", "llm_dashboard.wsgi:application"]

