FROM python:3.10

# Set working directory
WORKDIR /app

# Salin semua file ke container
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirement.txt

# Expose port 5000 untuk Flask
EXPOSE 5000

# Jalankan aplikasi
CMD ["python", "main.py"]
