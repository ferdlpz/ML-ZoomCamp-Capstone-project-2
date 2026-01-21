# Usamos una imagen base de Python
FROM python:3.11-slim

# Instalamos uv desde la imagen oficial
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Establecemos el directorio de trabajo
WORKDIR /app

# Copiamos los archivos de dependencias primero para aprovechar el caché de capas
COPY pyproject.toml uv.lock ./

# Instalamos las dependencias
# --system instala los paquetes en el entorno global del contenedor (ideal para Docker)
RUN uv sync --frozen --no-cache

# Copiamos el resto de la aplicación
COPY predict.py model.bin column_transformer.bin ./

# Exponemos el puerto
EXPOSE 9696

# Ejecutamos el servicio
# Usamos 'uv run' para asegurar que el entorno sea el correcto
ENTRYPOINT ["uv", "run", "gunicorn", "--bind=0.0.0.0:9696", "predict:app"]
