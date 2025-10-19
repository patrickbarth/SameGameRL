# For more information, please refer to https://aka.ms/vscode-docker-python
FROM python:3.13-trixie

# Python
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

# Poetry
ENV POETRY_CACHE_DIR='/var/cache/pypoetry' \
    POETRY_HOME='/usr/local'

RUN apt-get update && apt-get install -y --no-install-recommends curl
RUN curl -sSL https://install.python-poetry.org | python3 -

WORKDIR /app
COPY poetry.lock /app
COPY pyproject.toml /app

RUN poetry install

RUN apt-get update && apt-get install -y --no-install-recommends vim

COPY . /app

# Creates a non-root user with an explicit UID and adds permission to access the /app folder
# For more info, please refer to https://aka.ms/vscode-docker-python-configure-containers
RUN adduser -u 5678 --disabled-password --gecos "" appuser && chown -R appuser /app
USER appuser

CMD ["poetry", "run", "pytest"]