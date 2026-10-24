# Transit desk + headless hunt worker.
#
#   docker build -t transit-hunt .
#   docker run -p 8765:8765 -e DESK_TOKEN=change-me -v $PWD/var:/app/var transit-hunt          # desk + UI
#   docker run -e STATE_BUCKET=s3://bucket/hunt -e AWS_ACCESS_KEY_ID=... transit-hunt worker    # headless

FROM node:20-slim AS web
WORKDIR /web
COPY web/package.json web/package-lock.json ./
RUN npm ci
COPY web/ ./
RUN npm run build -- --outDir /web/dist --emptyOutDir

FROM python:3.11-slim AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DESK_DATA=/app/var/desk \
    MPLBACKEND=Agg
RUN apt-get update \
    && apt-get install -y --no-install-recommends git tini \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt ./
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt boto3
COPY . .
COPY --from=web /web/dist ./tool/static
COPY docker/entrypoint.sh /usr/local/bin/entrypoint
RUN chmod 755 /usr/local/bin/entrypoint \
    && useradd --create-home hunter \
    && mkdir -p /app/var /app/cache /app/models/checkpoints \
    && chown -R hunter /app/var /app/cache /app/models/checkpoints
USER hunter
EXPOSE 8765
ENTRYPOINT ["tini", "--", "entrypoint"]
CMD ["desk"]
