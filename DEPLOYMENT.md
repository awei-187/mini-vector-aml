# Production deployment

Validated layout: Ubuntu 22.04, Python 3.10, single process, Nginx HTTPS.

- Source: /opt/mini-vector/app; virtualenv: /opt/mini-vector/venv.
- Private app settings: /opt/mini-vector/app/.env.
- Private embedding settings: /etc/mini-vector/qwen.env.
- Persistent database: /var/lib/mini-vector/enhanced-v2.sqlite3 (new; the
  baseline database memory.sqlite3 has a different identity and is rejected).
- The same private qwen.env also holds QWEN_RERANK_ENDPOINT and
  QWEN_RERANK_API_KEY_ENV for qwen3-rerank (see qwen.env.example).
- Run as the dedicated mini-vector user, with source read-only and write access
  to /var/lib/mini-vector. Configs can be root:mini-vector mode 640, their private
  parent directory mode 750, and data directory mini-vector mode 700.
- Keep API bound to 127.0.0.1:8000. Allow public TCP 80/443 for Nginx; restrict SSH
  separately. The API token authenticates Add/Search; do not put it in URLs.

Example systemd unit (save as /etc/systemd/system/mini-vector.service):

```ini
[Unit]
Description=Mini Vector AML API
Wants=network-online.target
After=network-online.target

[Service]
Type=simple
User=mini-vector
Group=mini-vector
WorkingDirectory=/opt/mini-vector/app
EnvironmentFile=/opt/mini-vector/app/.env
Environment=AML_EMBEDDING_ENV_FILE=/etc/mini-vector/qwen.env
Environment=MINI_VECTOR_RERANKER_ENV_FILE=/etc/mini-vector/qwen.env
Environment=PYTHONDONTWRITEBYTECODE=1
Environment=PYTHONUNBUFFERED=1
ExecStart=/opt/mini-vector/venv/bin/python -m mini_vector.aml_api --host 127.0.0.1 --port 8000 --db-path /var/lib/mini-vector/enhanced-v2.sqlite3
Restart=on-failure
RestartSec=5
TimeoutStopSec=120
UMask=0077
NoNewPrivileges=true
PrivateTmp=true
ProtectHome=true
ProtectSystem=strict
ReadWritePaths=/var/lib/mini-vector

[Install]
WantedBy=multi-user.target
```

Final submitted configuration (enhanced-v2 + time_v2 extraction, `list` Qwen
instruction, observation-date prefix in Search content). It needs a new database
path, because the stored identity differs from enhanced-v2.sqlite3:

```ini
ExecStart=/opt/mini-vector/venv/bin/python -m mini_vector.aml_api --host 127.0.0.1 --port 8000 --db-path /var/lib/mini-vector/final-v2list.sqlite3 --extraction-prompt time_v2 --search-time-format date_prefix --reranker-instruction-preset list
```

Example Nginx HTTP configuration; replace aml.example.com with your domain:

```nginx
server {
    listen 80;
    server_name aml.example.com;
    client_max_body_size 50m;
    access_log off;
    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_set_header Authorization $http_authorization;
        proxy_set_header Connection "";
        proxy_connect_timeout 5s;
        proxy_send_timeout 600s;
        proxy_read_timeout 600s;
        proxy_next_upstream off;
    }
}
```

Check nginx -t before reloading. Configure Certbot's Nginx integration after DNS
and public HTTP work; test HTTPS /health and certbot renew --dry-run and verify
the renewal timer. Keep TCP 80 reachable for HTTP-01 renewal. See the current
Certbot instructions: https://certbot.eff.org/instructions?ws=nginx&os=snap.

The service is synchronous and concurrent within one process: Adds for different
users and all Searches overlap, Adds for one user are serialized. Throughput is
bounded by provider rate limits and the host's CPU for vector scans; a synthetic
live test reached 21.5 Adds/minute at 16-way concurrency with ~30 s per Add.
Nginx's 600s timeout is an inactivity timeout, not a request-duration guarantee.
There is no distributed queue or horizontal scale.

AML requires a public repository and fixed commit for the open-source division,
plus the live API. Submit the service token only through the platform's private
credential field. Obtain Eval Key after review, then run official Textual Smoke.
Deployment checks and the synthetic script do not consume or replace official
Smoke. Full evaluation is a separate action and is not part of this release.
