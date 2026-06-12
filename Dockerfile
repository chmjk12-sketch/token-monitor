FROM nginx:alpine
COPY dashboard.html /usr/share/nginx/html/index.html
COPY api_logs.jsonl /usr/share/nginx/html/api_logs.jsonl
RUN echo 'server { listen 80; root /usr/share/nginx/html; location / { try_files $uri $uri/ /index.html; } location /health { return 200 "OK"; } }' > /etc/nginx/conf.d/default.conf
EXPOSE 80
