FROM python:3.9-slim as builder

# Install dependencies

RUN pip install streamlit

# Install NGINX
RUN apt-get update && apt-get install -y nginx && \
    rm -rf /var/lib/apt/lists/*

# Copy NGINX config
COPY nginx/default.conf /etc/nginx/conf.d/default.conf


WORKDIR /app
COPY . /app


# Create entrypoint script
RUN echo '#!/bin/bash\n\
streamlit run /app/app.py --server.port 8080 &\n\
nginx -g "daemon off;"' > /start.sh && chmod +x /start.sh

EXPOSE 8890
CMD ["/bin/bash", "/start.sh"]
