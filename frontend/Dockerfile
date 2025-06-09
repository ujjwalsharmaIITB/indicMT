# Use official nginx image
FROM nginx:alpine

# Remove default nginx static files
RUN rm -rf /usr/share/nginx/html/*

# Copy custom build into nginx html folder
COPY build/ /usr/share/nginx/html

# Optional: custom NGINX config
# COPY nginx.conf /etc/nginx/conf.d/default.conf

# Expose port (optional — for documentation)
EXPOSE 80

# Start Nginx
CMD ["nginx", "-g", "daemon off;"]
