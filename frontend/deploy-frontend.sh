docker build . -t indicmtapp-frontend
docker run -p 8890:80 indicmtapp-frontend