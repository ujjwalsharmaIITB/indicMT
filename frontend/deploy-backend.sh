# streamlit run app.py --server.port 8890  --server.address "https://www.cfilt.iitb.ac.in/" --server.baseUrlPath "indicMTApp/"


docker build -t streamlit-nginx-app .
docker run -p 8890:8080 --name indicMT streamlit-nginx-app