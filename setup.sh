mkdir -p ~/.streamlit/

echo "\
[server]\n\
headless = true\n\
enableCORS=false\n\
port=$PORT
" > ~/.streamlit/config.toml
