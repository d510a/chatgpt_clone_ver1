#!/usr/bin/env bash
mkdir -p ~/.streamlit/
echo "\
[server]\n\
headless = true\n\
enableCORS = false\n\
enableXsrfProtection = false\n\
" > ~/.streamlit/config.toml
