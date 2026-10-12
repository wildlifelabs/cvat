#!/bin/bash
#

curl -s https://api.github.com/repos/nuclio/nuclio/releases/latest \
	| grep -i "browser_download_url.*nuctl.*$(uname)" \
	| cut -d : -f 2,3 \
	| tr -d \" \
	| wget -O nuctl -qi - && chmod +x nuctl


