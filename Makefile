	SHELL := /bin/bash
# (C)2022-2025
# Version 1.4
# Written by Joe Cincotta
#

help: ## This help
	@echo "Zooniverse front end"
	@echo -e "$$(grep -hE '^\S+:.*##' $(MAKEFILE_LIST) | sed -e 's/:.*##\s*/:/' -e 's/^\(.\+\):\(.*\)/\\x1b[36m\1\\x1b[m:\2/' | column -c2 -t -s :)"

force-rebuild: ## Force rebuild front end
	export COMPOSE_BAKE=true;docker compose build --no-cache
	@echo "Finished"
	@date +"%F %T"

build: ## Build front end
	export COMPOSE_BAKE=true;docker compose build
	export COMPOSE_BAKE=true;docker compose pull
	@echo "Finished"
	@date +"%F %T"

setup: ## Set up environment
	docker exec -it cvat_server bash -ic 'python3 ~/manage.py createsuperuser'
	@echo "Finished"
	@date +"%F %T"

health: ## Healthcheck 
	docker exec -t cvat_server python manage.py health_check
	@echo "Finished"
	@date +"%F %T"

start: ## Start Local Docker Instances
	docker compose  -f docker-compose.yml -f components/serverless/docker-compose.serverless.yml up -d
	@echo "Finished"
	@date +"%F %T"

run: ## Start Local Docker Instances
	docker compose -f docker-compose.yml -f components/serverless/docker-compose.serverless.yml up
	# docker compose run --rm project start --build
	@echo "Finished"
	@date +"%F %T"

stop: ## Stop Local Docker Instances
	docker compose stop
	@echo "Finished"
	@date +"%F %T"

restart: stop start ## Restart the server
	@echo "Done"

attach: ## attach to server
	docker exec -it cvat_server bash

deploy-tracker: ## Deploy tracker serverless function
	cd serverless;./deploy_cpu.sh onnx/cvat/tracker/nuclio/

