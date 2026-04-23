# ══════════════════════════════════════════════════════════════════════════════
#  ChurnAI — Makefile convenience targets
# ══════════════════════════════════════════════════════════════════════════════

.PHONY: help up down scale logs ps build test-lb clean

SCALE ?= 3   # default: 3 Node instances

help:          ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*##' Makefile | awk 'BEGIN{FS=":.*##"}{printf "\033[36m%-20s\033[0m %s\n",$$1,$$2}'

build:         ## Build all Docker images
	docker compose build --no-cache

up:            ## Start full stack (3 Node instances + ML + Nginx + MLflow)
	docker compose up -d --scale node-backend=$(SCALE)
	@echo "✅  Stack is up — http://localhost"

down:          ## Stop and remove all containers
	docker compose down

scale:         ## Hot-scale Node instances: make scale SCALE=5
	docker compose up -d --scale node-backend=$(SCALE) --no-recreate
	@echo "✅  Scaled node-backend to $(SCALE) instances"

logs:          ## Tail logs from all services
	docker compose logs -f

logs-nginx:    ## Tail Nginx access logs
	docker compose logs -f nginx

logs-node:     ## Tail Node gateway logs
	docker compose logs -f node-backend

logs-ml:       ## Tail ML service logs
	docker compose logs -f ml-service

ps:            ## Show running containers and health status
	docker compose ps

test-lb:       ## Fire 10 requests and show which Node instance handled each
	@echo "🔁  Testing load balancing across Node instances..."
	@for i in 1 2 3 4 5 6 7 8 9 10; do \
		curl -s http://localhost/node-health | python3 -c \
		  "import sys,json; d=json.load(sys.stdin); print(f'  req $$i → instance: {d[\"instance\"]}')" ; \
	done

test-predict:  ## Send a sample prediction request through Nginx
	curl -s -X POST http://localhost/api/predict-user \
	  -H "Content-Type: application/json" \
	  -d '{"user_id":"USR-TEST","txn_7d":5,"txn_30d":20,"txn_90d":45,"recency_days":2,"frequency":120,"monetary":500}' \
	  | python3 -m json.tool

health:        ## Check health of all services
	@echo "Nginx:    " && curl -s http://localhost/health | python3 -m json.tool
	@echo "Node:     " && curl -s http://localhost/node-health | python3 -m json.tool
	@echo "ML Svc:   " && curl -s http://localhost:8000/health | python3 -m json.tool || echo "(not exposed publicly)"

clean:         ## Remove containers, volumes, and built images
	docker compose down -v --rmi local
	docker system prune -f
