# ══════════════════════════════════════════════════════════════════════════════
#  Node.js API Gateway Dockerfile — Production Multi-Stage Build
# ══════════════════════════════════════════════════════════════════════════════

# ── Stage 1: install production-only dependencies ─────────────────────────────
FROM node:20-alpine AS deps

WORKDIR /app
COPY package.json package-lock.json* ./
RUN npm ci --omit=dev --prefer-offline


# ── Stage 2: production runtime ───────────────────────────────────────────────
FROM node:20-alpine AS runtime

LABEL maintainer="devops-team"
LABEL description="ChurnAI Node.js API Gateway — Stateless, Horizontally Scalable"

# Install wget (healthcheck) + tini (PID 1 signal handler)
RUN apk add --no-cache wget tini

WORKDIR /app

# Copy production node_modules from deps stage (layer cache)
COPY --from=deps /app/node_modules ./node_modules

# Copy application source
COPY package.json ./
COPY src/          ./src/

# ── Security: run as non-root user ────────────────────────────────────────────
RUN addgroup -g 1001 -S nodeapp && \
    adduser  -u 1001 -S nodeapp -G nodeapp && \
    chown -R nodeapp:nodeapp /app
USER nodeapp

# ── Environment defaults (override via Compose) ────────────────────────────────
ENV NODE_ENV=production \
    PORT=3000 \
    ML_SERVICE_URL=http://ml-service:8000 \
    LOG_LEVEL=info

EXPOSE 3000

# Healthcheck used by docker-compose depends_on condition: service_healthy
HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=3 \
    CMD wget -qO- http://localhost:3000/health || exit 1

# tini as PID 1 for correct SIGTERM propagation (graceful shutdown)
ENTRYPOINT ["/sbin/tini", "--"]
CMD ["node", "src/app.js"]
