/**
 * Node.js API Gateway — Production Entry Point
 *
 * Stateless design: every instance is identical.
 * Instance ID is reported in health and access logs for tracing
 * across load-balanced replicas.
 */

'use strict';

const express  = require('express');
const cors     = require('cors');
const morgan   = require('morgan');
const config   = require('./config/env');
const connectDB = require('./db/mongo');
const os       = require('os');
const apiRoutes = require('./routes/api.routes');
const analyticsRoutes = require('./routes/analytics.routes');

// ── Database ────────────────────────────────────────────────────────────────
connectDB();

// ── Config ─────────────────────────────────────────────────────────────────
const INSTANCE_ID    = process.env.HOSTNAME || os.hostname();

// ── App bootstrap ──────────────────────────────────────────────────────────
const app = express();

// ── Trust Nginx reverse proxy ──────────────────────────────────────────────
app.set('trust proxy', 1);

// ── Middleware ─────────────────────────────────────────────────────────────
app.use(cors({ origin: '*' }));
app.use(express.json({ limit: '1mb' }));

// Structured access logging — prefix with instance ID for multi-replica trace
morgan.token('instance', () => INSTANCE_ID);
const logFormat = config.NODE_ENV === 'production'
  ? '{"time":":date[iso]","method":":method","url":":url","status"::status,"ms"::response-time,"instance":":instance"}'
  : 'dev';
app.use(morgan(logFormat));

// ── Routes ──────────────────────────────────────────────────────────────────
app.use('/api', apiRoutes);
app.use('/api/analytics', analyticsRoutes);

// ── Health endpoint — reports instance identity for load-balancer tracing ──
app.get('/health', (req, res) => {
  res.json({
    status:      'ok',
    service:     'node-backend',
    instance:    INSTANCE_ID,
    ml_service:  config.ML_SERVICE_URL,
    uptime_s:    Math.round(process.uptime()),
    env:         config.NODE_ENV,
    timestamp:   new Date().toISOString(),
  });
});

// ── 404 catch-all ──────────────────────────────────────────────────────────
app.use((req, res) => {
  res.status(404).json({ error: 'Not Found', path: req.originalUrl });
});

const logService = require('./services/log.service');

// ── Global error handler ───────────────────────────────────────────────────
app.use((err, req, res, _next) => {
  const status  = err.status || 500;
  const message = err.message || 'Internal Server Error';
  
  logService.error(`API Error: ${message}`, {
    path: req.originalUrl,
    status,
    method: req.method
  }, err);

  res.status(status).json({ error: message });
});

// ── Start ──────────────────────────────────────────────────────────────────
app.listen(config.PORT, '0.0.0.0', () => {
  console.log(JSON.stringify({
    level:      'info',
    message:    'Node Gateway started',
    instance:   INSTANCE_ID,
    port:       config.PORT,
    ml_service: config.ML_SERVICE_URL,
    env:        config.NODE_ENV,
  }));
});

module.exports = app;   // exported for testing
