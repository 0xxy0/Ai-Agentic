const axios = require('axios');
const config = require('../config/env');

const mlClient = axios.create({
  baseURL: config.ML_SERVICE_URL,
  timeout: 30000, // ML tasks can be slow
  headers: {
    'Content-Type': 'application/json'
  }
});

module.exports = mlClient;
