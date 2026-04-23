const path = require('path');
require('dotenv').config({ path: path.join(__dirname, '../../../.env') });

const config = {
    PORT: process.env.PORT || 3000,
    NODE_ENV: process.env.NODE_ENV || 'development',
    ML_SERVICE_URL: process.env.ML_SERVICE_URL || 'http://localhost:8000',
    BASE_URL: process.env.BASE_URL || 'http://localhost:3000',
    MONGO_URI: process.env.MONGO_URI || 'mongodb://localhost:27017/churnai',
    JWT_SECRET: process.env.JWT_SECRET || 'dev-secret'
};

// Validate critical config
const required = ['ML_SERVICE_URL', 'MONGO_URI'];
required.forEach(key => {
    if (!config[key]) {
        console.error(`CRITICAL ERROR: Missing environment variable ${key}`);
        process.exit(1);
    }
});

module.exports = config;
