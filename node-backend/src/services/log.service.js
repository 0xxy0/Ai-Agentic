const Log = require('../models/log.model');
const config = require('../config/env');
const os = require('os');

const INSTANCE_ID = process.env.HOSTNAME || os.hostname();

class LogService {
    async info(message, metadata = {}) {
        await this._save('Info', message, metadata);
    }

    async error(message, metadata = {}, error = null) {
        if (error) {
            metadata.stack = error.stack;
            metadata.errorMessage = error.message;
        }
        await this._save('Error', message, metadata);
    }

    async warning(message, metadata = {}) {
        await this._save('Warning', message, metadata);
    }

    async audit(message, metadata = {}) {
        await this._save('Audit', message, metadata);
    }

    async _save(type, message, metadata) {
        try {
            const logEntry = new Log({
                type,
                message,
                instance: INSTANCE_ID,
                metadata,
                timestamp: new Date()
            });
            await logEntry.save();
            
            // Still log to console for container logs (Docker/Kubernetes)
            if (type === 'Error') {
                console.error(`[${type}] ${message}`, JSON.stringify(metadata));
            } else {
                console.log(`[${type}] ${message}`);
            }
        } catch (err) {
            console.error('Failed to save log to MongoDB:', err.message);
        }
    }
}

module.exports = new LogService();
