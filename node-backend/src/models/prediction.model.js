const mongoose = require('mongoose');

const predictionSchema = new mongoose.Schema({
    customer_id: {
        type: String,
        required: true,
        index: true
    },
    churn_score: {
        type: Number,
        required: true
    },
    risk_level: {
        type: String,
        enum: ['High', 'Medium', 'Low'],
        required: true
    },
    metadata: {
        type: Object,
        default: {}
    },
    created_at: {
        type: Date,
        default: Date.now
    }
});

module.exports = mongoose.model('Prediction', predictionSchema);
