const mongoose = require('mongoose');

const reportSchema = new mongoose.Schema({
    report_id: {
        type: String,
        required: true,
        unique: true
    },
    file_path: {
        type: String,
        required: true
    },
    summary: {
        total_users: Number,
        high_risk_count: Number,
        avg_churn_score: Number
    },
    status: {
        type: String,
        enum: ['Pending', 'Completed', 'Failed'],
        default: 'Pending'
    },
    created_at: {
        type: Date,
        default: Date.now
    }
});

module.exports = mongoose.model('Report', reportSchema);
