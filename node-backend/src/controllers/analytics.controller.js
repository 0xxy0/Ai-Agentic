const Prediction = require('../models/prediction.model');

exports.getSummary = async (req, res, next) => {
    try {
        const stats = await Prediction.aggregate([
            {
                $group: {
                    _id: null,
                    total_users: { $sum: 1 },
                    high_risk_count: {
                        $sum: { $cond: [{ $eq: ["$risk_level", "High"] }, 1, 0] }
                    },
                    avg_churn_score: { $avg: "$churn_score" }
                }
            }
        ]);
        res.json(stats[0] || { total_users: 0, high_risk_count: 0, avg_churn_score: 0 });
    } catch (error) {
        next(error);
    }
};

exports.getTrends = async (req, res, next) => {
    try {
        // Since we have history in metadata, we can aggregate that
        // For simplicity in this demo, we'll return the monthly activity totals
        const trends = await Prediction.aggregate([
            { $unwind: "$metadata.activity_history" },
            {
                $group: {
                    _id: { 
                        month: "$metadata.activity_history.month",
                        year: "$metadata.activity_history.year"
                    },
                    activity: { $sum: "$metadata.activity_history.txns" },
                    spend: { $sum: "$metadata.activity_history.spend" }
                }
            },
            { $sort: { "_id.year": 1, "_id.month": 1 } },
            {
                $project: {
                    month: "$_id.month",
                    year: "$_id.year",
                    activity: 1,
                    spend: 1,
                    _id: 0
                }
            }
        ]);
        res.json(trends);
    } catch (error) {
        next(error);
    }
};

exports.getSegments = async (req, res, next) => {
    try {
        const segments = await Prediction.aggregate([
            {
                $group: {
                    _id: "$risk_level",
                    count: { $sum: 1 }
                }
            }
        ]);
        res.json(segments);
    } catch (error) {
        next(error);
    }
};

exports.getHeatmap = async (req, res, next) => {
    try {
        const heatmap = await Prediction.aggregate([
            { $unwind: "$metadata.activity_history" },
            {
                $group: {
                    _id: {
                        risk: "$risk_level",
                        month: "$metadata.activity_history.month"
                    },
                    value: { $sum: "$metadata.activity_history.txns" }
                }
            },
            {
                $project: {
                    risk: "$_id.risk",
                    month: "$_id.month",
                    value: 1,
                    _id: 0
                }
            }
        ]);
        res.json(heatmap);
    } catch (error) {
        next(error);
    }
};

exports.getScatter = async (req, res, next) => {
    try {
        const scatter = await Prediction.find({}, {
            user_id: 1,
            churn_score: 1,
            'metadata.monetary': 1,
            'metadata.frequency': 1,
            risk_level: 1,
            _id: 0
        }).limit(500);
        
        const formatted = scatter.map(s => ({
            id: s.user_id,
            x: s.metadata.monetary,
            y: s.churn_score * 100, // Scale to percentage
            z: s.metadata.frequency,
            risk: s.risk_level
        }));
        res.json(formatted);
    } catch (error) {
        next(error);
    }
};

exports.getTopUsers = async (req, res, next) => {
    try {
        const users = await Prediction.find()
            .sort({ churn_score: -1 })
            .limit(10);
        res.json(users);
    } catch (error) {
        next(error);
    }
};
