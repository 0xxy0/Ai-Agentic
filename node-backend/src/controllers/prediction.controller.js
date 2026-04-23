const mlService = require('../services/ml.service');
const Prediction = require('../models/prediction.model');
const mongoose = require('mongoose');

exports.predictUser = async (req, res, next) => {
  try {
    const result = await mlService.predictUser(req.body);
    
    // Save to MongoDB
    const prediction = new Prediction({
        customer_id: req.body.customer_id || req.body.user_id || 'unknown',
        churn_score: result.churn_probability || result.churn_score || 0,
        risk_level: result.risk_level || 'Low',
        metadata: result
    });
    await prediction.save();

    res.json(result);
  } catch (error) {
    console.warn('ML Service unreachable, checking database for existing prediction');
    try {
        const cid = req.body.customer_id || req.body.user_id;
        // 1. Check Prediction collection first
        let existing = await Prediction.findOne({ customer_id: cid });
        if (existing) {
            return res.json({
                ...existing.metadata,
                churn_probability: existing.churn_score,
                risk_level: existing.risk_level,
                source: 'Cache (ML Offline)'
            });
        }
        
        // 2. Check forecasts collection
        const db = mongoose.connection.db;
        existing = await db.collection('forecasts').findOne({ customer_id: cid });
        if (existing) {
            return res.json({
                ...existing,
                churn_probability: existing.predicted_churn_probability,
                risk_level: existing.risk_level,
                source: 'Forecast Cache (ML Offline)'
            });
        }

        res.status(503).json({ 
            message: 'ML Service Offline: No historical data found for this ID. Please ensure the ML microservice is running or the customer ID is correct.' 
        });
    } catch (fallbackError) {
        next(error);
    }
  }
};

exports.train = async (req, res, next) => {
  try {
    const result = await mlService.trainModel(req.body);
    res.json(result);
  } catch (error) {
    next(error);
  }
};
