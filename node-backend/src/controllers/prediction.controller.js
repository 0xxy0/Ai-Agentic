const mlService = require('../services/ml.service');
const Prediction = require('../models/prediction.model');

exports.predictUser = async (req, res, next) => {
  try {
    const result = await mlService.predictUser(req.body);
    
    // Save to MongoDB
    const prediction = new Prediction({
        user_id: req.body.user_id || 'unknown',
        churn_score: result.churn_probability || result.churn_score || 0,
        risk_level: result.risk_level || 'Low',
        metadata: result
    });
    await prediction.save();

    res.json(result);
  } catch (error) {
    next(error);
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
