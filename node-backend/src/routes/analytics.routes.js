const express = require('express');
const router = express.Router();
const analyticsController = require('../controllers/analytics.controller');

router.get('/summary', analyticsController.getSummary);
router.get('/trends', analyticsController.getTrends);
router.get('/segments', analyticsController.getSegments);
router.get('/heatmap', analyticsController.getHeatmap);
router.get('/scatter', analyticsController.getScatter);
router.get('/top-users', analyticsController.getTopUsers);

module.exports = router;
