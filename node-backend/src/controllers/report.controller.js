const mlService = require('../services/ml.service');
const pdfService = require('../services/pdf.service');
const Report = require('../models/report.model');

exports.getDashboard = async (req, res, next) => {
  try {
    const data = await mlService.getDashboard();
    res.json(data);
  } catch (error) {
    console.warn('ML Service unreachable, using MongoDB fallback for dashboard data');
    try {
        const Prediction = require('../models/prediction.model');
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
        const summary = stats[0] || { total_users: 0, high_risk_count: 0, avg_churn_score: 0 };
        res.json({
            summary,
            status: 'Offline (Using Cache)',
            timestamp: new Date()
        });
    } catch (fallbackError) {
        next(error); // If fallback also fails, throw original error
    }
  }
};

exports.getReports = async (req, res, next) => {
  try {
    const reports = await Report.find().sort({ created_at: -1 });
    res.json(reports);
  } catch (error) {
    next(error);
  }
};

exports.generateReport = async (req, res, next) => {
  try {
    const result = await mlService.generateReport();
    res.json(result);
  } catch (error) {
    next(error);
  }
};

exports.downloadPdf = async (req, res, next) => {
  try {
    // 1. Fetch latest dashboard data from ML microservice
    const dashboardData = await mlService.getDashboard();
    
    // 2. Generate professional PDF via Puppeteer
    const pdfBuffer = await pdfService.generateChurnReport(dashboardData);
    
    // 3. Save Record to MongoDB
    const reportRecord = new Report({
        report_id: `REP-${Date.now()}`,
        file_path: 'generated_on_the_fly',
        summary: {
            total_users: dashboardData.summary.total_users,
            high_risk_count: dashboardData.summary.high_risk_count,
            avg_churn_score: dashboardData.summary.avg_churn_score
        },
        status: 'Completed'
    });
    await reportRecord.save();

    // 4. Send response
    res.setHeader('Content-Type', 'application/pdf');
    res.setHeader('Content-Disposition', 'attachment; filename=ChurnAI_Intelligence_Report.pdf');
    res.send(Buffer.from(pdfBuffer));
  } catch (error) {
    console.error('Controller Error in downloadPdf:', error);
    next(error);
  }
};

exports.downloadCsv = async (req, res, next) => {
  try {
    const response = await mlService.getLatestCsv();
    res.setHeader('Content-Type', 'text/csv');
    res.setHeader('Content-Disposition', 'attachment; filename=churn_predictions.csv');
    res.send(Buffer.from(response.data, 'binary'));
  } catch (error) {
    next(error);
  }
};
