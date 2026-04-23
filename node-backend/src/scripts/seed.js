const mongoose = require('mongoose');
const fs = require('fs');
const path = require('path');
const csv = require('csv-parser');
const config = require('../config/env');
const Prediction = require('../models/prediction.model');
const Report = require('../models/report.model');

const SEED_DATA_DIR = path.join(__dirname, '../../../ml-service/data/batch_results');

async function seed() {
    try {
        console.log('Connecting to MongoDB...');
        await mongoose.connect(config.MONGO_URI);
        console.log('Connected.');

        // 1. Find latest files
        const files = fs.readdirSync(SEED_DATA_DIR);
        const csvFile = files.filter(f => f.startsWith('churn_scores_') && f.endsWith('.csv')).sort().reverse()[0];
        const jsonFile = files.filter(f => f.startsWith('dashboard_') && f.endsWith('.json')).sort().reverse()[0];

        if (!csvFile || !jsonFile) {
            console.error('No seed files found in ml-service/data/batch_results');
            process.exit(1);
        }

        console.log(`Seeding from: ${csvFile} and ${jsonFile}`);

        // 2. Clear existing data (optional, but good for "uploading" fresh)
        await Prediction.deleteMany({});
        await Report.deleteMany({});
        console.log('Cleared existing predictions and reports.');

        // 3. Process Activity CSV (for features)
        const userActivity = {};
        await new Promise((resolve, reject) => {
            fs.createReadStream(path.join(__dirname, '../../../ml-service/data/activity.csv'))
                .pipe(csv())
                .on('data', (data) => {
                    const uid = data.user_id;
                    if (!userActivity[uid]) {
                        userActivity[uid] = { total_spend: 0, total_txns: 0, months: [] };
                    }
                    userActivity[uid].total_spend += parseFloat(data.spend || 0);
                    userActivity[uid].total_txns += parseInt(data.txn_count || 0);
                    userActivity[uid].months.push({
                        month: parseInt(data.month),
                        year: parseInt(data.year),
                        spend: parseFloat(data.spend || 0),
                        txns: parseInt(data.txn_count || 0)
                    });
                })
                .on('end', resolve)
                .on('error', reject);
        });
        console.log('Processed activity data.');

        // 4. Process Churn CSV (Predictions)
        const predictions = [];
        await new Promise((resolve, reject) => {
            fs.createReadStream(path.join(SEED_DATA_DIR, csvFile))
                .pipe(csv())
                .on('data', (data) => {
                    const uid = data.user_id;
                    predictions.push({
                        user_id: uid,
                        churn_score: parseFloat(data.churn_score),
                        risk_level: data.risk_level.charAt(0) + data.risk_level.slice(1).toLowerCase(),
                        metadata: {
                            monetary: userActivity[uid]?.total_spend || 0,
                            frequency: userActivity[uid]?.total_txns || 0,
                            activity_history: userActivity[uid]?.months || []
                        },
                        created_at: new Date()
                    });
                })
                .on('end', resolve)
                .on('error', reject);
        });

        await Prediction.insertMany(predictions);
        console.log(`Inserted ${predictions.length} enriched predictions.`);

        // 4. Process JSON (Report)
        const dashboardData = JSON.parse(fs.readFileSync(path.join(SEED_DATA_DIR, jsonFile), 'utf8'));
        const reportRecord = new Report({
            report_id: `SEED-${Date.now()}`,
            file_path: 'seeded_from_historical_data',
            summary: {
                total_users: dashboardData.summary.total_users,
                high_risk_count: dashboardData.summary.high_risk_count,
                avg_churn_score: dashboardData.summary.avg_churn_score
            },
            status: 'Completed',
            created_at: new Date(dashboardData.generated_at)
        });

        await reportRecord.save();
        console.log('Inserted seed report record.');

        console.log('Seeding completed successfully.');
        process.exit(0);
    } catch (error) {
        console.error('Seeding failed:', error);
        process.exit(1);
    }
}

seed();
