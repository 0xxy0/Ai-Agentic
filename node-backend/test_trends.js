const mongoose = require('mongoose');
const path = require('path');
require('dotenv').config({ path: path.join(__dirname, '../.env') });
const Prediction = require('./src/models/prediction.model');

const MONGO_URI = process.env.MONGO_URI;

mongoose.connect(MONGO_URI).then(async () => {
    const trends = await Prediction.aggregate([
        { $unwind: '$metadata.activity_history' },
        {
            $group: {
                _id: { 
                    month: '$metadata.activity_history.month',
                    year: '$metadata.activity_history.year'
                },
                activity: { $sum: '$metadata.activity_history.txns' },
                spend: { $sum: '$metadata.activity_history.spend' }
            }
        },
        { $sort: { '_id.year': 1, '_id.month': 1 } }
    ]);
    console.log('Trends count:', trends.length);
    console.log(JSON.stringify(trends.slice(0, 3), null, 2));
    process.exit(0);
}).catch(err => {
    console.error(err);
    process.exit(1);
});
