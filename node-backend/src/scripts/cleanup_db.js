const mongoose = require('mongoose');
require('dotenv').config({ path: 'd:/congnigant/AiAgentic/.env' });

async function cleanup() {
    try {
        await mongoose.connect(process.env.MONGO_URI);
        const db = mongoose.connection.db;
        
        console.log('Cleaning up database...');
        
        const res1 = await db.collection('forecasts').deleteMany({
            $or: [
                { customer_id: null },
                { customer_id: '' },
                { customer_id: 'null' },
                { customer_id: { $exists: false } }
            ]
        });
        
        const res2 = await db.collection('predictions').deleteMany({
            $or: [
                { customer_id: null },
                { customer_id: '' },
                { customer_id: 'null' },
                { customer_id: { $exists: false } }
            ]
        });
        
        console.log('Deleted corrupt forecasts:', res1.deletedCount);
        console.log('Deleted corrupt predictions:', res2.deletedCount);
        
        process.exit(0);
    } catch (err) {
        console.error('Cleanup failed:', err);
        process.exit(1);
    }
}

cleanup();
