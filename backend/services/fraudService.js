const hf = require('./huggingfaceService');
module.exports = { analyze: async (data) => {
    const res = await hf.callModel('fraud-ae', data);
    return { domain: 'Banking/Fraud', status: 'success', insights: res.is_anomaly ? 'High deviation in transaction pattern' : 'Transaction looks normal', ...res };
}};