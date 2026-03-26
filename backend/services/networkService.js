const hf = require('./huggingfaceService');
module.exports = { analyze: async (data) => {
    const res = await hf.callModel('network-ae', data);
    return { domain: 'IT/Network Intrusion', status: 'success', insights: res.is_anomaly ? 'Anomalous traffic signature detected' : 'Standard traffic patterns identified', ...res };
}};