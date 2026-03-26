const hf = require('./huggingfaceService');
module.exports = { analyze: async (data) => {
    const res = await hf.callModel('image-ae', data);
    return { domain: 'Healthcare/Imaging', status: 'success', insights: res.is_anomaly ? 'Abnormal features detected in scan' : 'Scan features align with baseline normal', ...res };
}};