const fraudService = require('../services/fraudService');
const imageService = require('../services/imageService');
const networkService = require('../services/networkService');

const process = async (domain, data) => {
    const d = domain.toLowerCase();
    if (d.includes('fraud') || d.includes('bank')) return await fraudService.analyze(data);
    if (d.includes('image') || d.includes('health')) return await imageService.analyze(data);
    if (d.includes('network') || d.includes('it')) return await networkService.analyze(data);
    throw new Error('Unsupported domain: ' + domain);
};
module.exports = { process };