// MOCK HF SERVICE
const callModel = async (modelEndpoint, payload) => {
    return new Promise((resolve) => {
        setTimeout(() => {
            const errorScore = Math.random();
            const isAnomaly = errorScore > 0.75;
            resolve({
                reconstruction_error: errorScore.toFixed(4),
                is_anomaly: isAnomaly,
                confidence: (errorScore > 0.75 ? errorScore * 100 : (1 - errorScore) * 100).toFixed(2) + '%'
            });
        }, 1200);
    });
};
module.exports = { callModel };