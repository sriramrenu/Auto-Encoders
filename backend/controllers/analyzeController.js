const modelRouter = require('../router/modelRouter');

const handleAnalysis = async (req, res, next) => {
    try {
        const { domain, textData } = req.body;
        const file = req.file;
        if (!domain) return res.status(400).json({ error: 'Domain must be specified' });
        if (!file && !textData) return res.status(400).json({ error: 'No data provided' });
        
        const result = await modelRouter.process(domain, { file, textData });
        res.json(result);
    } catch (error) { next(error); }
};
module.exports = { handleAnalysis };