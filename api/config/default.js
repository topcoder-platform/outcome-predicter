/**
 * The configuration file.
 */
module.exports = {
  LOG_LEVEL: process.env.LOG_LEVEL || 'debug',
  PORT: process.env.PORT || 3000,
  TOPCODER_API: 'http://api.topcoder.com/v4',
  AWS_REGION: process.env.AWS_REGION,
  DEFAULT_MODEL: 'lightgbmModel',
  MODELS: {
    lightgbmModel: {
      endpointName: '<ENDPOINT_NAME>',
      path: '/models/LightGbmModel.js',
      enabled: true
    }
  }
}
