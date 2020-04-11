/**
 * The configuration file.
 */
module.exports = {
  LOG_LEVEL: process.env.LOG_LEVEL || 'debug',
  PORT: process.env.PORT || 3000,
  TOPCODER_API: 'http://api.topcoder.com/v4',
  TOPCODER_AUTH_TOKEN: process.env.TOPCODER_AUTH_TOKEN,
  AWS_REGION: process.env.AWS_REGION,
  AWS_ACCESS_KEY: process.env.AWS_ACCESS_KEY,
  AWS_SECRET_KEY: process.env.AWS_SECRET_KEY,
  DEFAULT_MODEL: 'lightgbmModel',
  MODELS: {
    lightgbmModel: {
      endpointName: '<ENDPOINT_NAME>',
      path: '/models/LightGbmModel.js',
      enabled: true
    }
  }
}
