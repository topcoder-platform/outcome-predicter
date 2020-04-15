/**
 * The configuration file.
 */
module.exports = {
  LOG_LEVEL: process.env.LOG_LEVEL || 'debug',
  PORT: process.env.PORT || 3000,
  TOPCODER_API: process.env.TOPCODER_API || 'http://api.topcoder.com/v4',
  TOPCODER_V3_API: process.env.TOPCODER_V3_API || 'http://api.topcoder.com/v3',
  TOPCODER_AUTH_TOKEN: process.env.TOPCODER_AUTH_TOKEN,
  AWS_REGION: process.env.AWS_REGION,
  AWS_ACCESS_KEY: process.env.AWS_ACCESS_KEY,
  AWS_SECRET_KEY: process.env.AWS_SECRET_KEY,
  DEFAULT_MODEL: 'lgbRegistrationNum',
  MODELS: {
    lgbRegistrationNum: {
      endpointName: process.env.LGB_REGISTRATION_NUM_MODEL_ENDPOINT || '<ENDPOINT_NAME>',
      path: '/models/lgbRegistrationNum.js',
      enabled: true
    },
    lgbRegistrationDetail: {
      endpointName: process.env.LGB_REGISTRATION_DETAIL_MODEL_ENDPOINT || '<ENDPOINT_NAME>',
      path: '/models/lgbRegistrationDetail.js',
      enabled: true
    }
  }
}
