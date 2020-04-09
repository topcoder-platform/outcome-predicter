const Joi = require('joi')

module.exports = {
  '/predict': {
    post: {
      controller: 'ChallengePredictionController',
      method: 'predictSuccess',
      schema: {
        challengeId: Joi.string().required(),
        model: Joi.string()
      }
    }
  }
}
