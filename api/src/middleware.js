const Joi = require('joi')
const logger = require('./common/logger')
const HttpStatus = require('http-status-codes')

const middleware = (schema, property) => {
  return (req, res, next) => {
    const { error } = Joi.validate(req[property], schema)
    const valid = error == null
    if (valid) { next() } else {
      const { details } = error
      const message = details.map(i => i.message).join(',')
      logger.error(`Validation body parameters:  ${message}`)
      res.status(HttpStatus.UNPROCESSABLE_ENTITY).json({ message: message })
    }
  }
}
module.exports = middleware
