/**
 * Configure all routes for express app
 */
const _ = require('lodash')
const HttpStatus = require('http-status-codes')
const routes = require('./src/routes')
const middleware = require('./src/middleware')

/**
 * Wrap async function to standard express function
 * @param {Function} fn the async function
 * @returns {Function} the wrapped function
 */
function wrapExpress (fn) {
  return function (req, res, next) {
    fn(req, res, next).catch(next)
  }
}

/**
 * Wrap all functions from object
 * @param obj the object (controller exports)
 * @returns {Object|Array} the wrapped object
 */
function autoWrapExpress (obj) {
  if (_.isArray(obj)) {
    return obj.map(autoWrapExpress)
  }
  if (_.isFunction(obj)) {
    if (obj.constructor.name === 'AsyncFunction') {
      return wrapExpress(obj)
    }
    return obj
  }
  _.each(obj, (value, key) => {
    obj[key] = autoWrapExpress(value)
  })
  return obj
}

/**
 * Configure all routes for express app
 * @param app the express app
 */
module.exports = (app) => {
  // Load all routes
  _.each(routes, (verbs, path) => {
    _.each(verbs, (def, verb) => {
      const controllerPath = `./src/controllers/${def.controller}`
      const method = require(controllerPath)[def.method]; // eslint-disable-line
      const schema = def.schema
      if (!method) {
        throw new Error(`${def.method} is undefined`)
      }
      const actions = []
      actions.push((req, res, next) => {
        req.signature = `${def.controller}#${def.method}`
        next()
      })
      actions.push(middleware(schema, 'body'))
      actions.push(method)
      app[verb](`${path}`, autoWrapExpress(actions))
    })
  })

  // Check if the route is not found or HTTP method is not supported
  app.use('*', (req, res) => {
    const route = routes[req.baseUrl]
    let status
    let message
    if (route) {
      status = HttpStatus.METHOD_NOT_ALLOWED
      message = 'The requested HTTP method is not supported.'
    } else {
      status = HttpStatus.NOT_FOUND
      message = 'The requested resource cannot be found.'
    }
    res.status(status).json({ message })
  })
}
