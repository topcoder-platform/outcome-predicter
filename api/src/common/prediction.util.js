/**
 * This module contains the common util functions
 */
const config = require('config')
const AWS = require('aws-sdk')
const logger = require('../common/logger')

const sageMakerRuntime = new AWS.SageMakerRuntime({
  region: config.AWS_REGION,
  accessKeyId: config.AWS_ACCESS_KEY,
  secretAccessKey: config.AWS_SECRET_KEY
})

/**
 * Send a request to a model endpoint
 * @param endpointName
 * @param body
 * @return {Promise<unknown>}
 */
const sendRequestToEndpoint = async (endpointName, body) => {
  const params = {
    Body: body,
    EndpointName: endpointName,
    ContentType: 'text/csv'
  }

  return new Promise((resolve, reject) => {
    sageMakerRuntime.invokeEndpoint(params, function (error, data) {
      if (error) {
        logger.error(error)
        reject(error)
      } else {
        resolve(Buffer.from(data.Body).toString('utf8'))
      }
    })
  })
}

/**
 * Process an endpoint response
 * @param endpointResponse
 * @return {{prediction: string}}
 */
const processEndpointResult = (endpointResponse) => {
  const lines = endpointResponse.split(/\s*[\r\n]+\s*/g)
  const score = Number(lines[1].split(',')[0])
  if (score === 1) {
    return { prediction: 'success' }
  } else if (score === 0) {
    return { prediction: 'fail' }
  } else {
    throw new Error('The prediction is undefined')
  }
}

/**
 * Predict if a challenge will success or not.
 * @param endpointName
 * @param params
 * @return {Promise<{prediction: string}>}
 */
const predictSuccess = async (endpointName, requestBody) => {
  const endPointResponse = await sendRequestToEndpoint(endpointName, requestBody)
  const result = processEndpointResult(endPointResponse)
  return result
}

module.exports = {
  predictSuccess
}
