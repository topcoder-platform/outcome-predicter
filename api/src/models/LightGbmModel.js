const config = require('config')
const AWS = require('aws-sdk')
const _ = require('lodash')
const logger = require('../common/logger')
const util = require('../common/util')
const sageMakerRuntime = new AWS.SageMakerRuntime({
  region: config.AWS_REGION,
  accessKeyId: config.AWS_ACCESS_KEY,
  secretAccessKey: config.AWS_SECRET_KEY
})

/**
 * Create a request body using challenge data
 * @param params object includes challenge and challengeResources
 * @return {string}
 */
const createRequestBody = (params) => {
  const challenge = params.challenge
  const challengeResources = params.challengeResources
  const firstCopilot = _.find(challengeResources, { role: 'Copilot' })
  const firstManager = _.find(challengeResources, { role: 'Manager' })
  const data = new Map()
  data.set('Challenge ID', challenge.challengeId)
  data.set('Submitby Date Time', util.formatDateTime(challenge.submissionEndDate))
  data.set('Tc Direct Project ID', challenge.projectId)
  data.set('Challenge Manager', firstManager.properties.Handle)
  data.set('Challenge Copilot', firstCopilot.properties.Handle)
  data.set('Posting Date Date', util.formatDate(challenge.postingDate))
  data.set('Posting Date Date', '31/3/2020')
  data.set('Track', util.makeTheFirstLetterUppercase(challenge.challengeCommunity))
  data.set('Technology List', `"${challenge.technologies.join(',')}"`)
  data.set('Duration', 0)
  data.set('Num Valid Submissions', 0)
  data.set('First Place Prize', `US$${challenge.prizes[0]}`)
  data.set('Num Appeals', 0)
  data.set('Num Checkpoint Submissions', 0)
  data.set('Num Registrations', challenge.numberOfRegistrants)
  data.set('Num Submissions', challenge.numberOfSubmissions)
  data.set('Num Submissions Passed Review', 0)
  data.set('Num Successful Appeals', 0)
  data.set('Num Valid Checkpoint Submissions', 0)
  data.set('Contest Prizes Total', 'US$0')
  data.set('Copilot Cost', 'US$0')
  data.set('Total Prize', `US$${_.sum(challenge.prizes)}`)
  return [...data.keys()].join(',') + '\n' + [...data.values()].join(',') + '\n'
}

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
  const score = lines[1].split(',')[0]
  if (score === '1.0') {
    return { prediction: 'success' }
  } else if (score === '0.0') {
    return { prediction: 'fail' }
  } else {
    throw new Error('The prediction is undefined')
  }
}

/**
 * Predict if a challenge will success or not.
 * @param params
 * @return {Promise<{prediction: string}>}
 */
const predictSuccess = async (params) => {
  const requestBody = createRequestBody(params)
  const endPointResponse = await sendRequestToEndpoint(params.endpointName, requestBody)
  const result = processEndpointResult(endPointResponse)
  return result
}

module.exports = {
  predictSuccess
}
