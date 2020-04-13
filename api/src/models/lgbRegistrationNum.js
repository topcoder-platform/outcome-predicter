const _ = require('lodash')
const util = require('../common/util')
const prediction = require('../common/prediction.util')

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
  return util.toCSV([data])
}

/**
 * Predict if a challenge will success or not.
 * @param params
 * @return {Promise<{prediction: string}>}
 */
const predictSuccess = async (params) => {
  const requestBody = createRequestBody(params)
  return prediction.predictSuccess(params.endpointName, requestBody)
}

module.exports = {
  predictSuccess
}
