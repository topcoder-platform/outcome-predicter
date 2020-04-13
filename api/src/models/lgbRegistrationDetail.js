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
  const members = _.reduce(_.groupBy(challengeResources, 'properties.Handle'), (result, group, handle) => {
    result[handle] = group[0]
    return result
  }, {}) // grouped by member's handle
  const firstCopilot = _.find(challengeResources, { role: 'Copilot' })
  const firstManager = _.find(challengeResources, { role: 'Manager' })
  const records = []
  for (const registrant of challenge.registrants) {
    const data = new Map()
    if (!members[registrant.handle]) {
      continue
    }
    data.set('Challenge Stats Challenge ID', challenge.challengeId)
    data.set('Challenge Stats Challenge Name', challenge.challengeTitle)
    data.set('Challenge Stats Project Category Name', 'Code') // TODO
    data.set('Challenge Stats Submitby Date Date', util.formatDateTime(challenge.submissionEndDate))
    data.set('Challenge Stats Complete Date Date', '2020-01-03 00:00:00') // TODO the end date of last phase?
    data.set('Challenge Stats Status Desc', challenge.currentStatus)
    data.set('Challenge Stats Tco Track', '') // TODO
    data.set('Challenge Stats Submitby Date Time', '2020-01-01 03:36:45') // TODO is it the same as Challenge Stats Submitby Date Date?
    data.set('Challenge Stats Tc Direct Project ID', challenge.projectId)
    data.set('Challenge Stats Challenge Manager', firstManager.properties.Handle)
    data.set('Challenge Stats Challenge Copilot', firstCopilot.properties.Handle)
    data.set('Challenge Stats Posting Date Date', util.formatDateTime(challenge.postingDate))
    data.set('Challenge Stats Track', util.makeTheFirstLetterUppercase(challenge.challengeCommunity))
    data.set('Challenge Stats Technology List', `"${challenge.technologies.join(',')}"`)
    data.set('Challenge Stats Registrant Handle', registrant.handle)
    data.set('Challenge Stats Submit Ind', 0) // TODO
    data.set('Challenge Stats Valid Submission Ind', 0) // TODO
    data.set('Member Profile Advanced Reporting Country', '') // TODO Missed in challenge resources
    data.set('User Member Since Date', util.formatDateTime(members[registrant.handle].properties['Registration Date']))
    data.set('Challenge Stats Duration', 0)
    data.set('Challenge Stats Num Valid Submissions', 0)
    data.set('Challenge Stats First Place Prize', `${challenge.prizes[0]}`)
    data.set('Challenge Stats Num Appeals', 0)
    data.set('Challenge Stats Num Checkpoint Submissions', 0)
    data.set('Challenge Stats Num Registrations', challenge.numberOfRegistrants)
    data.set('Challenge Stats Num Submissions', challenge.numberOfSubmissions)
    data.set('Challenge Stats Num Submissions Passed Review', 0)
    data.set('Challenge Stats Num Successful Appeals', 0)
    data.set('Challenge Stats Num Valid Checkpoint Submissions', 0)
    data.set('Challenge Stats Contest Prizes Total', '0')
    data.set('Challenge Stats Copilot Cost', '0')
    data.set('Challenge Stats Total Prize', `${_.sum(challenge.prizes)}`)
    data.set('Challenge Stats Num Ratings', registrant.rating || 0)
    data.set('Challenge Stats Old Rating', 0)
    data.set('Challenge Stats Raw Score', 0)
    data.set('Challenge Stats score', 0)
    data.set('Challenge Stats Wins', 0)
    records.push(data)
  }
  return util.toCSV(records)
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
