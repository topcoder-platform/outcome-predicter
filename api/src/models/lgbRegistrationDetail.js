const _ = require('lodash')
const util = require('../common/util')
const prediction = require('../common/prediction.util')

/**
 * Format a date string into the format 'YYYY-mm-dd'.
 * @param dateStr
 * @return {string}
 */
const formatDate = (dateStr) => {
  const date = new Date(dateStr)
  return date ? `${date.getFullYear()}-${date.getMonth() + 1}-${date.getDate()}` : ''
}

/**
 * Format a date time string into the format 'YYYY-mm-dd HH:mm:ss'.
 * @param dateTimeStr
 * @return {string}
 */
const formatDateTime = (dateTimeStr) => {
  const dateTime = new Date(dateTimeStr)
  return dateTime ? `${formatDate(dateTime)} ${util.lpad(dateTime.getHours(), 2)}:${util.lpad(dateTime.getMinutes(), 2)}:00` : ''
}

/**
 * Create a request body using challenge data
 * @param params object includes challenge and challengeResources
 * @return {string}
 */
const createRequestBody = (params) => {
  const challenge = params.challenge
  const challengeResources = params.challengeResources
  console.log(challenge)
  console.log(challengeResources)
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
    data.set('Challenge Stats Project Category Name', 'Code') // hardcode Code
    data.set('Challenge Stats Submitby Date Date', formatDateTime(challenge.submissionEndDate))
    data.set('Challenge Stats Complete Date Date', formatDateTime(challenge.phases[challenge.phases.length - 1]['actualEndTime']))
    data.set('Challenge Stats Status Desc', challenge.currentStatus)
    data.set('Challenge Stats Tco Track', challenge.event[0]['eventShortDesc'])
    data.set('Challenge Stats Submitby Date Time', formatDateTime(challenge.submissionEndDate))
    data.set('Challenge Stats Tc Direct Project ID', challenge.projectId)
    data.set('Challenge Stats Challenge Manager', firstManager.properties.Handle)
    data.set('Challenge Stats Challenge Copilot', firstCopilot.properties.Handle)
    data.set('Challenge Stats Posting Date Date', formatDateTime(challenge.postingDate))
    data.set('Challenge Stats Track', util.makeTheFirstLetterUppercase(challenge.challengeCommunity))
    data.set('Challenge Stats Technology List', `"${challenge.technologies.join(',')}"`)
    data.set('Challenge Stats Registrant Handle', registrant.handle)
    data.set('Challenge Stats Submit Ind', 0) // FIXME
    data.set('Challenge Stats Valid Submission Ind', 0) // FIXME
    data.set('Member Profile Advanced Reporting Country', '') // TODO Missed in challenge resources
    data.set('User Member Since Date', formatDateTime(members[registrant.handle].properties['Registration Date']))
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
