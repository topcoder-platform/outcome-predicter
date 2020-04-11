const HttpStatus = require('http-status-codes')
const config = require('config')
const logger = require('../common/logger')
const superagent = require('superagent')

/**
 * Predict if a challenge will success or not.
 * @param req
 * @param res
 * @return {Promise<void>}
 */
async function predictSuccess (req, res) {
  const challengeId = req.body.challengeId
  const modelName = req.body.model
  const defaultModelName = config.DEFAULT_MODEL
  const selectedModelName = modelName || defaultModelName
  const selectedModel = config.MODELS[selectedModelName]
  let httpStatus
  let resultResponse
  logger.debug(`Predict success using the model=${selectedModelName} for challengeId=${challengeId}`)
  if (!selectedModel) {
    httpStatus = HttpStatus.BAD_REQUEST
    resultResponse = { message: `The model '${selectedModelName}' is undefined` }
  } else if (selectedModel.enabled === false) {
    httpStatus = HttpStatus.BAD_REQUEST
    resultResponse = { message: `The model '${selectedModelName}' is disabled` }
  } else {
    const challenge = await getChallenge(challengeId)
    const challengeResources = await getChallengeResources(challengeId)
    const modelFile = require(`../${selectedModel.path}`)
    if (!modelFile) {
      throw new Error(`The model path '${selectedModel.path}' is undefined`)
    }
    httpStatus = HttpStatus.OK
    resultResponse = await modelFile.predictSuccess({ challenge: challenge, challengeResources: challengeResources, endpointName: selectedModel.endpointName })
  }
  logger.debug(`Result for challengeId=${challengeId}: ${JSON.stringify(resultResponse)}`)
  res.status(httpStatus).json(resultResponse)
}

/**
 * Get challenge by challengeId
 * @param challengeId
 * @return {Promise<Object>}
 */
async function getChallenge (challengeId) {
  const response = await superagent.get(`${config.TOPCODER_API}/challenges/${challengeId}`)
  return response.body.result.content
}

/**
 * Get challenge resources by challengeId
 * @param challengeId
 * @return {Promise<Object>}
 */
async function getChallengeResources (challengeId) {
  const response = await superagent.get(`${config.TOPCODER_API}/challenges/${challengeId}/resources`)
    .set({ Authorization: `Bearer ${config.TOPCODER_AUTH_TOKEN}` })
  return response.body.result.content
}

module.exports = {
  predictSuccess
}
