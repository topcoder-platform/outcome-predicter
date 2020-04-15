/*
 * Wrappers for Topcoder API.
 */
const config = require('config')
const request = require('superagent')
const _ = require('lodash')
const country = require('i18n-iso-countries')
const logger = require('./logger')

const countryNameList = country.getNames('en')

/**
 * Get country name by 3-letters country code or numeric country code.
 *
 * @param {String} countryCode the country code
 * @param {String} standard='ISO3' valid values are "ISO3" and "ISO2"
 * @returns {String} the country name
 */
function getCountryName (countryCode, standard = 'ISO3') {
  if (!isNaN(countryCode)) {
    return countryNameList[country.numericToAlpha2(countryCode)]
  }
  if (countryCode.length !== 3) {
    throw new Error(`Invalid country code or alpha2 country code: ${countryCode}`)
  }
  return countryNameList[country.alpha3ToAlpha2(countryCode)]
}

/**
 * Get member country name by handle.
 * The member's country is retrieved by calling TC Member API. The response data
 * counld be inconsistent, such as value is missed at field competitionCountryCode or homeCountryCode,
 * or only numeric code(instead of alpha3 code) available.
 *
 * @param {String} memberHandle the handle of the member
 * @returns {String} the country name
 */
async function getMemberCountry (memberHandle) {
  const result = await request.get(`${config.TOPCODER_V3_API}/members/${memberHandle}`)
  const countryCode = _.get(result.body, 'result.content.competitionCountryCode') || _.get(result.body, 'result.content.homeCountryCode')
  const countryName = getCountryName(countryCode)
  logger.debug(`country name for member ${memberHandle} is ${countryName}`)
  return countryName
}

module.exports = {
  getMemberCountry
}
