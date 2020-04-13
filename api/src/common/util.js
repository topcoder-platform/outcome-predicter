/**
 * This module contains the common util functions
 */

/**
 * Make the first letter uppercase
 * @param str
 * @return {string}
 */
const makeTheFirstLetterUppercase = (str) => {
  return str.charAt(0).toUpperCase() + str.slice(1)
}

/**
 * Format a date string into the format 'mm/dd/YYYY'.
 * @param dateStr
 * @return {string}
 */
const formatDate = (dateStr) => {
  const date = new Date(dateStr)
  return date ? `${date.getDate()}/${date.getMonth() + 1}/${date.getFullYear()}` : ''
}

/**
 * Format a date time string into the format 'mm/dd/YYYY HH:mm'.
 * @param dateTimeStr
 * @return {string}
 */
const formatDateTime = (dateTimeStr) => {
  const dateTime = new Date(dateTimeStr)
  return dateTime ? `${formatDate(dateTime)} ${lpad(dateTime.getHours(), 2)}:${lpad(dateTime.getMinutes(), 2)}` : ''
}

/**
 * Padding the left side of a string with zero
 * @param number
 * @param digits
 * @return {string}
 */
const lpad = (number, digits) => {
  return Array(Math.max(digits - String(number).length + 1, 0)).join(0) + number
}

/**
 * Convert test data from a list of Map to String.
 * @param records
 * @returns {string}
 */
const toCSV = (records) => {
  let result = [...records[0].keys()].join(',') + '\n'
  for (const record of records) {
    result += [...record.values()].join(',') + '\n'
  }
  return result
}

module.exports = {
  makeTheFirstLetterUppercase,
  formatDate,
  formatDateTime,
  lpad,
  toCSV
}
