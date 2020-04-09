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

module.exports = {
  makeTheFirstLetterUppercase,
  formatDate,
  formatDateTime,
  lpad
}
