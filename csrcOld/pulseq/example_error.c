/* Example usage of custom error handling */
/*pulseqlib_Diagnostic diag;
int result = pulseqlib_findTRInSequence(&trDesc, &diag, ...);
if (result == 0) {
    // Use vendor-specific logging
    VENDOR_LOG_ERROR("Sequence analysis failed: %s", pulseqlib_getErrorMessage(diag.code));
    VENDOR_LOG_INFO("Hint: %s", pulseqlib_getErrorHint(diag.code));
    if (diag.blockIndex >= 0) {
        VENDOR_LOG_DEBUG("Problem at block %d", diag.blockIndex);
    }
}
*/
