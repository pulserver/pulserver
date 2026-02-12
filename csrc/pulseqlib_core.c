int pulseqlib_load(
    pulseqlib_sequence_descriptor_collection* collection,
    pulseqlib_diagnostic* diag,
    const char* file_path,
    const pulseqlib_opts* opts)
{
    pulseqlib__seq_file_collection raw_coll = {0, NULL, NULL};
    pulseqlib_sequence_descriptor* descs = NULL;
    int rc;
    int i;

    if (!file_path || !opts || !collection || !diag) {
        return PULSEQLIB_ERR_NULL_POINTER;
    }

    pulseqlib_diagnostic_init(diag);

    /* Step 1: Parse all .seq files in the chain */
    rc = pulseqlib__read_seq_collection(&raw_coll, file_path, opts);
    if (PULSEQLIB_FAILED(rc)) {
        diag->code = rc;
        goto cleanup;
    }

    /* Step 2-4: For each subsequence, build descriptors */
    descs = (pulseqlib_sequence_descriptor*)ALLOC(
        sizeof(pulseqlib_sequence_descriptor) * raw_coll.num_sequences);
    if (!descs) {
        rc = PULSEQLIB_ERR_ALLOC_FAILED;
        diag->code = rc;
        goto cleanup;
    }

    for (i = 0; i < raw_coll.num_sequences; i++) {
        pulseqlib_sequence_descriptor desc = PULSEQLIB_SEQUENCE_DESCRIPTOR_INIT;

        rc = pulseqlib__get_unique_blocks(&raw_coll.sequences[i], &desc);
        if (PULSEQLIB_FAILED(rc)) {
            diag->code = rc;
            goto cleanup_descs;
        }

        rc = pulseqlib__find_tr_in_sequence(&desc, diag);
        if (PULSEQLIB_FAILED(rc)) goto cleanup_descs;

        rc = pulseqlib__find_segments_in_tr(&raw_coll.sequences[i], &desc, diag);
        if (PULSEQLIB_FAILED(rc)) goto cleanup_descs;

        descs[i] = desc;
    }

    /* Step 5: Build the collection */
    rc = pulseqlib__get_collection_descriptors(&raw_coll, collection, diag);
    if (PULSEQLIB_FAILED(rc)) goto cleanup_descs;

    /* Success - clean up raw parsed data, keep descriptors */
    pulseqlib__seq_file_collection_free(&raw_coll);
    return PULSEQLIB_OK;

cleanup_descs:
    for (i = 0; i < raw_coll.num_sequences; i++) {
        pulseqlib_sequence_descriptor_free(&descs[i]);
    }
    FREE(descs);
cleanup:
    pulseqlib__seq_file_collection_free(&raw_coll);
    return rc;
}