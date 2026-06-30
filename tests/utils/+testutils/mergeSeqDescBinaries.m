function mergeSeqDescBinaries(out_path, src_paths)
%MERGESEQDESCBINARIES  Merge per-subseq _seq_desc.truth files into one collection
%                      truth file, mirroring the layout of TruthBuilder
%                      exportSequenceDescription (Section 5 wire format).
%
%   out_path   : destination .bin path
%   src_paths  : cell array of source .bin paths (one per subsequence/part)
%
%   The header is recomputed (min/max aggregations + sum of num_subseqs
%   and total_scan_time_us); per-subseq payloads are appended verbatim.
%   subseq_idx fields in the appended payloads are NOT renumbered — they
%   are reader-side informational. Tested against TruthBuilder output.

    if ischar(src_paths) || isstring(src_paths), src_paths = {char(src_paths)}; end
    if isempty(src_paths), error('truth:input', 'No source seq_desc files'); end

    HEADER_BYTES = 4*5 + 4 + 4*3;   % 5 floats + num_subseqs + reserved[3]

    min_te    =  inf;
    min_tr    =  inf;
    max_tr    = -inf;
    max_flip  = 0;
    total_us  = 0;
    nsubs_tot = 0;
    payloads  = cell(1, numel(src_paths));

    for i = 1:numel(src_paths)
        sp = src_paths{i};
        f  = fopen(sp, 'rb');
        if f < 0, error('truth:io', 'Cannot open %s', sp); end
        hdr = fread(f, 5, 'single=>single');
        nsub = fread(f, 1, 'int32=>int32');
        fread(f, 3, 'int32');                  % reserved
        body = fread(f, inf, 'uint8=>uint8');
        fclose(f);

        if numel(hdr) < 5 || isempty(nsub)
            error('truth:format', 'Bad seq_desc file: %s', sp);
        end

        if hdr(1) > 0 && hdr(1) < min_te,   min_te = hdr(1); end
        if hdr(2) > 0 && hdr(2) < min_tr,   min_tr = hdr(2); end
        if hdr(3) > max_tr,                 max_tr = hdr(3); end
        if hdr(4) > max_flip,               max_flip = hdr(4); end
        total_us  = total_us + double(hdr(5));
        nsubs_tot = nsubs_tot + double(nsub);
        payloads{i} = body;
    end

    if ~isfinite(min_te), min_te = 0; end
    if ~isfinite(min_tr), min_tr = 0; end
    if ~isfinite(max_tr), max_tr = 0; end

    f = fopen(out_path, 'wb');
    if f < 0, error('truth:io', 'Cannot open %s for write', out_path); end
    fwrite(f, single(min_te),   'float32');
    fwrite(f, single(min_tr),   'float32');
    fwrite(f, single(max_tr),   'float32');
    fwrite(f, single(max_flip), 'float32');
    fwrite(f, single(total_us), 'float32');
    fwrite(f, int32(nsubs_tot), 'int32');
    fwrite(f, int32([0 0 0]),   'int32');
    for i = 1:numel(payloads)
        fwrite(f, payloads{i}, 'uint8');
    end
    fclose(f);
end
