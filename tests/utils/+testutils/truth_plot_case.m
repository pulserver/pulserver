function truth = truth_plot_case(base, varargin)
%TRUTH_PLOT_CASE Convenience entrypoint for report and plots.
%
% truth_plot_case('gre_2d_1sl_1avg')
% truth_plot_case('gre_2d_1sl_1avg', 'show_report', true)

    p = inputParser;
    addParameter(p, 'show_report',      true,      @islogical);
    addParameter(p, 'plot_tr',          true,      @islogical);
    addParameter(p, 'plot_segments',    true,      @islogical);
    addParameter(p, 'plot_freqmod',     true,      @islogical);
    addParameter(p, 'print_scan_table', true,      @islogical);
    addParameter(p, 'print_label_table',true,      @islogical);
    addParameter(p, 'plot_kspace_traj', true,      @islogical);
    addParameter(p, 'scan_shift',       [0 0 0],   @(x) isnumeric(x) && numel(x)==3);
    parse(p, varargin{:});

    truth = testutils.truth_parse_case(base);

    if p.Results.show_report
        testutils.truth_report_case(base);
    end
    if p.Results.plot_tr
        testutils.truth_plot_tr_waveforms(truth);
    end
    if p.Results.plot_segments
        testutils.truth_plot_segments(truth);
    end
    if p.Results.plot_freqmod && truth.freqmod_def.num_defs > 0
        for s = 1:truth.segment_def.num_segments
            seg = truth.segment_def.segments(s);
            has_fm = false;
            for b = 1:seg.num_blocks
                if seg.blocks(b).has_freq_mod
                    has_fm = true;
                    break;
                end
            end
            if has_fm
                testutils.truth_plot_freqmod_defs(truth, 'segment_idx', s);
            end
        end
    end
    if p.Results.print_scan_table
        testutils.truth_print_scan_table(truth, 'shift', p.Results.scan_shift);
    end
    if p.Results.print_label_table
        testutils.truth_print_label_table(truth);
    end
    if p.Results.plot_kspace_traj
        testutils.truth_plot_kspace_traj(truth);
    end
end
