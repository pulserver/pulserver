function tests = test_sequence_collection
% test_sequence_collection  Unit tests for pulserver.SequenceCollection.
%
%   results = runtests('test_sequence_collection')
%
% Requires the Pulseq MATLAB toolbox (mr.*) and compiled MEX.

    tests = functiontests(localfunctions);
end

%% --- Helper: build a simple 2D GRE sequence ---

function seq = make_simple_gre()
    sys = mr.opts('maxGrad', 32, 'gradUnit', 'mT/m', ...
                  'maxSlew', 130, 'slewUnit', 'T/m/s');
    seq = mr.Sequence(sys);

    [rf, gz] = mr.makeSincPulse(15*pi/180, 'Duration', 3e-3, ...
        'SliceThickness', 5e-3, 'system', sys);
    gx = mr.makeTrapezoid('x', 'FlatArea', 128/0.25, ...
        'FlatTime', 3.2e-3, 'system', sys);
    adc = mr.makeAdc(128, 'Duration', gx.flatTime, ...
        'Delay', gx.riseTime, 'system', sys);
    gxPre = mr.makeTrapezoid('x', 'Area', -gx.area/2, ...
        'Duration', 1e-3, 'system', sys);
    gzReph = mr.makeTrapezoid('z', 'Area', -gz.area/2, ...
        'Duration', 1e-3, 'system', sys);

    nPE = 16;
    peAreas = linspace(-0.5, 0.5, nPE) * (128/0.25);
    for i = 1:nPE
        gyPre = mr.makeTrapezoid('y', 'Area', peAreas(i), ...
            'Duration', 1e-3, 'system', sys);
        seq.addBlock(rf, gz);
        seq.addBlock(gxPre, gyPre, gzReph);
        seq.addBlock(gx, adc);
    end
end

%% --- Construction tests ---

function test_construction_from_sequence(testCase)
    seq = make_simple_gre();
    sc = pulserver.SequenceCollection(seq);
    verifyGreaterThan(testCase, sc.NumSequences, 0);
end

function test_construction_from_cell(testCase)
    seq = make_simple_gre();
    sc = pulserver.SequenceCollection({seq});
    verifyEqual(testCase, sc.NumSequences, 1);
end

function test_construction_from_file(testCase)
    seq = make_simple_gre();
    tmpFile = [tempname '.seq'];
    cleanObj = onCleanup(@() delete_if_exists(tmpFile));
    seq.write(tmpFile);
    sc = pulserver.SequenceCollection(tmpFile);
    verifyGreaterThan(testCase, sc.NumSequences, 0);
end

%% --- Accessor tests ---

function test_num_segments(testCase)
    seq = make_simple_gre();
    sc = pulserver.SequenceCollection(seq);
    r = sc.report();
    verifyGreaterThanOrEqual(testCase, numel(r.segments), 1);
end

function test_tr_size(testCase)
    seq = make_simple_gre();
    sc = pulserver.SequenceCollection(seq);
    r = sc.report();
    verifyGreaterThanOrEqual(testCase, r.tr_size, 1);
end

function test_report(testCase)
    seq = make_simple_gre();
    sc = pulserver.SequenceCollection(seq);
    r = sc.report();
    verifyTrue(testCase, isstruct(r) || isobject(r));
end

%% --- Safety check tests ---

function test_check_passes(testCase)
    seq = make_simple_gre();
    sc = pulserver.SequenceCollection(seq);
    % Should not throw for a well-formed GRE
    sc.check();
    verifyTrue(testCase, true);  % reached here without error
end

%% --- Helper ---

function delete_if_exists(f)
    if exist(f, 'file')
        delete(f);
    end
end
