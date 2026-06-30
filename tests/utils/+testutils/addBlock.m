function addBlock(obj, varargin)
%ADDBLOCK Test helper wrapper for local generation scripts.
%
% Keeping calls as testutils.addBlock(seq, ...) lets scripts switch
% insertion behavior without editing Pulseq internals.

    if ~isobject(obj) || ~ismethod(obj, 'addBlock')
        error('testutils:addBlock', 'First argument must be a Sequence-like object.');
    end

    obj.addBlock(varargin{:});
end
