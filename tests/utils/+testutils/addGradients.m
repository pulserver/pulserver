function grad = addGradients(grads, varargin)
% addGradients  Drop-in replacement for mr.addGradients.
%
% Identical to mr.addGradients but fixes the internal call to
% mr.makeExtendedTrapezoid that passes 'system' as a name-value pair
% (unsupported in Octave's inputParser with addOptional).

persistent parser

if isempty(parser)
    parser = inputParser;
    parser.FunctionName = 'addGradients';
    parser.addRequired('grads');
    parser.addOptional('system', [], @isstruct);
    parser.addParamValue('maxGrad', 0, @isnumeric);
    parser.addParamValue('maxSlew', 0, @isnumeric);
end
parse(parser, grads, varargin{:});
opt = parser.Results;

if isempty(opt.system)
    system=mr.opts();
else
    system=opt.system;
end

maxSlew = system.maxSlew;
maxGrad = system.maxGrad;
if opt.maxGrad > 0
    maxGrad = opt.maxGrad;
end
if opt.maxSlew > 0
    maxSlew = opt.maxSlew;
end

if ~iscell(grads)
    error('gradients have to be passed as cell array');
end

if length(grads)<2
    error('cannot add less then two gradients');
end

channel = grads{1}.channel;

delays = [];
firsts = [];
lasts = [];
durs=[];
is_trap=[];
is_arb=[];
is_osa=[];
for ii = 1:length(grads)
    if grads{ii}.channel~=channel
        error('cannot add gradients on different channels');
    end
    delays = [delays, grads{ii}.delay];
    durs = [durs, mr.calcDuration(grads{ii})];
    is_trap = [is_trap, strcmp(grads{ii}.type,'trap')];
    if is_trap(end)
        is_arb = [is_arb, false];
        is_osa = [is_osa, false];
        firsts = [firsts, 0];
        lasts = [lasts, 0];
    else
        tt_rast=grads{ii}.tt/system.gradRasterTime;
        is_arb = [is_arb, all(abs(tt_rast(:)'+0.5-(1:length(tt_rast)))<1e-6)];
        is_osa = [is_osa, all(abs(tt_rast(:)'-0.5*(1:length(tt_rast)))<1e-6)];
        firsts = [firsts, grads{ii}.first];
        lasts = [lasts, grads{ii}.last];
    end
end
common_delay = min(delays);
total_duration = max(durs);
is_etrap=(~is_trap)&(~is_arb)&(~is_osa);

if all(is_trap)
    gradsa=cell2mat(grads);
    if 1==length(unique([gradsa.delay])) && ...
       1==length(unique([gradsa.riseTime])) && ...
       1==length(unique([gradsa.flatTime])) && ...
       1==length(unique([gradsa.fallTime]))
        grad=gradsa(1);
        grad.amplitude = sum([gradsa.amplitude]);
        grad.area = sum([gradsa.area]);
        grad.flatArea = sum([gradsa.flatArea]);
        return;
    end
end

if all(is_trap | is_etrap)
    times=[];
    for ii = 1:length(grads)
        g=grads{ii};
        if is_trap(ii)
            times = [times; cumsum([g.delay; g.riseTime; g.flatTime; g.fallTime])];
        else
            times = [times; g.delay+g.tt];
        end
    end
    times=unique(times);
    dt=times(2:end)-times(1:end-1);
    ieps=find(dt<eps);
    if ~isempty(ieps)
        dtx=[times(1); dt];
        dtx(ieps)=dtx(ieps)+dtx(ieps+1);
        dtx(ieps+1)=[];
        times=cumsum(dtx);
    end
    amplitudes=zeros(size(times));
    for ii = 1:length(grads)
        g=grads{ii};
        if strcmp(g.type,'trap')
            if g.flatTime>0
                g.tt=cumsum([0; g.riseTime; g.flatTime; g.fallTime]);
                g.waveform=[0; g.amplitude; g.amplitude; 0];
            else
                g.tt=cumsum([0; g.riseTime; g.fallTime]);
                g.waveform=[0; g.amplitude; 0];
            end
        end
        tt=g.delay+g.tt;
        [tmin, imin]=min(abs(tt(1)-times));
        if tmin<eps
            tt(1)=times(imin);
        end
        [tmin, imin]=min(abs(tt(end)-times));
        if tmin<eps
            tt(end)=times(imin);
        end
        if abs(g.waveform(1))>eps && tt(1)>eps
            tt(1)=tt(1)+eps;
        end
        amplitudes=amplitudes+interp1(tt,g.waveform,times,'linear',0);
    end
    % FIX: use positional system arg instead of 'system', system name-value
    grad=mr.makeExtendedTrapezoid(channel, system, 'amplitudes',amplitudes,'times',times);
    return;
end

waveforms = {};
max_length = 0;
some_osa=any(is_osa);
if some_osa
    target_raster=system.gradRasterTime/2;
else
    target_raster=system.gradRasterTime;
end
for ii = 1:length(grads)
    g = grads{ii};
    if ~is_trap(ii)
        if is_arb(ii)||is_osa(ii)
            if some_osa && is_arb(ii)
                waveforms{ii} = (g.waveform(floor(1:0.5:end))+g.waveform(ceil(1:0.5:end)))*0.5;
            else
                waveforms{ii} = g.waveform;
            end
        else
            waveforms{ii} = mr.pts2waveform(g.tt, g.waveform, target_raster);
        end
    else
        if (g.flatTime>0)
            times = [g.delay - common_delay; ...
                     g.delay - common_delay + g.riseTime; ...
                     g.delay - common_delay + g.riseTime + g.flatTime; ...
                     g.delay - common_delay + g.riseTime + g.flatTime + g.fallTime];
            amplitudes = [0; g.amplitude; g.amplitude; 0];
        else
            times = [g.delay - common_delay; ...
                     g.delay - common_delay + g.riseTime; ...
                     g.delay - common_delay + g.riseTime + g.fallTime];
            amplitudes = [0; g.amplitude; 0];
        end
        waveforms{ii} = mr.pts2waveform(times, amplitudes, target_raster);
    end
    if size(waveforms{ii},1)==1
        waveforms{ii}=waveforms{ii}';
    end
    if g.delay - common_delay > 0
        t_delay = (0:target_raster:g.delay-common_delay-target_raster).';
        waveforms{ii} = [t_delay*0; waveforms{ii}];
    end
    max_length = max(max_length, length(waveforms{ii}));
end

w = zeros(max_length,1);
for ii = 1:length(grads)
    w(1:length(waveforms{ii})) = w(1:length(waveforms{ii})) + waveforms{ii};
end

grad = mr.makeArbitraryGrad(channel, w, system, ...
                            'maxSlew', maxSlew,...
                            'maxGrad', maxGrad,...
                            'delay', common_delay,...
                            'oversampling',some_osa,...
                            'first',sum(firsts(delays==common_delay)),...
                            'last',sum(lasts(durs==total_duration)));

end
