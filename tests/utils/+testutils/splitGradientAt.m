function [varargout] = splitGradientAt(grad, timepoint, varargin)
% splitGradientAt  Drop-in replacement for mr.splitGradientAt.
%
% Identical to mr.splitGradientAt but fixes internal calls to
% mr.makeExtendedTrapezoid that use 'system' as a name-value pair
% (unsupported in Octave's inputParser with addOptional).

persistent parser

if isempty(parser)
    parser = inputParser;
    parser.FunctionName = 'splitGradientAt';
    parser.addRequired('grad', @isstruct);
    parser.addRequired('timepoint', @isnumeric);
    parser.addOptional('system', [], @isstruct);
end
parse(parser, grad, timepoint, varargin{:});
opt = parser.Results;

if isempty(opt.system)
    system=mr.opts();
else
    system=opt.system;
end

gradRasterTime = system.gradRasterTime;

% round the time point to the gradient raster;
timeindex = round(timepoint / gradRasterTime);
if abs(timepoint-timeindex*gradRasterTime)>1e-6
    warning('splitting the gradint at a point that is not on a gradient raster edge, substantial rounding is applied');
end
timepoint = timeindex * gradRasterTime;
timeindex = timeindex + 1; % convert to Matlab convention

ch = grad.channel;

if strcmp(grad.type, 'grad')
    % check if we have an arbitrary gradient or an exended trapezoid
    if abs(grad.tt(1)-0.5*gradRasterTime)<1e-10
        % it can be an arbitrary gradient or arbitrary gradient with oversampling
        isArb=all(abs(grad.tt(2:end)-grad.tt(1:end-1)-gradRasterTime)<1e-10);
        isArbOs=all(abs(grad.tt(2:end)-grad.tt(1:end-1)-gradRasterTime*0.5)<1e-10);
        if isArb || isArbOs
            if isArbOs
                timeindex = (timeindex-1)*2;
            end
            if timeindex == 1 || timeindex >= length(grad.tt)
                varargout{1} = grad;
            else
                grad1=grad;
                grad2=grad;
                if isArbOs
                    grad1.last=grad.waveform(timeindex);
                else
                    grad1.last=0.5*(grad.waveform(timeindex-1)+grad.waveform(timeindex));
                end
                grad2.first=grad1.last;
                grad2.delay=grad.delay + timepoint;
                grad1.tt=grad.tt(1:(timeindex-1));
                grad1.waveform=grad.waveform(1:(timeindex-1));
                if isArbOs
                    grad2.tt=grad.tt(timeindex+1:end) - timepoint;
                    grad2.waveform=grad.waveform(timeindex+1:end);
                else
                    grad2.tt=grad.tt(timeindex:end) - timepoint;
                    grad2.waveform=grad.waveform(timeindex:end);
                end
                grad1.shape_dur = grad1.tt(end) - grad1.tt(1) + gradRasterTime;
                grad2.shape_dur = grad2.tt(end) - grad2.tt(1) + gradRasterTime;

                if nargout==1
                    varargout{1} = [grad1 grad2];
                else
                    varargout{1} = grad1;
                    varargout{2} = grad2;
                end
            end
            return;
        end
    end

    times      = grad.tt';
    amplitudes = grad.waveform';

elseif strcmp(grad.type, 'trap')
    grad.delay    = round(grad.delay   /gradRasterTime)*gradRasterTime;
    grad.riseTime = round(grad.riseTime/gradRasterTime)*gradRasterTime;
    grad.flatTime = round(grad.flatTime/gradRasterTime)*gradRasterTime;
    grad.fallTime = round(grad.fallTime/gradRasterTime)*gradRasterTime;

    if grad.flatTime == 0
        times      = [0 grad.riseTime  grad.riseTime+grad.fallTime];
        amplitudes = [0 grad.amplitude 0];
    else
        times      = [0 grad.riseTime  grad.riseTime+grad.flatTime grad.riseTime+grad.flatTime+grad.fallTime];
        amplitudes = [0 grad.amplitude grad.amplitude              0];
    end
else
    error('Splitting of unsupported event.');
end

if timepoint >= grad.delay+times(end)
    error('trying to place the splitting time point after the end of the gradient');
end

if timepoint < grad.delay
    times=[0 grad.delay+times];
    amplitudes = [0 amplitudes];
    grad.delay=0;
else
    timepoint = timepoint - grad.delay;
end

amp_tp=interp1(times, amplitudes, timepoint, 'linear');
teps=1e-10;
times1 = [ times(times<timepoint-teps) timepoint ];
amplitudes1 = [ amplitudes(times<timepoint-teps) amp_tp ];
times2 = [ timepoint times(times>timepoint+teps) ] - timepoint;
amplitudes2 = [ amp_tp amplitudes(times>timepoint+teps) ];

% FIX: use positional system arg instead of 'system', system name-value
grad1 = mr.makeExtendedTrapezoid(ch, system, 'times', times1,...
                                  'amplitudes', amplitudes1, ...
                                  'skip_check', true);
grad1.delay = grad.delay;
grad2 = mr.makeExtendedTrapezoid(ch, system, 'times', times2,...
                                  'amplitudes', amplitudes2, ...
                                  'skip_check', true);
grad2.delay = timepoint + grad.delay;

if nargout==1
    varargout{1} = [grad1 grad2];
else
    varargout{1} = grad1;
    varargout{2} = grad2;
end

end
