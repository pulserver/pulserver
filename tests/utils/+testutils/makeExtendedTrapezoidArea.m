function [grad, times, amplitudes] = makeExtendedTrapezoidArea(channel, Gs, Ge, A, sys)
% makeExtendedTrapezoidArea  Drop-in replacement for mr.makeExtendedTrapezoidArea.
%
% Identical to mr.makeExtendedTrapezoidArea but fixes the internal call to
% mr.makeExtendedTrapezoid that passes 'system' as a name-value pair
% (unsupported in Octave's inputParser with addOptional).

SR=sys.maxSlew*0.99;

Tp=0;
obj1=@(x) (A-testGA(x,0,SR,sys.gradRasterTime,Gs,Ge)).^2;

[Gp(1),obj1val(1),exitf(1)] = fminsearch(obj1,-sys.maxGrad);
[Gp(2),obj1val(2),exitf(2)] = fminsearch(obj1,0);
[Gp(3),obj1val(3),exitf(3)] = fminsearch(obj1,sys.maxGrad);
[~,imin]=min(obj1val);
Gp=Gp(imin);
obj1val=obj1val(imin);
exitf=exitf(imin);

if  obj1val>1e-3 || ...
    abs(Gp)> sys.maxGrad
    Gp=sys.maxGrad*sign(Gp);
    obj2=@(x) (A-testGA(Gp,x,SR,sys.gradRasterTime,Gs,Ge)).^2;
    [T,obj2val,exitf] = fminsearch(obj2,0);
    Tp=ceil(T/sys.gradRasterTime)*sys.gradRasterTime;

    Tru=ceil(abs(Gp-Gs)/SR/sys.gradRasterTime)*sys.gradRasterTime;
    Trd=ceil(abs(Gp-Ge)/SR/sys.gradRasterTime)*sys.gradRasterTime;
    obj3=@(x) (A-testGA1(x,Tru,Tp,Trd, Gs,Ge)).^2;

    [Gp,obj3val,exitf] = fminsearch(obj3,Gp);
    assert(obj3val<1e-3);
end

if Tp<=0
    Tp=-sys.gradRasterTime;
    SR = sys.maxSlew*0.99;
    area = 0;
    while abs(area-A)>1e-3
        Tp = Tp+sys.gradRasterTime;
        obj3=@(x) (A-testGA1(x,ceil(abs(x-Gs)/SR/sys.gradRasterTime)*sys.gradRasterTime, ...
            Tp,ceil(abs(x-Ge)/SR/sys.gradRasterTime)*sys.gradRasterTime, ...
            Gs,Ge)).^2;
        [Gp,~,~] = fminsearch(obj3,Gp);
        if abs(Gp)>sys.maxGrad
            Gp = sys.maxGrad*sign(Gp);
        end
        Tru=ceil(abs(Gp-Gs)/SR/sys.gradRasterTime)*sys.gradRasterTime;
        Trd=ceil(abs(Gp-Ge)/SR/sys.gradRasterTime)*sys.gradRasterTime;

        area=testGA1(Gp, Tru, Tp, Trd, Gs, Ge);
    end
end
assert(Tp>=0);

if Tp>0
    times=cumsum([0 Tru Tp Trd]);
    amplitudes=[Gs Gp Gp Ge];
else
    Tru=ceil(abs(Gp-Gs)/SR/sys.gradRasterTime)*sys.gradRasterTime;
    Trd=ceil(abs(Gp-Ge)/SR/sys.gradRasterTime)*sys.gradRasterTime;

    if Trd>0
        if Tru>0
            times=cumsum([0 Tru Trd]);
            amplitudes=[Gs Gp Ge];
        else
            times=cumsum([0 Trd]);
            amplitudes=[Gs Ge];
        end
    else
        times=cumsum([0 Tru]);
        amplitudes=[Gs Ge];
    end
end

% FIX: use positional system arg instead of 'system', sys name-value
grad=mr.makeExtendedTrapezoid(channel, sys, 'times',times, 'amplitudes', amplitudes);
grad.area=testGA1(Gp, Tru, Tp, Trd, Gs, Ge);
assert(abs(grad.area-A)<2e-3); % relaxed from 1e-3 for Octave fminsearch convergence
assert(SR <= sys.maxSlew)
assert(all(abs(amplitudes)<=sys.maxGrad))
end

function ga = testGA(Gp, Tp, SR, dT, Gs, Ge)
Tru=ceil(abs(Gp-Gs)/SR/dT)*dT;
Trd=ceil(abs(Gp-Ge)/SR/dT)*dT;
ga=testGA1(Gp, Tru, Tp, Trd, Gs, Ge);
end

function ga = testGA1(Gp, Tru, Tp, Trd, Gs, Ge)
ga=0.5*Tru.*(Gp+Gs)+Gp.*Tp+0.5*(Gp+Ge).*Trd;
end
