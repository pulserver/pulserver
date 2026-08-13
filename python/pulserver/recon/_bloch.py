"""TorchSim EPG consumers for decoded SEQDESC event streams."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from ._seqdesc import AdcRole, EventType, RfUse, SequenceDescription, SequenceEvent

__all__ = [
    "BSSFP",
    "FSE",
    "SPGR",
    "SSFPFID",
    "EpgInterpreter",
    "SSFPEcho",
    "SimulationResult",
    "SubspaceBasis",
    "TissueProperties",
    "make_interpreter",
    "simulate_subspace",
]

RecordMode = Literal["all", "acquired", "echo"]


@dataclass(frozen=True)
class TissueProperties:
    """Single-pool tissue parameters accepted by the EPG interpreter.

    Values may be scalars, NumPy arrays, or Torch tensors. Broadcasted entries
    are simulated in parallel and form the dictionary columns used for
    subspace estimation.
    """

    t1_ms: Any
    t2_ms: Any
    m0: Any = 1.0
    b1: Any = 1.0
    b0_hz: Any = 0.0


@dataclass(frozen=True)
class SimulationResult:
    """Signals recorded by one state-machine simulation."""

    signal: Any
    time_us: Any
    event_index: Any
    repetition: Any
    echo: Any


@dataclass(frozen=True)
class SubspaceBasis:
    """Low-rank temporal basis and the dictionary from which it was estimated."""

    basis: Any
    singular_values: Any
    dictionary: Any
    simulation: SimulationResult


class EpgInterpreter:
    """Base policy for translating RF/ADC events into TorchSim EPG operations."""

    name = "base"

    def before_rf(self, states: Any, _event: SequenceEvent, _epg: Any) -> Any:
        """Apply sequence-specific operations immediately before RF."""

        return states

    def after_rf(self, states: Any, _event: SequenceEvent, _epg: Any) -> Any:
        """Apply sequence-specific operations immediately after RF."""

        return states

    def before_adc(self, states: Any, _event: SequenceEvent, _epg: Any) -> Any:
        """Apply sequence-specific operations immediately before ADC."""

        return states

    def after_adc(self, states: Any, _event: SequenceEvent, _epg: Any) -> Any:
        """Apply sequence-specific operations immediately after ADC."""

        return states

    def shifts_per_repetition(self, _description: SequenceDescription) -> int:
        """Upper bound used to size the EPG state matrix."""

        return 0

    def simulate(
        self,
        description: SequenceDescription,
        tissue: TissueProperties,
        *,
        repetitions: int = 1,
        record: RecordMode = "all",
        nstates: int | None = None,
        rf_raster_time_s: float = 1e-6,
        device: Any = None,
    ) -> SimulationResult:
        """Walk a decoded event stream and record selected ADC signals."""

        if repetitions < 1:
            raise ValueError("repetitions must be positive")
        if record not in {"all", "acquired", "echo"}:
            raise ValueError("record must be 'all', 'acquired', or 'echo'")

        torch, epg = _torchsim()
        device = torch.device("cpu") if device is None else torch.device(device)
        t1, t2, m0, b1, b0 = _prepare_tissue(torch, tissue, device)
        if torch.any(t1 <= 0) or torch.any(t2 <= 0):
            raise ValueError("T1 and T2 must be positive")

        minimum_states = 1 + repetitions * self.shifts_per_repetition(description)
        if nstates is None:
            nstates = max(8, minimum_states)
        if nstates < minimum_states:
            raise ValueError(
                f"nstates={nstates} is too small for {minimum_states - 1} crusher shifts"
            )

        states = epg.states_matrix(device=device, nstates=nstates, nlocs=t1.numel())
        states.Z[0, :, 0] = m0

        signals = []
        times = []
        event_indices = []
        repetition_indices = []
        echo_flags = []
        absolute_offset_us = 0.0

        for repetition in range(repetitions):
            current_us = 0.0
            for event_index, event in enumerate(description.events):
                if event.timestamp_us < current_us:
                    raise ValueError("SEQDESC event timestamps must be nondecreasing")
                states = _free_precess(
                    torch,
                    epg,
                    states,
                    t1,
                    t2,
                    b0,
                    (event.timestamp_us - current_us) * 1e-6,
                )
                current_us = event.timestamp_us

                if event.type is EventType.RF:
                    states = self.before_rf(states, event, epg)
                    definition = description.rf_definitions[event.rf_definition_id]
                    flip, integral_phase = definition.flip_angle(
                        event.rf_amplitude_hz,
                        rf_raster_time_s=rf_raster_time_s,
                    )
                    phase = event.rf_phase_rad + integral_phase
                    operator = epg.phased_rf_pulse_op(
                        torch.as_tensor(flip, dtype=torch.float32, device=device),
                        torch.as_tensor(phase, dtype=torch.float32, device=device),
                        B1=b1[None, :],
                    )
                    states = epg.rf_pulse(states, operator)
                    states = self.after_rf(states, event, epg)
                elif event.type is EventType.ADC:
                    states = self.before_adc(states, event, epg)
                    if _record_event(event, record):
                        signal = states.Fplus[0, :, 0] * torch.exp(
                            -1j
                            * torch.as_tensor(
                                event.adc_phase_rad,
                                dtype=torch.float32,
                                device=device,
                            )
                        )
                        signals.append(signal)
                        times.append(absolute_offset_us + event.timestamp_us)
                        event_indices.append(event_index)
                        repetition_indices.append(repetition)
                        echo_flags.append(event.is_echo)
                    states = self.after_adc(states, event, epg)

            tail_us = description.tr_duration_us - current_us
            if tail_us < -1e-3:
                raise ValueError(
                    "last SEQDESC event lies after the declared TR duration"
                )
            states = _free_precess(
                torch,
                epg,
                states,
                t1,
                t2,
                b0,
                max(tail_us, 0.0) * 1e-6,
            )
            absolute_offset_us += description.tr_duration_us

        if signals:
            signal_tensor = torch.stack(signals)
        else:
            signal_tensor = torch.empty(
                (0, t1.numel()), dtype=torch.complex64, device=device
            )
        return SimulationResult(
            signal=signal_tensor,
            time_us=torch.as_tensor(times, dtype=torch.float32, device=device),
            event_index=torch.as_tensor(
                event_indices, dtype=torch.int64, device=device
            ),
            repetition=torch.as_tensor(
                repetition_indices, dtype=torch.int64, device=device
            ),
            echo=torch.as_tensor(echo_flags, dtype=torch.bool, device=device),
        )


class FSE(EpgInterpreter):
    """Fast spin echo: crusher -> refocusing RF -> crusher."""

    name = "fse"

    def before_rf(self, states: Any, event: SequenceEvent, epg: Any) -> Any:
        if event.rf_use is RfUse.REFOCUSING:
            return epg.shift(states)
        return states

    def after_rf(self, states: Any, event: SequenceEvent, epg: Any) -> Any:
        if event.rf_use is RfUse.REFOCUSING:
            return epg.shift(states)
        return states

    def shifts_per_repetition(self, description: SequenceDescription) -> int:
        return 2 * sum(
            event.type is EventType.RF and event.rf_use is RfUse.REFOCUSING
            for event in description.events
        )


class SPGR(EpgInterpreter):
    """Spoiled GRE: record the ADC, then apply ideal transverse spoiling."""

    name = "spgr"

    def after_adc(self, states: Any, _event: SequenceEvent, epg: Any) -> Any:
        return epg.spoil(states)


class SSFPFID(EpgInterpreter):
    """Gradient-spoiled SSFP-FID: record the ADC, then apply a crusher."""

    name = "ssfp-fid"

    def after_adc(self, states: Any, _event: SequenceEvent, epg: Any) -> Any:
        return epg.shift(states)

    def shifts_per_repetition(self, description: SequenceDescription) -> int:
        return sum(event.type is EventType.ADC for event in description.events)


class SSFPEcho(EpgInterpreter):
    """SSFP-Echo: apply the dephasing crusher immediately before the ADC."""

    name = "ssfp-echo"

    def before_adc(self, states: Any, _event: SequenceEvent, epg: Any) -> Any:
        return epg.shift(states)

    def shifts_per_repetition(self, description: SequenceDescription) -> int:
        return sum(event.type is EventType.ADC for event in description.events)


class BSSFP(EpgInterpreter):
    """Balanced SSFP: no crusher is attached to RF or ADC events."""

    name = "bssfp"


def make_interpreter(name: str) -> EpgInterpreter:
    """Construct one of the built-in sequence policies."""

    normalized = name.lower().replace("_", "-")
    policies = {
        "fse": FSE,
        "spgr": SPGR,
        "ssfp-fid": SSFPFID,
        "ssfp-echo": SSFPEcho,
        "bssfp": BSSFP,
    }
    try:
        return policies[normalized]()
    except KeyError as exc:
        raise ValueError(f"unknown EPG interpreter {name!r}") from exc


def simulate_subspace(
    description: SequenceDescription,
    interpreter: EpgInterpreter | str,
    tissue: TissueProperties,
    *,
    rank: int,
    repetitions: int = 1,
    record: RecordMode = "all",
    nstates: int | None = None,
    rf_raster_time_s: float = 1e-6,
    device: Any = None,
) -> SubspaceBasis:
    """Simulate a signal dictionary and return its leading temporal basis."""

    if rank < 1:
        raise ValueError("rank must be positive")
    if isinstance(interpreter, str):
        interpreter = make_interpreter(interpreter)
    result = interpreter.simulate(
        description,
        tissue,
        repetitions=repetitions,
        record=record,
        nstates=nstates,
        rf_raster_time_s=rf_raster_time_s,
        device=device,
    )
    if rank > min(result.signal.shape):
        raise ValueError(
            f"rank={rank} exceeds dictionary dimensions {tuple(result.signal.shape)}"
        )
    torch, _ = _torchsim()
    basis, singular_values, _ = torch.linalg.svd(result.signal, full_matrices=False)
    return SubspaceBasis(
        basis=basis[:, :rank],
        singular_values=singular_values,
        dictionary=result.signal,
        simulation=result,
    )


def _record_event(event: SequenceEvent, mode: RecordMode) -> bool:
    if mode == "all":
        return True
    if mode == "acquired":
        return event.adc_role is not AdcRole.NON_ACQUIRED
    return event.is_echo


def _torchsim() -> tuple[Any, Any]:
    try:
        import torch
        from torchsim import epg
    except ImportError as exc:
        raise ImportError(
            "SEQDESC EPG simulation requires TorchSim; install pulserver[recon-sim]"
        ) from exc
    return torch, epg


def _prepare_tissue(
    torch: Any,
    tissue: TissueProperties,
    device: Any,
) -> tuple[Any, Any, Any, Any, Any]:
    values = [
        torch.as_tensor(value, dtype=torch.float32, device=device).reshape(-1)
        for value in (
            tissue.t1_ms,
            tissue.t2_ms,
            tissue.m0,
            tissue.b1,
            tissue.b0_hz,
        )
    ]
    return tuple(value.contiguous() for value in torch.broadcast_tensors(*values))


def _free_precess(
    torch: Any,
    epg: Any,
    states: Any,
    t1_ms: Any,
    t2_ms: Any,
    b0_hz: Any,
    duration_s: float,
) -> Any:
    if duration_s <= 0.0:
        return states
    duration = torch.as_tensor(duration_s, dtype=torch.float32, device=t1_ms.device)
    e1, recovery = epg.longitudinal_relaxation_op(1e3 / t1_ms[None, :, None], duration)
    e2 = epg.transverse_relaxation_op(1e3 / t2_ms[None, :, None], duration)
    states = epg.longitudinal_relaxation(states, e1, recovery)
    states = epg.transverse_relaxation(states, e2)

    phase = torch.exp(-1j * 2.0 * torch.pi * b0_hz[None, :, None] * duration)
    states.Fplus = states.Fplus * phase
    states.Fminus = states.Fminus * phase.conj()
    return states
