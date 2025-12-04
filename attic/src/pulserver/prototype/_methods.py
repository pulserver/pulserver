__all__ = [
    "pulserver_get_num_blocks_per_tr",
    "pulserver_get_block_groups",
    "pulserver_get_segments",
    "pulserver_PlanningDescriptor",
    "pulserver_RTDescriptor",
    "pulserver_BlockGroup",
    "pulserver_Segment",
    "pulserver_build_planning_descriptors",
    "pulserver_build_rt_descriptors",
    "pulserver_get_num_segments",
    "pulserver_get_num_blocks_in_segment",
    "pulserver_get_block",
]

import numpy as np


# %% utils
def pulserver_get_num_blocks_per_tr(sequences):
    num_blocks_per_tr = []
    for seq in sequences:
        found, count = 0, 0
        while found < 2:
            b = seq.get_block(count + 1)
            found += hasattr(b, "label") and "TRID" in [
                lbl.label for lbl in b.label.values()
            ]
            count += 1
        num_blocks_per_tr.append(count - 1)
    return num_blocks_per_tr


class pulserver_BlockGroup:
    def __init__(self):
        self.label = (
            None  # BLOCKID label value (identifies base block invariant params)
        )
        self.start = None  # starting block index (0-based) in TR
        self.size = None  # number of blocks in group


class pulserver_Segment:
    def __init__(self):
        self.label = None  # COREID label value (identifies segment)
        self.start = None  # starting block index (0-based) in TR
        self.size = None  # number of blocks in segment


class pulserver_PlanningDescriptor:
    """
    Full descriptor for planning time: enables building segments with concatenation.
    - num_blocks: int
    - block_groups: list of pulserver_BlockGroup
    - segments: list of pulserver_Segment
    - block2group: list[int] mapping block index -> block_group index
    - block2segment: list[int] mapping block index -> segment index
    - group2segment: list[int] mapping block_group index -> segment index (groups must be fully inside one segment)
    - segment2groups: list[list[int]] mapping segment index -> list of block_group indices (in traversal order)
    - block_groups_table: list of BLOCKID labels in block traversal order
    - segments_table: list of COREID labels in block traversal order
    """

    def __init__(self):
        self.num_blocks = 0
        self.block_groups = []
        self.segments = []
        self.block2group = []
        self.block2segment = []
        self.group2segment = []
        self.segment2groups = []
        self.block_groups_table = []
        self.segments_table = []


class pulserver_RTDescriptor:
    """
    Minimal descriptor for RT: just what's needed for sequential segment playback and dynamics.
    - num_segments: int
    - segment_labels: list of COREID labels in playback order
    - segment_group_counts: list[int] num groups per segment (for dynamics loop if needed)
    - group_block_ranges: list[tuple[int, int]] (start_block, size) per group (for last-block dynamics)
    """

    def __init__(self):
        self.num_segments = 0
        self.segment_labels = []
        self.segment_group_counts = []
        self.group_block_ranges = []


def pulserver_get_block_groups(sequences, num_blocks_per_tr):
    num_block_groups_per_tr = []
    block_group_labels = []
    block_group_starts = []
    block_group_ends = []
    block_group_sizes = []
    block_groups_table = []

    for n, seq in enumerate(sequences):
        num_block_groups_per_tr.append(0)
        block_group_labels.append([])
        block_group_starts.append([])
        block_group_ends.append([])
        block_group_sizes.append([])
        block_groups_table.append([])
        for b in range(num_blocks_per_tr[n]):
            block = seq.get_block(b + 1)

            # First block
            if b == 0:
                num_block_groups_per_tr[n] += 1
                block_label = None
                for lbl in block.label.values():
                    if lbl.label == "BLOCKID":
                        block_label = lbl.value
                if block_label is None:
                    raise RuntimeError(
                        "First block in sequence is a parent block by definition and must have a label"
                    )
                current_label = block_label
                block_group_starts[n].append(b)
                block_groups_table[n].append(block_label)
                continue
            else:
                block_label = current_label  # default
                if hasattr(block, "label"):
                    for lbl in block.label.values():
                        if lbl.label == "BLOCKID":
                            block_label = lbl.value

            # Second to last block in TR
            if block_label != current_label:
                block_groups_table[n].append(block_label)

                # If previous block is new, add its end and its label to the list
                if current_label not in block_group_labels[n]:
                    block_group_labels[n].append(current_label)
                    block_group_ends[n].append(b)

                # Update current block
                current_label = block_label

                # If current block is new, add its start to the list
                if current_label not in block_group_labels[n]:
                    num_block_groups_per_tr[n] += 1
                    block_group_starts[n].append(b)

            # Last block in TR
            if b == num_blocks_per_tr[n] - 1:
                if current_label not in block_group_labels[n]:
                    block_group_labels[n].append(current_label)
                    block_group_ends[n].append(b + 1)

    # Get Group sizes
    for n, num_block_groups in enumerate(num_block_groups_per_tr):
        for b in range(num_block_groups):
            block_group_sizes[n].append(
                block_group_ends[n][b] - block_group_starts[n][b]
            )

    # Transform to structs
    block_groups = []
    for n, num_block_groups in enumerate(num_block_groups_per_tr):
        block_groups.append([])
        for b in range(num_block_groups):
            block_groups[n].append(pulserver_BlockGroup())
            block_groups[n][b].label = block_group_labels[n][b]
            block_groups[n][b].start = block_group_starts[n][b]
            block_groups[n][b].size = block_group_sizes[n][b]

    return block_groups, block_groups_table


def pulserver_get_segments(sequences, num_blocks_per_tr):
    num_segments_per_tr = []
    segment_labels = []
    segment_starts = []
    segment_ends = []
    segment_sizes = []
    segments_table = []

    for n, seq in enumerate(sequences):
        num_segments_per_tr.append(0)
        segment_labels.append([])
        segment_starts.append([])
        segment_ends.append([])
        segment_sizes.append([])
        segments_table.append([])
        for b in range(num_blocks_per_tr[n]):
            block = seq.get_block(b + 1)

            # First block
            if b == 0:
                num_segments_per_tr[n] += 1
                segment_label = None
                for lbl in block.label.values():
                    if lbl.label == "COREID":
                        segment_label = lbl.value
                if segment_label is None:
                    raise RuntimeError(
                        "First block in sequence is a segment start by definition and must have a label"
                    )
                current_label = segment_label
                segment_starts[n].append(b)
                segments_table[n].append(segment_label)
                continue
            else:
                segment_label = current_label  # default
                if hasattr(block, "label"):
                    for lbl in block.label.values():
                        if lbl.label == "COREID":
                            segment_label = lbl.value

            # Second to last block in TR
            if segment_label != current_label:
                segments_table[n].append(segment_label)

                # If previous block is new, add its end and its label to the list
                if current_label not in segment_labels[n]:
                    segment_labels[n].append(current_label)
                    segment_ends[n].append(b)

                # Update current block
                current_label = segment_label

                # If current block is new, add its start to the list
                if current_label not in segment_labels[n]:
                    num_segments_per_tr[n] += 1
                    segment_starts[n].append(b)

            # Last block in TR
            if b == num_blocks_per_tr[n] - 1:
                if current_label not in segment_labels[n]:
                    segment_labels[n].append(current_label)
                    segment_ends[n].append(b + 1)

    # Get Group sizes
    for n, num_segments in enumerate(num_segments_per_tr):
        for b in range(num_segments):
            segment_sizes[n].append(segment_ends[n][b] - segment_starts[n][b])

    # Transform to structs
    segments = []
    for n, num_segments in enumerate(num_segments_per_tr):
        segments.append([])
        for b in range(num_segments):
            segments[n].append(pulserver_Segment())
            segments[n][b].label = segment_labels[n][b]
            segments[n][b].start = segment_starts[n][b]
            segments[n][b].size = segment_sizes[n][b]

    return segments, segments_table


# %% New: separate builders for planning and RT
def pulserver_build_planning_descriptors(sequences):
    """
    Build full planning descriptors for each sequence (TR).
    Includes all mappings for building segments with concatenation.
    Ensures groups do not cross segment boundaries.
    """
    descriptors = []
    num_blocks_per_tr = pulserver_get_num_blocks_per_tr(sequences)
    block_groups_all, block_groups_table = pulserver_get_block_groups(
        sequences, num_blocks_per_tr
    )
    segments_all, segments_table = pulserver_get_segments(sequences, num_blocks_per_tr)

    for n, seq in enumerate(sequences):
        desc = pulserver_PlanningDescriptor()
        desc.num_blocks = num_blocks_per_tr[n]
        desc.block_groups = block_groups_all[n]
        desc.segments = segments_all[n]
        desc.block_groups_table = block_groups_table[n]
        desc.segments_table = segments_table[n]

        # initialize mappings
        desc.block2group = [-1] * desc.num_blocks
        desc.block2segment = [-1] * desc.num_blocks
        desc.group2segment = [-1] * len(desc.block_groups)
        desc.segment2groups = [[] for _ in desc.segments]

        # map blocks -> group
        for g_idx, g in enumerate(desc.block_groups):
            g_start = g.start
            g_end = g.start + g.size
            if g_start < 0:
                g_start = 0
            if g_end > desc.num_blocks:
                g_end = desc.num_blocks
            for b in range(g_start, g_end):
                desc.block2group[b] = g_idx

        # map blocks -> segment
        for s_idx, s in enumerate(desc.segments):
            s_start = s.start
            s_end = s.start + s.size
            if s_start < 0:
                s_start = 0
            if s_end > desc.num_blocks:
                s_end = desc.num_blocks
            for b in range(s_start, s_end):
                desc.block2segment[b] = s_idx

        # group -> segment: segment of first block (assume groups don't cross)
        for g_idx, g in enumerate(desc.block_groups):
            first_block = g.start
            if 0 <= first_block < desc.num_blocks:
                desc.group2segment[g_idx] = desc.block2segment[first_block]

        # segment -> groups
        for g_idx, seg_idx in enumerate(desc.group2segment):
            if seg_idx >= 0:
                desc.segment2groups[seg_idx].append(g_idx)

        # safety: verify groups do not cross segment boundaries
        for g_idx, g in enumerate(desc.block_groups):
            seg_idx = desc.group2segment[g_idx]
            if seg_idx < 0:
                continue
            s = desc.segments[seg_idx]
            g_end = g.start + g.size
            if not (s.start <= g.start and g_end <= s.start + s.size):
                raise RuntimeError(
                    f"BlockGroup {g_idx} (blocks {g.start}-{g_end-1}) crosses Segment {seg_idx} (blocks {s.start}-{s.start+s.size-1}) boundaries"
                )

        # final sanity: ensure every block has group mapping
        for b in range(desc.num_blocks):
            if desc.block2group[b] == -1:
                # create singleton group
                ng = pulserver_BlockGroup()
                ng.label = None
                ng.start = b
                ng.size = 1
                desc.block_groups.append(ng)
                new_g_idx = len(desc.block_groups) - 1
                desc.block2group[b] = new_g_idx
                sidx = desc.block2segment[b]
                desc.group2segment.append(sidx)
                if sidx >= 0:
                    desc.segment2groups[sidx].append(new_g_idx)

        descriptors.append(desc)

    return descriptors


def pulserver_build_rt_descriptors(sequences):
    """
    Build minimal RT descriptors for each sequence (TR) directly from sequences.
    Extracts only data needed for RT: segment order and group ranges for dynamics.
    Assumes groups do not cross segments (no full validation here for RT efficiency).
    """
    descriptors = []
    num_blocks_per_tr = pulserver_get_num_blocks_per_tr(sequences)
    block_groups_all, _ = pulserver_get_block_groups(sequences, num_blocks_per_tr)
    segments_all, _ = pulserver_get_segments(sequences, num_blocks_per_tr)

    for n in range(len(sequences)):
        rt_desc = pulserver_RTDescriptor()
        rt_desc.num_segments = len(segments_all[n])
        rt_desc.segment_labels = [s.label for s in segments_all[n]]
        # Count groups per segment: need to map groups to segments minimally
        segment_group_counts = [0] * rt_desc.num_segments
        for g in block_groups_all[n]:
            # Find segment containing group's first block
            first_block = g.start
            seg_idx = -1
            for s_idx, s in enumerate(segments_all[n]):
                if s.start <= first_block < s.start + s.size:
                    seg_idx = s_idx
                    break
            if seg_idx >= 0:
                segment_group_counts[seg_idx] += 1
        rt_desc.segment_group_counts = segment_group_counts
        rt_desc.group_block_ranges = [(g.start, g.size) for g in block_groups_all[n]]
        descriptors.append(rt_desc)

    return descriptors


# %% Pulse generation API
def pulserver_get_num_segments(desc):
    """
    Get the number of segments in the planning descriptor.
    """
    return len(desc.segments)


def pulserver_get_num_blocks_in_segment(desc, segment_idx):
    """
    Get the number of blocks in the specified segment.
    """
    if 0 <= segment_idx < len(desc.segments):
        return desc.segments[segment_idx].size
    else:
        raise IndexError("Invalid segment index")


def pulserver_get_block(desc, seq, segment_idx, block_idx_in_segment):
    """
    Get the block (rf, gx, gy, gz, adc events) from the sequence, given segment and block index within segment.
    """
    if not (0 <= segment_idx < len(desc.segments)):
        raise IndexError("Invalid segment index")
    seg = desc.segments[segment_idx]
    if not (0 <= block_idx_in_segment < seg.size):
        raise IndexError("Invalid block index in segment")
    block_index_in_tr = seg.start + block_idx_in_segment
    return seq.get_block(block_index_in_tr + 1)  # +1 for 1-based


def pulserver_get_block_duration(block):
    return block.duration


class pulserver_RFEvent:
    def __init__(self):
        self.defined = False
        self.delay = 0
        self.magnitude_signal = None
        self.phase_signal = None


def pulserver_get_block_rf(block):
    rf = pulserver_RFEvent()
    if block.rf is not None:
        rf.defined = True
        rf.delay = block.rf.delay
        rf.magnitude_signal = np.abs(block.rf.signal)
        rf.phase_signal = np.angle(block.rf.signal)


def pulserver_get_block_gx(block):
    gx = pulserver_GradEvent()
    if block.gx is not None:
        gx.defined = True
        gx.delay = block.rf.delay
        gx.magnitude_signal = np.abs(block.rf.signal)
        gx.phase_signal = np.angle(block.rf.signal)


# %% Planning stage


# Pre-download
# 1) Get echo filters
def pulserver_get_adc(adc_library): ...


def pulserver_get_rf(sequences): ...


# Pulsegen

# In Predownload
#

# In Pulsegen:
#
# 1) loop over segment definitions (for each TR)
# 2) for each block group in segment definitions get the concatenated
#    rf, gx, gy, gz waveforms, the ADC filter indexes and each event time axis
# 3) use the above to build the segments - initializing each waveform to worst case

# %% Real Time stage
