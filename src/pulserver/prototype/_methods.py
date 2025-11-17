
__all__ = ["pulserver_get_num_blocks_per_tr", "pulserver_get_block_groups"]

#%% utils
def pulserver_get_num_blocks_per_tr(sequences):
    num_blocks_per_tr = []
    for seq in sequences:
        found, count = 0, 0
        while found < 2:
            b = seq.get_block(count+1)
            found += hasattr(b, 'label') and 'TRID' in [lbl.label for lbl in b.label.values()]
            count += 1
        num_blocks_per_tr.append(count-1)
    return num_blocks_per_tr

class pulserver_BlockGroup:
    def __init__(self):
        self.label = None
        self.start = None
        self.size = None

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
            block = seq.get_block(b+1)
            
            # First block
            if b == 0:
                num_block_groups_per_tr[n] += 1
                block_label = None
                for lbl in block.label.values():
                    if lbl.label == 'BLOCKID':
                        block_label = lbl.value
                if block_label is None:
                    raise RuntimeError("First block in sequence is a parent block by definition and must have a label")
                current_label = block_label
                block_group_starts[n].append(b)
                block_groups_table[n].append(block_label)
                continue
            else:
                block_label = current_label # default
                if hasattr(block, 'label'):
                    for lbl in block.label.values():
                        if lbl.label == 'BLOCKID':
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
                    block_group_ends[n].append(b+1)
                    
    # Get Group sizes
    for n, num_block_groups in enumerate(num_block_groups_per_tr):
        for b in range(num_block_groups):
            block_group_sizes[n].append(block_group_ends[n][b] - block_group_starts[n][b])

    # Transform to structs
    block_groups = []
    for n, num_block_groups in enumerate(num_block_groups_per_tr):
        block_groups.append([])
        for b in range(num_block_groups):
            block_groups[n].append(pulserver_BlockGroup())
            block_groups[n][b].label = block_group_labels[n][b]
            block_groups[n][b].start = block_group_starts[n][b]
            block_groups[n][b].size = block_group_sizes[n][b]
            
    return  block_groups, block_groups_table 
            
# %% Planning stage

# Pre-download
# 1) Get echo filters
def pulserver_get_adc(adc_library):
    ...            
    
def pulserver_get_rf(sequences):
    ...
    
# Pulsegen
def pulserver_get_segments(sequences):
    ...


# %% Real Time stage