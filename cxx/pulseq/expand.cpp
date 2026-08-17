#include "pulseq/expand.hpp"

#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "pulseq/sequence.hpp"

namespace pulseq
{

    namespace
    {

        /**
         * The `ONCE` value in force at every block, and whether the counter
         * about to be written is already spoken for.
         *
         * Both answers come off the same walk because both are questions about
         * the extension chains, and a scan has one chain per block: asking
         * twice would be two passes over the largest thing here.
         *
         * `ONCE` is sticky -- a `LABELSET` applies to the block that carries it
         * and to every block after it -- so this is a running value rather than
         * a per-block lookup.  Only the label libraries are consulted for the
         * counter, but through the chains rather than by scanning the library:
         * a row nobody points at is not a claim on the counter, and refusing on
         * one would be refusing over a leftover.
         */
        struct Scan
        {
            /** 1-based; index 0 is unused. */
            std::vector<int> once;
            bool counter_written = false;
        };

        Scan scan_blocks(const Sequence& seq, int once_id, int counter_id, int labelset,
                         int labelinc)
        {
            const int n = seq.num_blocks();
            Scan out;
            out.once.assign(static_cast<size_t>(n) + 1, 0);

            const int32_t* events = seq.block_events();
            int current = 0;
            for (int b = 1; b <= n; ++b)
            {
                for (int32_t link = events[static_cast<size_t>(b - 1) * BLOCK_WIDTH + 5];
                     link != 0;)
                {
                    const int32_t* row = seq.extensions_library().row(link);
                    if (row[1] != 0)
                    {
                        if (labelset != 0 && row[0] == labelset)
                        {
                            const int32_t* entry = seq.label_set_library().row(row[1]);
                            if (once_id != 0 && entry[1] == once_id)
                                current = entry[0];
                            if (counter_id != 0 && entry[1] == counter_id)
                                out.counter_written = true;
                        }
                        else if (labelinc != 0 && row[0] == labelinc)
                        {
                            const int32_t* entry = seq.label_inc_library().row(row[1]);
                            if (once_id != 0 && entry[1] == once_id)
                                current += entry[0];
                            if (counter_id != 0 && entry[1] == counter_id)
                                out.counter_written = true;
                        }
                    }
                    link = row[2];
                }
                out.once[static_cast<size_t>(b)] = current;
            }
            return out;
        }

        /**
         * The chain @p head with its `ONCE` links removed.
         *
         * Everything else survives in its original order -- a rotation on a
         * readout block is not this pass's to discard -- so the chain is
         * collected head-first and relinked in reverse.  `chain_extension`
         * searches, which is what collapses the handful of distinct chains a
         * scan really has back onto themselves.
         */
        int32_t strip_once_links(Sequence& seq, int32_t head, int once_id, int labelset,
                                 int labelinc)
        {
            std::vector<std::pair<int32_t, int32_t>> keep;
            for (int32_t link = head; link != 0;)
            {
                const int32_t* row = seq.extensions_library().row(link);
                const int32_t type = row[0];
                const int32_t ref = row[1];
                const int32_t next = row[2];
                bool mine = false;
                if (ref != 0)
                {
                    if (labelset != 0 && type == labelset)
                        mine = seq.label_set_library().row(ref)[1] == once_id;
                    else if (labelinc != 0 && type == labelinc)
                        mine = seq.label_inc_library().row(ref)[1] == once_id;
                }
                if (!mine)
                    keep.emplace_back(type, ref);
                link = next;
            }

            int32_t out = 0;
            for (size_t k = keep.size(); k-- > 0;)
                out = static_cast<int32_t>(seq.chain_extension(keep[k].first, keep[k].second, out));
            return out;
        }

    }  // namespace

    ExpandResult expand_repeats(Sequence& seq, int repeats, const ExpandOptions& options)
    {
        if (repeats < 1)
            throw std::invalid_argument("expand_repeats(): repeats must be at least 1, got " +
                                        std::to_string(repeats));

        const int n = seq.num_blocks();
        const int labelset = seq.find_extension_type_id("LABELSET");
        const int labelinc = seq.find_extension_type_id("LABELINC");
        const int once_id = seq.find_label_id("ONCE");

        /* Reserved before the scan so the scan can recognise it, but only
         * registered on a sequence that is going to carry it. */
        int counter_id = 0;
        if (!options.label.empty() && repeats > 1)
            counter_id = seq.label_id(options.label);

        const Scan scan = scan_blocks(seq, once_id, counter_id, labelset, labelinc);

        if (scan.counter_written)
            throw std::runtime_error("expand_repeats(): the sequence already writes " +
                                     options.label +
                                     "; it would mean two things at once.  Label the repetitions "
                                     "with a different counter, or pass an empty label to leave "
                                     "them unlabelled.");

        ExpandResult result;
        result.repeats = repeats;
        result.blocks_before = n;

        for (int b = 1; b <= n; ++b)
        {
            const int state = scan.once[static_cast<size_t>(b)];
            if (state < 0 || state > 2)
                throw std::runtime_error("expand_repeats(): block " + std::to_string(b) +
                                         " carries ONCE=" + std::to_string(state) +
                                         "; ONCE is a three-state flag (0, 1 or 2)");
            if (state == 1)
                ++result.prep_blocks;
            else if (state == 2)
                ++result.cooldown_blocks;
            else
                ++result.body_blocks;
        }

        if (repeats > 1 && result.body_blocks == 0)
            throw std::runtime_error(
                "expand_repeats(): every block is marked ONCE=1 or ONCE=2, so there is no body to "
                "repeat.  ONCE=0 is what plays on every repetition.");

        /*
         * Chains first, so the emission loop is a lookup.
         *
         * There are few distinct chains in a scan and many blocks, and
         * `chain_extension` appends -- doing this inside the loop would rebuild
         * the same chain once per repetition of every block that carries it.
         */
        std::map<int32_t, int32_t> stripped;
        if (options.strip_once && once_id != 0 && (labelset != 0 || labelinc != 0))
        {
            const int32_t* events = seq.block_events();
            for (int b = 1; b <= n; ++b)
            {
                const int32_t head = events[static_cast<size_t>(b - 1) * BLOCK_WIDTH + 5];
                if (head != 0 && stripped.find(head) == stripped.end())
                    stripped[head] =
                        strip_once_links(seq, head, once_id, labelset, labelinc);
            }
        }

        /* One LABELSET row per repetition index, reused across the blocks that
         * would otherwise each register their own. */
        std::map<int32_t, int> counter_rows;
        const int32_t labelset_out =
            (counter_id != 0) ? static_cast<int32_t>(seq.extension_type_id("LABELSET")) : 0;

        const long long total = static_cast<long long>(repeats) * result.body_blocks +
                                result.prep_blocks + result.cooldown_blocks;

        std::vector<int32_t> events;
        std::vector<double> durations;
        events.reserve(static_cast<size_t>(total) * BLOCK_WIDTH);
        durations.reserve(static_cast<size_t>(total));

        for (int pass = 0; pass < repeats; ++pass)
        {
            const bool first = (pass == 0);
            const bool last = (pass == repeats - 1);
            bool stamped = (counter_id == 0 || pass == 0);

            for (int b = 1; b <= n; ++b)
            {
                const int state = scan.once[static_cast<size_t>(b)];
                if (state == 1 && !first)
                    continue;
                if (state == 2 && !last)
                    continue;

                /* Refetched every block: the source table is `seq`'s own, and
                 * registering a label row below can move it. */
                const Block block = seq.get_block(b);
                int32_t head = block.ext;
                if (options.strip_once && head != 0)
                {
                    const std::map<int32_t, int32_t>::const_iterator it = stripped.find(head);
                    if (it != stripped.end())
                        head = it->second;
                }

                if (!stamped)
                {
                    const int32_t value = static_cast<int32_t>(pass);
                    std::map<int32_t, int>::iterator row = counter_rows.find(value);
                    if (row == counter_rows.end())
                        row = counter_rows
                                  .emplace(value, seq.register_label_set(
                                                      value, static_cast<int32_t>(counter_id)))
                                  .first;
                    head = static_cast<int32_t>(
                        seq.chain_extension(labelset_out, static_cast<int32_t>(row->second), head));
                    stamped = true;
                }

                events.push_back(block.rf);
                events.push_back(block.gx);
                events.push_back(block.gy);
                events.push_back(block.gz);
                events.push_back(block.adc);
                events.push_back(head);
                durations.push_back(block.duration);
            }
        }

        result.blocks_after = static_cast<int>(durations.size());
        seq.set_blocks(events.data(), durations.data(), result.blocks_after);

        return result;
    }

}  // namespace pulseq
