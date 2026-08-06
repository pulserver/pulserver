/**
 * @file dedup.cpp
 * @brief Sequence::remove_duplicates() -- collapse every library onto its
 *        distinct rows and renumber everything that points at them.
 *
 * ### What decides that two events are one event
 *
 * The precision the `.seq` file writes the column at, and nothing else.  A
 * trapezoid's amplitude is written `%12g`, so two amplitudes agreeing to six
 * significant digits are written identically and cannot be two rows; its ramp
 * times are written as whole microseconds, so two delays agreeing to 1e-6 s
 * are the same delay.  Rounding to that precision therefore costs nothing --
 * the file could not have carried the difference -- and it is what turns the
 * six million gradient rows of a 3D protocol into the couple of thousand the
 * scan actually plays.
 *
 * A profile entry is *significant digits* when positive and *decimal places*
 * when zero or negative.  The distinction is about units: significant-digit
 * rounding is scale invariant, so a column stored in seconds and written in
 * microseconds may use it, while a column whose file precision is absolute
 * (a delay quantised to the microsecond, a dwell time to the nanosecond) has
 * to name decimal places in the unit the library holds -- SI.
 *
 * The rounded row is what survives.  That is not a detail: were the original
 * kept, a sequence written and read back would hold different numbers from the
 * one deduplicated here, and the second write would not match the first.
 *
 * ### Two deliberate differences from the Python this replaces
 *
 * The ADC library's shape reference rounds *exactly* rather than to six
 * significant digits.  It is an id, not a measurement, and at six digits a
 * sequence with more than 999999 shapes would have collapsed two different
 * ones onto each other.
 *
 * Negative zero is folded onto zero.  The two are the same number and differ
 * only in the bits a hash looks at, so a phase encode scaled by minus nothing
 * would otherwise be a row of its own.
 *
 * ### Order
 *
 * Bottom up: shapes first, then the events that name them, then the gradient
 * numbering that spans two of those, then each extension's own specification,
 * then the chains that string the specifications together, and last the block
 * table that points at all of it.  Every level is renumbered before anything
 * above it is rewritten.
 *
 * New ids are handed out in order of first appearance, which is the order
 * plain Pulseq would have assigned had it registered each row as it came.
 */

#include "pulseq/sequence.hpp"

#include <cmath>
#include <cstring>
#include <string>
#include <unordered_map>

namespace pulseq
{

    namespace
    {

        /* ============================================================== */
        /*  Rounding                                                      */
        /* ============================================================== */

        enum class Mode
        {
            Copy,
            Significant,
            Fixed
        };

        constexpr int POW_MIN = -340;
        constexpr int POW_MAX = 340;

        /** 10^k for every k a double can hold, so rounding costs no `pow`. */
        const double* power_table()
        {
            static const std::vector<double> table = [] {
                std::vector<double> values(static_cast<size_t>(POW_MAX - POW_MIN + 1));
                for (int k = POW_MIN; k <= POW_MAX; ++k)
                    values[static_cast<size_t>(k - POW_MIN)] = std::pow(10.0, static_cast<double>(k));
                return values;
            }();
            return table.data() - POW_MIN;  // indexable by the exponent itself
        }

        inline double power_of_ten(double exponent, const double* table)
        {
            if (exponent >= static_cast<double>(POW_MIN) && exponent <= static_cast<double>(POW_MAX))
                return table[static_cast<int>(exponent)];
            return std::pow(10.0, exponent);
        }

        /**
         * One value at the precision that decides identity.
         *
         * Halfway cases round to even -- `rint` under the default rounding
         * mode, which is what NumPy's `round` does too, so the two
         * implementations of this pass agree on the ties.
         *
         * Zero is answered directly rather than by softening `log10` with a
         * small epsilon, which is how the NumPy version of this avoids taking
         * the logarithm of nothing.  That epsilon is not harmless: it is a
         * *floor*, so every value below it shares one exponent and rounding to
         * nine significant digits keeps four.  Sequences really do carry such
         * numbers -- a phase that came out of a cosine at pi/2 is 6e-17, and a
         * shape sample can be smaller still -- and there is no reason for the
         * pass that decides two events are the same to also blunt them.
         */
        inline double round_one(double v, Mode mode, double scale, const double* powers)
        {
            if (mode == Mode::Significant)
            {
                if (v == 0.0)
                    return 0.0;
                const double s = power_of_ten(scale - std::ceil(std::log10(std::fabs(v))), powers);
                v = std::rint(v * s) / s;
            }
            else if (mode == Mode::Fixed)
            {
                v = std::rint(v * scale) / scale;
            }
            // Written out rather than left as `v + 0.0`, which an optimiser is
            // entitled to drop.  It is not a no-op: it folds -0.0 onto 0.0.
            return (v == 0.0) ? 0.0 : v;
        }

        /** A rounding profile, expanded per column. */
        class Rounding
        {
        public:
            Rounding(int width, const int* digits, int count) : mode_(width, Mode::Copy),
                                                                scale_(width, 1.0)
            {
                const int named = count < width ? count : width;
                for (int c = 0; c < named; ++c)
                {
                    if (digits[c] > 0)
                    {
                        mode_[c] = Mode::Significant;
                        scale_[c] = static_cast<double>(digits[c]);
                    }
                    else
                    {
                        mode_[c] = Mode::Fixed;
                        scale_[c] = std::pow(10.0, static_cast<double>(-digits[c]));
                    }
                }
            }

            /** Round @p width values in place. */
            void apply(double* values, int width) const
            {
                const double* powers = power_table();
                for (int c = 0; c < width; ++c)
                    values[c] = round_one(values[c], mode_[c], scale_[c], powers);
            }

        private:
            std::vector<Mode> mode_;
            std::vector<double> scale_;
        };

        /** A profile that is one number for every column, however many there are. */
        std::vector<int> uniform(int digit, int width)
        {
            return std::vector<int>(static_cast<size_t>(width), digit);
        }

        /**
         * Round a whole run of values to one profile entry.
         *
         * The ragged libraries -- shapes, and a pTx shim vector -- have no
         * columns to speak of: every entry means the same kind of thing, and
         * how many there are is the coil's business or the waveform's.
         */
        void round_run(double* values, int count, int digit)
        {
            const double* powers = power_table();
            const Mode mode = digit > 0 ? Mode::Significant : Mode::Fixed;
            const double scale = digit > 0 ? static_cast<double>(digit)
                                           : std::pow(10.0, static_cast<double>(-digit));
            for (int i = 0; i < count; ++i)
                values[i] = round_one(values[i], mode, scale, powers);
        }

        /* ============================================================== */
        /*  The distinct rows of a table                                  */
        /* ============================================================== */

        /**
         * An open-addressed index over fixed-width rows, keyed on their bits.
         *
         * Bitwise is the right comparison: these numbers are the file, and two
         * rows that are equal but not identical would be written differently.
         * The hash only decides where probing starts -- every apparent hit is
         * confirmed against the row itself -- so it need not agree with any
         * other implementation's.
         *
         * The table grows from small rather than being sized for the worst
         * case.  A library holds millions of rows drawn from a couple of
         * thousand distinct ones, so the index that answers for them stays in
         * cache for the whole pass; sizing for the input would mean tens of
         * megabytes of mostly empty slots.
         */
        class RowIndex
        {
        public:
            explicit RowIndex(int width) : width_(width), slot_(16, -1), mask_(15) {}

            /** The 0-based code for @p row, appending it if it is new. */
            int32_t claim(const double* row)
            {
                size_t at = static_cast<size_t>(digest(row)) & mask_;
                for (;;)
                {
                    const int32_t held = slot_[at];
                    if (held < 0)
                    {
                        const int32_t issued = count();
                        rows_.insert(rows_.end(), row, row + width_);
                        slot_[at] = issued;
                        if (static_cast<size_t>(issued + 1) * 2 > slot_.size())
                            grow();
                        return issued;
                    }
                    if (std::memcmp(at_row(held), row, static_cast<size_t>(width_) * sizeof(double)) == 0)
                        return held;
                    at = (at + 1) & mask_;
                }
            }

            int32_t count() const { return static_cast<int32_t>(rows_.size()) / width_; }
            const double* at_row(int32_t code) const
            {
                return rows_.data() + static_cast<size_t>(code) * width_;
            }

        private:
            uint64_t digest(const double* row) const
            {
                uint64_t h = 0xcbf29ce484222325ULL;
                for (int c = 0; c < width_; ++c)
                {
                    uint64_t word;
                    std::memcpy(&word, row + c, sizeof(word));
                    h ^= word;
                    h *= 0x100000001b3ULL;
                }
                // FNV's low bits -- the ones an open-addressed table indexes
                // with -- are weak on inputs sharing a suffix, which whole
                // columns of these tables do.
                h ^= h >> 33;
                h *= 0xff51afd7ed558ccdULL;
                h ^= h >> 33;
                return h;
            }

            void grow()
            {
                slot_.assign(slot_.size() * 2, -1);
                mask_ = slot_.size() - 1;
                const int32_t n = count();
                for (int32_t code = 0; code < n; ++code)
                {
                    size_t at = static_cast<size_t>(digest(at_row(code))) & mask_;
                    while (slot_[at] >= 0)
                        at = (at + 1) & mask_;
                    slot_[at] = code;
                }
            }

            int width_;
            std::vector<double> rows_;
            std::vector<int32_t> slot_;
            size_t mask_;
        };

        /** An id column of a library row, and the lookup that renumbers it. */
        struct Remap
        {
            int column;
            const std::vector<int32_t>* lookup;
        };

        /**
         * Renumber one reference.
         *
         * Zero is "no shape" and -1 is Pulseq's half-raster flag; neither is an
         * id and neither moves.  Anything else must name a row that exists --
         * a reference past the end of the library it points into is a corrupt
         * sequence, not something to carry through quietly.
         */
        double renumber(const std::vector<int32_t>& lookup, double value)
        {
            const long id = std::lround(value);
            if (id <= 0)
                return value;
            if (static_cast<size_t>(id) >= lookup.size())
                throw std::out_of_range("event references library row " + std::to_string(id) +
                                        ", which does not exist");
            return static_cast<double>(lookup[static_cast<size_t>(id)]);
        }

        /**
         * Collapse @p table onto its distinct rows.
         *
         * @param uses  RF's `use`, keyed on alongside the row without being
         *              part of it -- it decides identity but is stored beside
         *              the data.  Null for every other library.
         * @return the lookup from old id to new, with index 0 left at 0
         *         because 0 is "no event" wherever a block holds one.
         */
        std::vector<int32_t> collapse(Table& table, const std::vector<int>& digits,
                                      const std::vector<Remap>& remaps = {},
                                      std::vector<char>* uses = nullptr)
        {
            const int width = table.width();
            const int n = table.size();
            std::vector<int32_t> lookup(static_cast<size_t>(n) + 1, 0);
            if (n == 0)
                return lookup;

            const Rounding rounding(width, digits.data(), static_cast<int>(digits.size()));
            const int keyed = width + (uses ? 1 : 0);
            RowIndex index(keyed);
            std::vector<double> key(static_cast<size_t>(keyed));
            // Which old row each surviving one came from, so `uses` can follow.
            std::vector<int32_t> origin;

            for (int id = 1; id <= n; ++id)
            {
                std::memcpy(key.data(), table.row(id), static_cast<size_t>(width) * sizeof(double));
                // Before the rounding, not after: every profile treats a
                // reference column as an integer, so renumbering first cannot
                // change what the rounding produces.
                for (const Remap& remap : remaps)
                    key[static_cast<size_t>(remap.column)] =
                        renumber(*remap.lookup, key[static_cast<size_t>(remap.column)]);
                rounding.apply(key.data(), width);
                if (uses)
                    key[static_cast<size_t>(width)] = static_cast<double>((*uses)[static_cast<size_t>(id) - 1]);

                const int32_t code = index.claim(key.data());
                if (code == static_cast<int32_t>(origin.size()))
                    origin.push_back(id);
                lookup[static_cast<size_t>(id)] = code + 1;
            }

            const int32_t kept = index.count();
            if (uses)
            {
                std::vector<char> surviving(static_cast<size_t>(kept));
                for (int32_t code = 0; code < kept; ++code)
                    surviving[static_cast<size_t>(code)] =
                        (*uses)[static_cast<size_t>(origin[static_cast<size_t>(code)]) - 1];
                uses->swap(surviving);
            }

            table.resize(kept);
            for (int32_t code = 0; code < kept; ++code)
                std::memcpy(table.row(code + 1), index.at_row(code),
                            static_cast<size_t>(width) * sizeof(double));
            return lookup;
        }

        /** The same, for a library of whole numbers -- nothing to round. */
        std::vector<int32_t> collapse_int(IntTable& table)
        {
            const int width = table.width();
            const int n = table.size();
            std::vector<int32_t> lookup(static_cast<size_t>(n) + 1, 0);
            if (n == 0)
                return lookup;

            std::unordered_map<std::string, int32_t> seen;
            std::vector<int32_t> kept;
            kept.reserve(static_cast<size_t>(width));
            const size_t bytes = static_cast<size_t>(width) * sizeof(int32_t);

            for (int id = 1; id <= n; ++id)
            {
                const std::string key(reinterpret_cast<const char*>(table.row(id)), bytes);
                auto found = seen.find(key);
                if (found == seen.end())
                {
                    const int32_t issued = static_cast<int32_t>(kept.size()) / width + 1;
                    kept.insert(kept.end(), table.row(id), table.row(id) + width);
                    seen.emplace(key, issued);
                    lookup[static_cast<size_t>(id)] = issued;
                }
                else
                {
                    lookup[static_cast<size_t>(id)] = found->second;
                }
            }

            const int32_t rows = static_cast<int32_t>(kept.size()) / width;
            table.resize(rows);
            std::memcpy(table.data(), kept.data(), kept.size() * sizeof(int32_t));
            return lookup;
        }

    }  // namespace

    /* ================================================================== */
    /*  The pass                                                          */
    /* ================================================================== */

    void Sequence::remove_duplicates()
    {
        /* -- shapes ---------------------------------------------------- */
        //
        // Compressed samples are written `%.9g`, so nine significant digits is
        // exactly what the file records and rounding to it is lossless as far
        // as a written sequence is concerned.  A shape is keyed on its sample
        // count too: the same compressed run can decompress to two lengths.
        std::vector<int32_t> shape_map(static_cast<size_t>(shapes_.size()) + 1, 0);
        {
            std::unordered_map<std::string, int32_t> seen;
            ShapeLibrary kept;
            std::vector<double> samples;
            for (int id = 1; id <= shapes_.size(); ++id)
            {
                const int count = shapes_.num_compressed(id);
                samples.assign(shapes_.samples(id), shapes_.samples(id) + count);
                round_run(samples.data(), count, 9);

                std::string key(sizeof(int32_t) + samples.size() * sizeof(double), '\0');
                const int32_t length = shapes_.num_uncompressed(id);
                std::memcpy(&key[0], &length, sizeof(length));
                if (!samples.empty())
                    std::memcpy(&key[sizeof(int32_t)], samples.data(),
                                samples.size() * sizeof(double));

                auto found = seen.find(key);
                if (found == seen.end())
                {
                    // A shape still held raw stays raw: it is deduplicated
                    // here and compressed later, which is the whole point of
                    // registering it uncompressed.
                    const int32_t issued =
                        shapes_.is_compressed(id)
                            ? kept.append(length, samples.data(), count)
                            : kept.append_raw(samples.data(), count);
                    seen.emplace(std::move(key), issued);
                    shape_map[static_cast<size_t>(id)] = issued;
                }
                else
                {
                    shape_map[static_cast<size_t>(id)] = found->second;
                }
            }
            shapes_ = std::move(kept);
        }

        /* -- gradients ------------------------------------------------- */
        //
        // The two halves deduplicate apart -- they cannot possibly be equal,
        // and their rows are not even the same width -- and are then given one
        // numbering again, in order of first appearance across both, because
        // that is the numbering the file carries.
        const std::vector<int> grad_digits{6, -6, -6, -6, -6, -6};
        const std::vector<int32_t> arb_map =
            collapse(arb_, grad_digits, {{3, &shape_map}, {4, &shape_map}});
        const std::vector<int32_t> trap_map = collapse(trap_, grad_digits);

        std::vector<int32_t> grad_map(grad_slot_.size() + 1, 0);
        {
            std::unordered_map<int32_t, int32_t> seen;
            std::vector<int32_t> kept;
            for (size_t g = 0; g < grad_slot_.size(); ++g)
            {
                const int32_t slot = grad_slot_[g];
                const int32_t moved =
                    slot > 0 ? trap_map.at(static_cast<size_t>(slot))
                             : -arb_map.at(static_cast<size_t>(-slot));
                auto found = seen.find(moved);
                if (found == seen.end())
                {
                    kept.push_back(moved);
                    const int32_t issued = static_cast<int32_t>(kept.size());
                    seen.emplace(moved, issued);
                    grad_map[g + 1] = issued;
                }
                else
                {
                    grad_map[g + 1] = found->second;
                }
            }
            grad_slot_.swap(kept);
        }

        /* -- RF and ADC ------------------------------------------------ */
        const std::vector<int> rf_digits{6, 0, 0, 0, 6, 6, 6, 6, 6, 6};
        const std::vector<int32_t> rf_map =
            collapse(rf_, rf_digits, {{1, &shape_map}, {2, &shape_map}, {3, &shape_map}}, &rf_use_);

        // num, dwell (ns), delay (us), the four offsets, the phase shape id.
        const std::vector<int> adc_digits{0, -9, -6, 6, 6, 6, 6, 0};
        const std::vector<int32_t> adc_map = collapse(adc_, adc_digits, {{7, &shape_map}});

        /* -- extension specifications ---------------------------------- */
        const std::vector<int32_t> trigger_map = collapse(trigger_, std::vector<int>{0, 0, 9, 9});
        const std::vector<int32_t> label_set_map = collapse_int(label_set_);
        const std::vector<int32_t> label_inc_map = collapse_int(label_inc_);
        const std::vector<int32_t> rotation_map = collapse(rotation_, uniform(9, ROTATION_WIDTH));

        // A shim vector is as wide as the coil has channels, so its profile is
        // one number repeated rather than a fixed list.
        std::vector<int32_t> rf_shim_map(static_cast<size_t>(rf_shim_.size()) + 1, 0);
        {
            std::unordered_map<std::string, int32_t> seen;
            RaggedTable kept;
            std::vector<double> values;
            for (int id = 1; id <= rf_shim_.size(); ++id)
            {
                const int count = rf_shim_.length(id);
                values.assign(rf_shim_.row(id), rf_shim_.row(id) + count);
                round_run(values.data(), count, 9);

                const std::string key(reinterpret_cast<const char*>(values.data()),
                                      values.size() * sizeof(double));
                auto found = seen.find(key);
                if (found == seen.end())
                {
                    const int32_t issued = kept.append(values.data(), count);
                    seen.emplace(key, issued);
                    rf_shim_map[static_cast<size_t>(id)] = issued;
                }
                else
                {
                    rf_shim_map[static_cast<size_t>(id)] = found->second;
                }
            }
            rf_shim_ = std::move(kept);
        }

        // A soft delay ends in a string, so it is made unique by what it says
        // rather than by rounding it to the precision the file writes.
        std::vector<int32_t> soft_delay_map(soft_delays_.size() + 1, 0);
        {
            std::unordered_map<std::string, int32_t> seen;
            std::vector<SoftDelay> kept;
            for (size_t i = 0; i < soft_delays_.size(); ++i)
            {
                const SoftDelay& row = soft_delays_[i];
                std::string key(sizeof(int32_t) + 2 * sizeof(double), '\0');
                std::memcpy(&key[0], &row.num, sizeof(row.num));
                std::memcpy(&key[sizeof(int32_t)], &row.offset, sizeof(row.offset));
                std::memcpy(&key[sizeof(int32_t) + sizeof(double)], &row.factor, sizeof(row.factor));
                key += row.hint;

                auto found = seen.find(key);
                if (found == seen.end())
                {
                    kept.push_back(row);
                    const int32_t issued = static_cast<int32_t>(kept.size());
                    seen.emplace(std::move(key), issued);
                    soft_delay_map[i + 1] = issued;
                }
                else
                {
                    soft_delay_map[i + 1] = found->second;
                }
            }
            soft_delays_.swap(kept);
        }

        /* -- extension chains ------------------------------------------ */
        //
        // A chain is canonicalised once per *node*, not once per block: what
        // hangs off a node is decided by that node alone, so a scan whose every
        // TR carries the same two labels canonicalises two chains rather than
        // repeating the walk for each of a million blocks.
        //
        // A node's `next` is always an earlier node -- chains are built tail
        // first -- so ascending id order reaches every tail before its head.
        std::vector<int32_t> new_head(static_cast<size_t>(extensions_.size()) + 1, 0);
        if (!extensions_.empty())
        {
            std::unordered_map<int32_t, const std::vector<int32_t>*> ref_maps;
            const std::pair<const char*, const std::vector<int32_t>*> bindings[] = {
                {"TRIGGERS", &trigger_map}, {"LABELSET", &label_set_map},
                {"LABELINC", &label_inc_map}, {"ROTATIONS", &rotation_map},
                {"RF_SHIMS", &rf_shim_map}, {"DELAYS", &soft_delay_map},
            };
            for (const auto& binding : bindings)
            {
                const int id = find_extension_type_id(binding.first);
                if (id)
                    ref_maps.emplace(static_cast<int32_t>(id), binding.second);
            }

            IntTable fresh(EXTENSION_WIDTH);
            std::map<std::array<int32_t, EXTENSION_WIDTH>, int> cache;
            for (int node = 1; node <= extensions_.size(); ++node)
            {
                const int32_t* row = extensions_.row(node);
                const int32_t type_id = row[0];
                const int32_t next = row[2];
                if (next >= node)
                    throw std::runtime_error(
                        "extension chain node " + std::to_string(node) + " points forward to " +
                        std::to_string(next) + "; chains have to be built tail first");
                const int32_t tail = next >= 1 ? new_head[static_cast<size_t>(next)] : 0;

                int32_t ref = row[1];
                auto lookup = ref_maps.find(type_id);
                if (lookup != ref_maps.end())
                    ref = static_cast<int32_t>(
                        renumber(*lookup->second, static_cast<double>(ref)));
                if (ref == 0)
                {
                    // Nothing left to point at: the block inherits the rest of
                    // the chain and this link disappears.
                    new_head[static_cast<size_t>(node)] = tail;
                    continue;
                }

                const std::array<int32_t, EXTENSION_WIDTH> key{type_id, ref, tail};
                auto found = cache.find(key);
                if (found == cache.end())
                {
                    const int id = fresh.append(key.data());
                    cache.emplace(key, id);
                    new_head[static_cast<size_t>(node)] = static_cast<int32_t>(id);
                }
                else
                {
                    new_head[static_cast<size_t>(node)] = static_cast<int32_t>(found->second);
                }
            }
            extensions_ = std::move(fresh);
            chain_index_ = std::move(cache);
        }

        /* -- the block table ------------------------------------------- */
        //
        // The one structure whose size is the scan's rather than the event
        // vocabulary's, so it is renumbered as six column lookups over one
        // integer array instead of a dictionary read per block.
        // Checked rather than indexed blindly: a block naming an event that was
        // never registered is a corrupt sequence, and the bounds test costs
        // nothing beside the six scattered lookups it guards.  Saying which
        // block and which library is the whole value of noticing at all.
        const size_t blocks = durations_.size();
        const auto follow = [](const std::vector<int32_t>& map, int32_t id, const char* what,
                               size_t block) {
            if (id < 0 || static_cast<size_t>(id) >= map.size())
                throw std::out_of_range("block " + std::to_string(block + 1) + " names " + what +
                                        " " + std::to_string(id) + ", which does not exist");
            return map[static_cast<size_t>(id)];
        };
        for (size_t b = 0; b < blocks; ++b)
        {
            int32_t* row = blocks_.data() + b * BLOCK_WIDTH;
            row[0] = follow(rf_map, row[0], "RF event", b);
            row[1] = follow(grad_map, row[1], "gradient", b);
            row[2] = follow(grad_map, row[2], "gradient", b);
            row[3] = follow(grad_map, row[3], "gradient", b);
            row[4] = follow(adc_map, row[4], "ADC event", b);
            row[5] = follow(new_head, row[5], "extension chain", b);
        }
    }

}  // namespace pulseq
